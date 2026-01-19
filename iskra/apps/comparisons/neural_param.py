# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import iskra.sparse as sp
from iskra import dec
from iskra.geometry import triangle_areas, triangle_coordinate_system
from iskra.geometry.cotan_weights import cotan_weights_intrinsic
from iskra.geometry.volume import edge_lengths
from iskra.mesh import Mesh
from iskra.sparse_linalg import default_solver, eigsh
from iskra.topology import boundary, face_index, get_subfaces


def triangle_to_local(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = face_index(verts, faces)
    _, t, b = triangle_coordinate_system(triangles)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    world_to_local = torch.stack([t, b], -2)
    local = world_to_local @ edge_vecs.mT
    return local


def uv_local(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # Do not project on a local coordinate frame because that will
    # leave us not knowing if there is a flip or not!
    triangles = face_index(uv, faces)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    return edge_vecs.mT


def symmetric_dirichlet(
    rest_local: torch.Tensor, param_local: torch.Tensor, rest_areas: torch.Tensor
) -> torch.Tensor:
    jac = param_local @ torch.linalg.inv(rest_local)
    energy_fwd = (jac**2).sum((-2, -1))
    energy_bwd = (torch.linalg.inv(jac) ** 2).sum((-2, -1))
    energy = rest_areas * (energy_fwd + energy_bwd)

    is_flipped = torch.linalg.det(param_local.mT) <= 0
    # if is_flipped.count_nonzero() > 0:
    #     print(f"Flipped {is_flipped.count_nonzero()} triangles.")
    energy[is_flipped] = float("inf")
    return energy


def vertex_area_matrix(
    n_vertices: int, faces: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    bdr_edges = boundary(faces)
    bdr_edges_bwd = bdr_edges[:, (1, 0)]
    n_bdr_edges = bdr_edges.shape[0]
    idcs_i = torch.cat([bdr_edges, bdr_edges_bwd + n_vertices], -2).flatten(-2, -1)
    idcs_j = torch.cat([bdr_edges_bwd + n_vertices, bdr_edges], -2).flatten(-2, -1)
    values = torch.tensor([0.25, -0.25], device=faces.device, dtype=dtype)
    values = values[None, :].expand(2 * n_bdr_edges, -1).flatten(-2, -1)
    return torch.sparse_coo_tensor(
        torch.stack([idcs_i, idcs_j], -2), values, size=[2 * n_vertices, 2 * n_vertices]
    )


def write_obj_with_uv(
    path: Path, verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor
):
    import trimesh

    obj_text = trimesh.exchange.obj.export_obj(
        trimesh.Trimesh(
            verts.detach().cpu().numpy(),
            faces.to(torch.int32).detach().cpu().numpy(),
            visual=trimesh.visual.texture.TextureVisuals(uv=uv.detach().cpu().numpy()),
        ),
        include_texture=True,
    )
    with path.open("w") as f:
        f.write(obj_text)


def compute_scp(
    lengths: torch.Tensor,
    face_edges: torch.Tensor,
    d_01: torch.Tensor,
    va_mat: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    # weight = cotan_weights_intrinsic(lengths, face_edges, clamp_min=1e-6)
    # print(weight.shape, lengths.shape)
    weight = lengths  # torch.exp(-0.5 * lengths)
    lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(weight), d_01)).coalesce()
    lhs = (sp.repdiag(lap, 2) - 2 * va_mat).coalesce()
    # print(lhs.to_dense())
    # print(torch.linalg.lu_factor(rhs.to_dense()))
    evals, evecs = eigsh(
        lhs,
        M=rhs,
        k=3,
        sigma=-1e-12,
        bwd_method="dodik-fixedpoint",
        bwd_max_iter=25,
    )
    # print(f"Difference between eigenvalues: {evals[1] - evals[2]}")
    uv_opt = evecs[:, 0:1].reshape(2, -1).mT
    return uv_opt


def main(mesh_path: Path):
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    edges, face_edges, face_edge_sign = get_subfaces(faces)
    d_01 = dec.d_01(faces, dtype=dtype)
    lines = face_index(verts, edges)
    lengths = edge_lengths(lines)
    # cot = cotan_weights_intrinsic(lengths, face_edges, clamp_min=1e-8)
    # lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(cot), d_01)).coalesce()
    bdr_idx = boundary(faces).flatten().unique()

    vertex_area = vertex_area_matrix(mesh.n_vertices, mesh.faces, dtype=dtype)
    bdr_ii = torch.stack([bdr_idx, bdr_idx], 0)
    bdr_val = torch.ones(bdr_ii.shape[1], dtype=dtype, device=device)
    rhs_block = torch.sparse_coo_tensor(
        bdr_ii, bdr_val, size=[mesh.n_vertices, mesh.n_vertices]
    )
    rhs = sp.repdiag(rhs_block, 2)

    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))

    lr = 10
    # verts_opt = torch.nn.Parameter(verts.clone())
    # optimizer = torch.optim.SGD([verts_opt], lr=lr)
    lengths_opt = torch.nn.Parameter(lengths.clone())
    optimizer = torch.optim.SGD([lengths_opt], lr=lr)

    # _, mass = dec.laplacian(verts, faces, clamp_min=0.0)
    lengths = edge_lengths(face_index(verts, edges))
    # cot = cotan_weights_intrinsic(lengths, face_edges, clamp_min=1e-8)
    # lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(cot), d_01)).coalesce()
    # h1_solver = default_solver(mass + 0.8 * lap)

    def length_map(length: torch.Tensor):
        return torch.exp(-0.5 * length**2) + 1e-6

    # uv_opt = compute_scp(
    #     edge_lengths(face_index(verts_opt, edges)), face_edges, d_01, vertex_area, rhs
    # )
    uv_opt = compute_scp(lengths_opt, face_edges, d_01, vertex_area, rhs)

    results_dir = Path.home() / "Dropbox" / "Results" / "iskra" / "scp" / mesh_path.stem
    results_dir.mkdir(exist_ok=True, parents=True)

    write_obj_with_uv(results_dir / "scp.obj", verts, faces, uv_opt)

    def step_fn():
        uv_opt = compute_scp(
            length_map(lengths_opt), face_edges, d_01, vertex_area, rhs
        )
        param_local = uv_local(uv_opt, faces)
        energy = symmetric_dirichlet(rest_local, param_local, rest_areas)
        energy.mean().backward()
        print(f"Loss: {energy.mean().detach().cpu().item()}")
        with torch.no_grad():
            if lengths_opt.grad is None:
                raise RuntimeError("verts_var.grad is None!")
            if not torch.isfinite(lengths_opt.grad).all():
                raise RuntimeError("verts_var.grad not finite!")
            # lengths_opt.grad = h1_solver(mass @ lengths_opt.grad)
            # lengths_opt.grad -= lengths_opt.grad.mean(0, keepdim=True)
            # print(lengths_opt.grad.min(), lengths_opt.grad.max())

            print(length_map(lengths_opt - lr * lengths_opt.grad).min())

            new_uv = compute_scp(
                length_map(lengths_opt - lr * lengths_opt.grad),
                face_edges,
                d_01,
                vertex_area,
                rhs,
            )
            param_local = uv_local(new_uv, faces)
            energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
            n_shrinks = 0
            while energy_new.mean() > energy.mean():
                lengths_opt.grad *= 0.1
                new_uv = compute_scp(
                    length_map(lengths_opt - lr * lengths_opt.grad),
                    face_edges,
                    d_01,
                    vertex_area,
                    rhs,
                )
                param_local = uv_local(new_uv, faces)
                energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
                n_shrinks += 1
            # with torch.no_grad():
            #     if verts_opt.grad is None:
            #         raise RuntimeError("verts_var.grad is None!")
            #     if not torch.isfinite(verts_opt.grad).all():
            #         raise RuntimeError("verts_var.grad not finite!")
            #     verts_opt.grad = h1_solver(mass @ verts_opt.grad)
            #     verts_opt.grad -= verts_opt.grad.mean(0, keepdim=True)
            #     print(verts_opt.grad.min(), verts_opt.grad.max())

            #     energy_new = symmetric_dirichlet(
            #         rest_local,
            #         uv_local(
            #             compute_scp(
            #                 edge_lengths(
            #                     face_index(verts_opt - lr * verts_opt.grad, edges)
            #                 ),
            #                 face_edges,
            #                 d_01,
            #                 vertex_area,
            #                 rhs,
            #             ),
            #             faces,
            #         ),
            #         rest_areas,
            #     )
            #     n_shrinks = 0
            #     while energy_new.mean() > energy.mean():
            #         verts_opt.grad *= 0.1
            #         energy_new = symmetric_dirichlet(
            #             rest_local,
            #             uv_local(
            #                 compute_scp(
            #                     edge_lengths(
            #                         face_index(verts_opt - lr * verts_opt.grad, edges)
            #                     ),
            #                     face_edges,
            #                     d_01,
            #                     vertex_area,
            #                     rhs,
            #                 ),
            #                 faces,
            #             ),
            #             rest_areas,
            #         )
            #         n_shrinks += 1
            if n_shrinks > 0:
                print(f"Shrunk the learning rate {n_shrinks} times.")

        return energy

    for _ in range(500):
        optimizer.zero_grad()
        step_fn()
        optimizer.step()

    write_obj_with_uv(results_dir / "optimized.obj", verts, faces, uv_opt)


if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates an inverse SCP parameterization.")
    parser.add_argument("mesh_path", type=Path, help="The path of the mesh to load.")
    args = parser.parse_args()
    mesh_path = args.mesh_path
    main(mesh_path)

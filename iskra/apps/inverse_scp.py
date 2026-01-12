# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import numpy as np
import torch

from iskra.dec import laplacian
from iskra.geometry import triangle_areas, triangle_coordinate_system
from iskra.mesh import Mesh
from iskra.sparse import diag, repdiag, torch_to_scipy
from iskra.sparse_linalg import _linear_solver_fn, eigsh
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
    rest_local: torch.Tensor, param_local: torch.Tensor
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates an inverse SCP parameterization.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, dtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    edges, face_edges, face_edge_sign = get_subfaces(faces)
    bdr_idx = boundary(faces).flatten().unique()

    vertex_area = vertex_area_matrix(mesh.n_vertices, mesh.faces, dtype=dtype)
    bdr_ii = torch.stack([bdr_idx, bdr_idx], 0)
    bdr_val = torch.ones(bdr_ii.shape[1], dtype=dtype, device=device)
    rhs_block = torch.sparse_coo_tensor(
        bdr_ii, bdr_val, size=[mesh.n_vertices, mesh.n_vertices]
    )
    rhs = repdiag(rhs_block, 2)

    def compute_scp(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        lap, _ = laplacian(verts, faces, clamp_min=1e-8)
        lhs = repdiag(lap, 2) - 2 * vertex_area
        evals, evecs = eigsh(lhs, M=rhs, k=4, sigma=-1e-12, adjoint="individual")
        print(f"Difference between eigenvalues: {evals[2] - evals[3]}")
        uv_opt = evecs[:, 2:3].reshape(2, -1).mT
        return uv_opt

    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))

    verts_opt = torch.nn.Parameter(verts.clone())
    lr = 0.01
    optimizer = torch.optim.SGD([verts_opt], lr=lr)
    lap, mass = laplacian(verts, faces, clamp_min=0.0)
    h1_solver = _linear_solver_fn(mass + 0.8 * lap)
    uv_opt = compute_scp(verts_opt, faces)

    def step_fn():
        uv_opt = compute_scp(verts_opt, faces)
        param_local = uv_local(uv_opt, faces)
        energy = symmetric_dirichlet(rest_local, param_local)
        energy.mean().backward()
        print(energy.mean())
        with torch.no_grad():
            if verts_opt.grad is None:
                raise RuntimeError("verts_var.grad is None!")
            if not torch.isfinite(verts_opt.grad).all():
                raise RuntimeError("verts_var.grad not finite!")
            verts_opt.grad = h1_solver(mass @ verts_opt.grad)
            verts_opt.grad -= verts_opt.grad.mean(0, keepdim=True)
            print(verts_opt.grad.min(), verts_opt.grad.max())

            # energy_new = symmetric_dirichlet(
            #     rest_local,
            #     uv_local(compute_scp(verts_opt - lr * verts_opt.grad, faces), faces),
            # )
            # n_shrinks = 0
            # while energy_new.mean() > energy.mean():
            #     verts_opt.grad *= 0.1
            #     energy_new = symmetric_dirichlet(
            #         rest_local,
            #         uv_local(
            #             compute_scp(verts_opt - lr * verts_opt.grad, faces), faces
            #         ),
            #     )
            #     n_shrinks += 1
            # if n_shrinks > 0:
            #     print(f"Shrunk the learning rate {n_shrinks} times.")

        return energy

    with torch.no_grad():
        uv_opt = compute_scp(verts_opt, faces)
        param_local = uv_local(uv_opt, faces)
        # energy = symmetric_dirichlet(rest_local, param_local)
    try:
        import polyscope as ps

        ps.init()
        ps_mesh_init = ps.register_surface_mesh(
            "Mesh SCP", verts.numpy() + np.array([1.0, 0.0, 0.0]), faces.numpy()
        )
        print(uv_opt.shape, verts.shape)
        ps_mesh_init.add_parameterization_quantity(
            "Param SCP", uv_opt.detach().numpy(), enabled=True
        )
        ps_mesh = ps.register_surface_mesh("Mesh", verts.numpy(), faces.numpy())
        ps_mesh.add_scalar_quantity(
            "det(J)",
            torch.linalg.det(rest_local).numpy(),
            defined_on="faces",
            enabled=True,
        )
        ps_mesh.add_parameterization_quantity(
            "Param SymmDir", uv_opt.detach().numpy(), enabled=True
        )
        ps_edges = ps.register_curve_network(
            "edges", verts.numpy(), edges.numpy(), enabled=False, radius=0.01
        )

        ps_param_mesh = ps.register_surface_mesh(
            "UV Mesh", uv_opt.detach().numpy(), faces.numpy(), edge_width=1
        )
        # ps_param_mesh.add_scalar_quantity(
        #     "energy", energy.detach().numpy(), defined_on="faces", enabled=True
        # )
        ve = face_index(verts_opt, edges).detach()
        ps_edges.add_scalar_quantity(
            "metric",
            torch.linalg.vector_norm(ve[..., 1, :] - ve[..., 0, :], dim=-1).numpy(),
            defined_on="edges",
        )

        optimizing = False

        def callback():
            global optimizing

            if ps.imgui.Button("Step" if not optimizing else "Step"):
                #     optimizing = not optimizing
                # if optimizing:
                for _ in range(1):
                    optimizer.zero_grad()
                    energy = step_fn()
                    optimizer.step()
                    uv_opt = compute_scp(verts_opt, faces)
                    param_local = uv_local(uv_opt, faces)
                    energy = symmetric_dirichlet(rest_local, param_local)

                    # ps_mesh.update_vertex_positions(verts_opt.detach().numpy())
                    ps_param_mesh.update_vertex_positions(uv_opt.detach().numpy())
                    ps_param_mesh.add_scalar_quantity(
                        "energy",
                        energy.detach().numpy(),
                        defined_on="faces",
                        enabled=True,
                    )
                    ps_mesh.add_parameterization_quantity(
                        "rand param", uv_opt.detach().numpy(), enabled=True
                    )
                    ve = face_index(verts_opt, edges).detach()
                    ps_edges.add_scalar_quantity(
                        "metric",
                        torch.linalg.vector_norm(
                            ve[..., 1, :] - ve[..., 0, :], dim=-1
                        ).numpy(),
                        defined_on="edges",
                    )

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )

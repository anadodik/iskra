# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from iskra import dec
from iskra.geometry import triangle_areas
from iskra.geometry.cotan_weights import cotan_weights, cotan_weights_intrinsic
from iskra.geometry.volume import edge_lengths
from iskra.harmonic_embedding import HarmonicEmbedding
from iskra.mesh import Mesh
from iskra.mlp import MLP, NetworkConfig
from iskra.parameterization import (
    boundary_matrix,
    scp_solve_from_weights,
    symmetric_dirichlet_energy,
    triangle_to_local,
    uv_local,
    vertex_area_matrix,
)
from iskra.sparse_linalg import default_solver
from iskra.topology import face_index, get_subfaces


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


def main(mesh_path: Path, ckpt_path: Path | None = None, ckpt_eval: bool = False):
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    edges, face_edges, face_edge_sign = get_subfaces(faces)
    d_01 = dec.d_01(faces, dtype=dtype)
    lines = face_index(verts, edges)
    # edge_vec = lines[:, 1, :] - lines[:, 0, :]
    midpoints = lines.mean(-2)
    # lengths = edge_lengths(lines)
    # cot = cotan_weights_intrinsic(lengths, face_edges, clamp_min=1e-8)
    # lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(cot), d_01)).coalesce()
    vertex_area = vertex_area_matrix(mesh.n_vertices, faces, dtype=dtype)
    boundary_mat = boundary_matrix(mesh.n_vertices, faces, dtype=dtype)

    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))
    network_config = NetworkConfig(
        otype="PyTorchMLP",
        activation="ReLU",
        encoding_config=None,
        n_neurons=64,
        n_hidden_layers=3,
    )
    n_harmonic_funcs = 4
    encoding = HarmonicEmbedding(n_harmonic_functions=n_harmonic_funcs).to(
        device=device, dtype=dtype
    )
    mlp = MLP(2 * 3 * n_harmonic_funcs + 3, 3, network_config).to(
        device=device, dtype=dtype
    )

    # verts_opt = torch.nn.Parameter(verts.clone())
    def net_forward(x):
        return torch.tanh(mlp(encoding(x)))

    def forward(x):
        # scale = 10
        # mlp_out = scale * torch.sigmoid(1 / scale * mlp(encoding(x)))
        # # mlp_out = torch.tanh(mlp(encoding(x)))
        # metric = mlp_out.reshape(-1, 3, 3) @ mlp_out.reshape(-1, 3, 3).mT
        # metric = metric + torch.eye(3, device=device, dtype=dtype)[None, :, :]
        # length_sq = (edge_vec[..., None, :] @ metric @ edge_vec[..., :, None]).squeeze(
        #     -1
        # )
        # mlp_out = torch.sigmoid(mlp(encoding(x))) + 0.1
        # mlp_out = (mlp(encoding(x)) ** 2 + 0.1).clamp_max(10.0)
        # length_sq = mlp_out.squeeze(-1) * (lengths**2)
        # print(length.min(), length.max())
        # length_sq = (lengths_opt.clamp(0.1, 10.0) * lengths) ** 2
        # weight = torch.exp(-0.5 * length_sq)
        mlp_out = net_forward(x)
        weight = cotan_weights_intrinsic(
            edge_lengths(face_index(verts + mlp_out.reshape(-1, 3), edges)),
            face_edges,
            clamp_min=0,
        )
        # print(weight.min(), weight.max())
        return weight

    if ckpt_path is not None:
        mlp.load_state_dict(torch.load(ckpt_path, weights_only=True))
        if ckpt_eval:
            cot = forward(verts)
            uv = scp_solve_from_weights(d_01, cot, vertex_area, boundary_mat)

            write_obj_with_uv(
                ckpt_path.parent / f"{mesh_path.stem}_from_ckpt.obj",
                verts,
                faces,
                uv,
            )
            write_obj_with_uv(
                ckpt_path.parent / f"{mesh_path.stem}_deformed_from_ckpt.obj",
                verts + net_forward(verts).reshape(-1, 3),
                faces,
                uv,
            )

            lengths = edge_lengths(face_index(verts, edges))
            cot = cotan_weights_intrinsic(lengths, face_edges)
            uv = scp_solve_from_weights(d_01, cot, vertex_area, boundary_mat)
            write_obj_with_uv(
                ckpt_path.parent / f"{mesh_path.stem}_scp.obj", verts, faces, uv
            )
            exit()

    # optimizer = torch.optim.SGD([verts_opt], lr=lr)
    # lengths_opt = torch.nn.Parameter(torch.ones_like(lengths.clone()))
    # optimizer = torch.optim.SGD([lengths_opt], lr=lr)
    if mesh_path.stem == "hand_lowres":
        lr = 2e-3
    elif mesh_path.stem == "cowhead":
        lr = 1e-3
    else:
        lr = 1e-3
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    # optimizer = torch.optim.LBFGS(
    #     mlp.parameters(), lr=1e-2, line_search_fn="strong_wolfe"
    # )
    # optimizer = torch.optim.Adam([lengths_opt], lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600], gamma=0.5
    )

    lap, mass = dec.laplacian(verts, faces, clamp_min=0.0)
    h1_solver = default_solver(mass + 0.8 * lap)

    # uv_opt = compute_scp(
    #     edge_lengths(face_index(verts_opt, edges)), face_edges, d_01, vertex_area, rhs
    # )

    results_dir = Path.cwd() / "results" / "iskra" / "scp" / mesh_path.stem
    results_dir.mkdir(exist_ok=True, parents=True)

    lengths = edge_lengths(face_index(verts, edges))
    cot = cotan_weights_intrinsic(lengths, face_edges)
    uv_opt = scp_solve_from_weights(d_01, cot, vertex_area, boundary_mat)
    write_obj_with_uv(results_dir / "scp.obj", verts, faces, uv_opt)

    cot = forward(verts)
    uv_opt = scp_solve_from_weights(d_01, cot, vertex_area, boundary_mat)
    param_local = uv_local(uv_opt, faces)
    # generator = torch.Generator(device=device)
    # generator.manual_seed(620)

    def step_fn(step: int):
        # lengths = edge_lengths(face_index(verts_opt, edges))
        # cot = cotan_weights_intrinsic(lengths, face_edges)
        cot = forward(verts)
        uv_opt = scp_solve_from_weights(d_01, cot, vertex_area, boundary_mat)
        param_local = uv_local(uv_opt, faces)
        energy = symmetric_dirichlet_energy(rest_local, param_local, rest_areas)
        # energy.mean().backward()
        # if step % 10 == 0:
        # with torch.no_grad():
        #     # if lengths_opt.grad is None:
        #     #     raise RuntimeError("verts_var.grad is None!")
        #     # if not torch.isfinite(lengths_opt.grad).all():
        #     #     raise RuntimeError("verts_var.grad not finite!")
        #     # lengths_opt.grad = h1_solver(mass @ lengths_opt.grad)
        #     # lengths_opt.grad -= lengths_opt.grad.mean(0, keepdim=True)
        #     # print(lengths_opt.grad.min(), lengths_opt.grad.max())

        #     print(length_map(lengths_opt - lr * lengths_opt.grad).min())

        #     new_uv = compute_scp(
        #         length_map(lengths_opt - lr * lengths_opt.grad),
        #         face_edges,
        #         d_01,
        #         vertex_area,
        #         rhs,
        #     )
        #     param_local = uv_local(new_uv, faces)
        #     energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
        #     n_shrinks = 0
        #     while energy_new.mean() > energy.mean():
        #         lengths_opt.grad *= 0.1
        #         new_uv = compute_scp(
        #             length_map(lengths_opt - lr * lengths_opt.grad),
        #             face_edges,
        #             d_01,
        #             vertex_area,
        #             rhs,
        #         )
        #         param_local = uv_local(new_uv, faces)
        #         energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
        #         n_shrinks += 1
        # with torch.no_grad():
        #     if verts_opt.grad is None:
        #         raise RuntimeError("verts_var.grad is None!")
        #     if not torch.isfinite(verts_opt.grad).all():
        #         raise RuntimeError("verts_var.grad not finite!")
        #     verts_opt.grad = h1_solver(mass @ verts_opt.grad)
        #     verts_opt.grad -= verts_opt.grad.mean(0, keepdim=True)
        #     # print(verts_opt.grad.min(), verts_opt.grad.max())

        #     # lengths = edge_lengths(face_index(verts_opt - lr * verts_opt.grad, edges))
        #     # cot = cotan_weights_intrinsic(lengths, face_edges)
        #     cot = forward(verts)
        #     uv_opt = compute_scp(cot, d_01, vertex_area, rhs)
        #     param_local = uv_local(uv_opt, faces)
        #     energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
        #     n_shrinks = 0
        #     while energy_new.mean() > energy.mean():
        #         verts_opt.grad *= 0.1
        #         lengths = edge_lengths(
        #             face_index(verts_opt - lr * verts_opt.grad, edges)
        #         )
        #         cot = cotan_weights_intrinsic(lengths, face_edges)
        #         uv_opt = compute_scp(cot, d_01, vertex_area, rhs)
        #         param_local = uv_local(uv_opt, faces)
        #         energy_new = symmetric_dirichlet(rest_local, param_local, rest_areas)
        #         n_shrinks += 1
        #     if n_shrinks > 0:
        #         print(f"Shrunk the learning rate {n_shrinks} times.")

        return energy.mean()

    max_steps = 100
    for step in range(2_000):
        optimizer.zero_grad()
        energy = step_fn(step)
        print(f"Step {step} | Loss: {energy.detach().cpu().item()}")
        energy.backward()
        optimizer.step(lambda: step_fn(step))
        # lr_scheduler.step()
        if step == 250:
            optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)
        if step % 50 == 0 or step == max_steps - 1:
            uv_opt = scp_solve_from_weights(
                d_01, forward(verts), vertex_area, boundary_mat
            )
            write_obj_with_uv(
                results_dir / f"optimized_{step}.obj", verts, faces, uv_opt
            )
            write_obj_with_uv(
                results_dir / f"deformed_{step}.obj",
                verts + net_forward(verts).reshape(-1, 3),
                faces,
                uv_opt,
            )
            torch.save(mlp.state_dict(), results_dir / f"model_weights_{step}.pth")


if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates an inverse SCP parameterization.")
    parser.add_argument("mesh_path", type=Path, help="The path of the mesh to load.")
    parser.add_argument("--ckpt_path", type=Path, default=None, help="Checkpoint path.")
    parser.add_argument(
        "--ckpt_eval", action="store_true", help="Eval with checkpoint and exit."
    )
    args = parser.parse_args()
    mesh_path = args.mesh_path
    main(mesh_path, args.ckpt_path, args.ckpt_eval)

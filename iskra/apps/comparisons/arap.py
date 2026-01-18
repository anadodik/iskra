# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import gc
import json
import os
import platform
import resource
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import numpy as np
import psutil
import torch

# torch.set_num_threads(32)
# torch.set_num_interop_threads(32)
torch.set_printoptions(linewidth=200, sci_mode=False)

import theseus as th

import iskra.sparse as sp
import iskra.sparse_linalg as spla
from iskra.adjoint import make_solver_layer
from iskra.dec import d_01, d_10, laplacian, laplacian_from_weights
from iskra.geometry import cotan_weights
from iskra.logging.logging import getLogger
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block, profile_fn
from iskra.signed_svd import closest_rot_3x3
from iskra.topology import (
    boundary,
    edge_to_vertex_adjacency,
    face_index,
    get_subfaces,
    reduce_on_subface,
    vertex_adjacency,
)

LOGGER = getLogger(__name__)


@profile_fn
def arap_step(
    verts_deformed,
    verts_rest,
    cots,
    vert_vert,
    lap,
    bc_idx,
    bc_vals,
    solver: spla.SolverT | None = None,
):
    n_vertices = verts_rest.shape[0]

    with profile_block("covs"):
        lines = face_index(verts_rest, vert_vert)
        vecs = lines[..., 1, :] - lines[..., 0, :]
        lines_deformed = face_index(verts_deformed, vert_vert)
        vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
        covs = cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]
        vert_covs = reduce_on_subface(covs, vert_vert[:, 0:1], n_vertices, "sum")

    with profile_block("svd"):
        vert_rot = closest_rot_3x3(vert_covs)

    with profile_block("rotation"):
        halfedge_rot = face_index(vert_rot.mT, vert_vert).mean(1)
        rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]
        rhs = reduce_on_subface(
            rotated_halfedge_vecs, vert_vert[:, 0:1], n_vertices, "sum"
        )

    with profile_block("solve"):
        verts_deformed = spla.min_quadratic_energy(
            lap, -rhs, bc_idx, bc_vals, solver=solver
        )[1]
    return verts_deformed, torch.zeros_like(verts_deformed[:, 0])


# def arap_step(
#     verts_deformed: torch.Tensor,
#     verts: torch.Tensor,
#     cots: torch.Tensor,
#     halfedges: torch.Tensor,
#     lap: torch.Tensor,
#     bc_idx: torch.Tensor,
#     bc_vals: torch.Tensor,
#     solver: spla.SolverT | None = None,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     n_vertices = verts.shape[0]
#     lines = face_index(verts, halfedges)
#     vecs = lines[..., 1, :] - lines[..., 0, :]
#     lines_deformed = face_index(verts_deformed, halfedges)
#     vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
#     halfedge_covs = (
#         cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]
#     )
#     vert_covs = reduce_on_subface(halfedge_covs, halfedges[:, 0:1], n_vertices, "sum")
#     # vert_u, _, vert_vt = signed_svd(vert_covs)
#     # vert_rot = vert_vt.mT @ vert_u.mT
#     vert_rot = closest_rot_3x3(vert_covs).mT
#     # Uncomment to debug SVD:
#     # vert_rot = vert_covs * 0.0 + torch.eye(3, dtype=vert_u.dtype)[None, :, :].expand(
#     #     n_vertices, -1, -1
#     # )
#     assert (torch.linalg.det(vert_rot) > 0).all()
#     # Following lines are energy only:
#     halfedge_vert_rot = face_index(vert_rot, halfedges)[:, 0, ...]
#     diff = vecs_deformed - (halfedge_vert_rot @ vecs[..., None])[..., 0]
#     weighted_dist = cots * torch.linalg.vector_norm(diff, dim=-1, ord=2) ** 2
#     vert_energy = reduce_on_subface(weighted_dist, halfedges[:, 0:1], n_vertices, "sum")
#     # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
#     halfedge_rot = face_index(vert_rot, halfedges).mean(1)
#     rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]
#     rhs = reduce_on_subface(rotated_halfedge_vecs, halfedges[:, 0:1], n_vertices, "sum")
#     verts_deformed = spla.min_quadratic_energy(
#         lap, -rhs, bc_idx, bc_vals, solver=solver
#     )[1]
#     return verts_deformed, vert_energy


def build_arap_layer(
    verts_rest: torch.Tensor,  # (N,3)
    edges: torch.Tensor,  # (E,2) each row [i,j]
    constraint_type: torch.Tensor,  # (N,) 0 free, 1 displaced, 2 fixed
    handle_offsets: torch.Tensor,  # (N,3)target offset for type==1, ignored otherwise
    w_fit_sqrt: float = 3.0**0.5,
    w_reg_sqrt: float = 12.0**0.5,
    w_rot_sqrt: float = 5.0**0.5,
    max_iters: int = 20,
):
    device = verts_rest.device
    dtype = verts_rest.dtype

    n_verts = verts_rest.shape[0]
    n_edges = edges.shape[0]
    i0 = edges[:, 0]
    i1 = edges[:, 1]

    vert_offset_init = torch.zeros((1, 3 * n_verts), device=device, dtype=dtype)
    rot_init = torch.eye(3, device=device, dtype=dtype)[None, None, ...]
    rot_init = rot_init.expand(-1, n_verts, -1, -1)

    vert_offset = th.Vector(tensor=vert_offset_init, name="vert_offset")  # # opt var
    rot = th.Vector(tensor=rot_init.reshape(1, 9 * n_verts), name="rot")  # opt var
    verts_rest_var = th.Variable(verts_rest.reshape(1, 3 * n_verts), name="verts_rest")
    handle_offsets_var = th.Variable(
        handle_offsets.reshape(1, 3 * n_verts), name="handle_offsets"
    )
    constraint_type_var = th.Variable(
        constraint_type.reshape(1, n_verts), name="constraint_type"
    )

    w_fit = th.ScaleCostWeight(
        torch.tensor(float(w_fit_sqrt), device=device, dtype=dtype)
    )
    w_reg = th.ScaleCostWeight(
        torch.tensor(float(w_reg_sqrt), device=device, dtype=dtype)
    )
    w_rot = th.ScaleCostWeight(
        torch.tensor(float(w_rot_sqrt), device=device, dtype=dtype)
    )

    def fit_err(optim_vars, aux_vars):
        (vert_offset_var,) = optim_vars
        (C_v, constraint_type_var) = aux_vars

        off = vert_offset_var.tensor.view(1, n_verts, 3)
        c = C_v.tensor.view(1, n_verts, 3)
        ctype = constraint_type_var.tensor.view(1, n_verts)

        m_disp = (ctype == 1).unsqueeze(-1)
        m_fix = (ctype == 2).unsqueeze(-1)

        r = torch.zeros_like(off)
        r = torch.where(m_fix, off, r)
        r = torch.where(m_disp, off - c, r)

        return r.reshape(1, 3 * n_verts)

    def rot_err(optim_vars, aux_vars):
        (rot_var,) = optim_vars
        rot = rot_var.tensor.view(1, n_verts, 3, 3)

        c0 = rot[:, :, :, 0]
        c1 = rot[:, :, :, 1]
        c2 = rot[:, :, :, 2]

        dot01 = (c0 * c1).sum(dim=-1)
        dot02 = (c0 * c2).sum(dim=-1)
        dot12 = (c1 * c2).sum(dim=-1)
        n0 = (c0 * c0).sum(dim=-1) - 1.0
        n1 = (c1 * c1).sum(dim=-1) - 1.0
        n2 = (c2 * c2).sum(dim=-1) - 1.0

        r = torch.stack([dot01, dot02, dot12, n0, n1, n2], dim=-1)  # (1,N,6)
        return r.reshape(1, 6 * n_verts)

    def reg_err(optim_vars, aux_vars):
        (vert_offset_var, rot_var) = optim_vars
        (verts_rest_var,) = aux_vars

        off = vert_offset_var.tensor.view(1, n_verts, 3)
        rot = rot_var.tensor.view(1, n_verts, 3, 3)
        u = verts_rest_var.tensor.view(1, n_verts, 3)

        i0b = i0  # (E,)
        i1b = i1  # (E,)

        du = (u[:, i1b, :] - u[:, i0b, :]).unsqueeze(-1)  # (1,E,3,1)
        rdu = torch.matmul(rot[:, i0b, :, :], du).squeeze(-1)  # (1,E,3)

        x1 = u[:, i1b, :] + off[:, i1b, :]
        x0 = u[:, i0b, :] + off[:, i0b, :]

        r = (x1 - x0) - rdu
        return r.reshape(1, 3 * n_edges)

    objective = th.Objective(dtype=dtype).to(device=device)

    objective.add(
        th.AutoDiffCostFunction(
            optim_vars=[vert_offset],
            err_fn=fit_err,
            dim=3 * n_verts,
            aux_vars=[handle_offsets_var, constraint_type_var],
            cost_weight=w_fit,
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            optim_vars=[rot],
            err_fn=rot_err,
            dim=6 * n_verts,
            aux_vars=[],
            cost_weight=w_rot,
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            optim_vars=[vert_offset, rot],
            err_fn=reg_err,
            dim=3 * n_edges,
            aux_vars=[verts_rest_var],
            cost_weight=w_reg,
        )
    )
    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=max_iters, step_size=2.0
    )
    layer = th.TheseusLayer(optimizer)
    return layer


@profile_fn
def arap_solve(
    verts: torch.Tensor,
    halfedge_weights: torch.Tensor,
    vert_vert: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
    lap_factors: spla.SolverT,
    max_iter: int = 100,
    eps: float = 1e-5,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    solver = make_solver_layer(
        partial(arap_step, solver=lap_factors),
        [(0, 0)],
        (1, 2, 4, 6),
        fwd_method="fixed-point",
        fwd_max_iter=max_iter,
        fwd_eps=eps,
        bwd_method="gmres",
        bwd_max_iter=200,
        bwd_eps=1e-4,
        verbose=verbose,
    )

    init = verts.clone()
    # TODO: Next line only necessary because of bad gradients with identity matrix?
    init[bc_idx] = bc_vals
    return solver(init, verts, halfedge_weights, vert_vert, lap, bc_idx, bc_vals)


def main(
    mesh_path: Path,
    target_mesh_path: Path,
    handles_path: Path,
    device_name: str,
    dtype_name: str,
    method: str,
    lr: float,
    arap_steps: int,
    max_steps: int,
):
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    results_dir = Path.home() / "Dropbox" / "Results" / "iskra" / "arap_2"
    results_dir = results_dir / mesh_path.stem / f"{method}_{device_name}_{dtype_name}"

    torch.cuda.reset_peak_memory_stats()

    # Load meshes
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    target_mesh, _ = Mesh.from_path(target_mesh_path, dtype=dtype, device=device)
    _, target_verts = target_mesh.topo.faces, target_mesh.geom.vertices

    # Load handles
    bdr_idx = boundary(faces)[:, 0]
    with Path(handles_path).open("r") as f:
        control_idx = torch.tensor(
            [int(i) for i in f.readline().split(", ")],
            device=device,
            dtype=torch.int64,
        )
    handle_idx = torch.cat([bdr_idx, control_idx])
    edges, _, _ = get_subfaces(faces)
    vert_vert = vertex_adjacency(faces)
    weights = cotan_weights(verts, faces, clamp_min=1e-5)
    vert_vert_weights = edge_to_vertex_adjacency(weights)
    lap = laplacian_from_weights(weights, faces)
    unknown_idx = sp.index_complement(mesh.n_vertices, handle_idx)
    lap_uk = spla.quad_energy_mat(lap, unknown_idx)
    lap_factors = spla.default_solver(lap_uk)

    handles = verts[handle_idx].clone()

    # print("Computing igl ARAP.")
    # arap_data_igl = igl.ARAPData()
    # arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    # arap_data_igl.max_iter = args.arap_steps
    # with profile_block("igl_arap_precomp"):
    #     igl.arap_precomputation(
    #         verts.detach().cpu().numpy(),
    #         faces.detach().cpu().numpy(),
    #         3,
    #         handle_idx.detach().cpu().numpy(),
    #         arap_data_igl,
    #     )
    # with profile_block(f"igl_arap_solve_{arap_data_igl.max_iter}_steps"):
    #     arap_deformed_igl = igl.arap_solve(
    #         handles.detach().cpu().numpy(),
    #         arap_data_igl,
    #         verts.detach().cpu().numpy(),
    #     )
    # print("Done computing igl ARAP.")

    anim_path = Path(results_dir, "animation")
    LOGGER.warning(f"Saving animations to: {anim_path}")
    anim_path.mkdir(exist_ok=True, parents=True)

    # igl.writeOBJ(
    #     str(anim_path / f"step_handles_{0}.obj"),
    #     handles.detach().cpu().numpy(),
    #     np.empty([0, 3]),
    # )
    if method == "theseus":
        torch.autograd.detect_anomaly(True)
        constraint_type = torch.zeros(
            [verts.shape[0]], dtype=verts.dtype, device=verts.device
        )
        constraint_type[handle_idx] = 1
        handle_offset = torch.zeros_like(verts)
        handle_offset[handle_idx] = handles.detach() - verts[handle_idx]
        th_layer = build_arap_layer(verts, edges, constraint_type, handle_offset)
        print("Success!")

    handles = handles.requires_grad_(True)
    optimizer = torch.optim.SGD([handles], lr=lr)

    for step in range(max_steps):
        print(f"Step {step}")
        with profile_block("optim_step"):
            optimizer.zero_grad()
            with profile_block("forward"):
                if method == "iskra":
                    deformed, energy = arap_solve(
                        verts,
                        vert_vert_weights,
                        vert_vert,
                        lap,
                        handle_idx,
                        handles,
                        lap_factors,
                        max_iter=arap_steps,
                        verbose=False,
                    )
                    print(f"ARAP energy: {energy.mean().detach().cpu().item()}")
                elif method == "theseus":
                    gc.collect()
                    handle_offset = torch.zeros_like(verts)
                    handle_offset[handle_idx] = handles - verts[handle_idx]
                    vert_offset_init = torch.zeros(
                        (1, 3 * verts.shape[0]), device=device, dtype=dtype
                    )
                    rot_init = torch.eye(3, device=device, dtype=dtype)[None, None, ...]
                    rot_init = rot_init.expand(-1, verts.shape[0], -1, -1)
                    inputs = {
                        "verts_rest": verts.flatten()[None, :].detach().clone(),
                        "handle_offsets": handle_offset.flatten()[None, :],
                        "constraint_type": constraint_type[None, :],
                        "rot": rot_init.reshape(1, 9 * verts.shape[0]),
                        "vert_offset": vert_offset_init,
                    }
                    outputs, _ = th_layer.forward(
                        input_tensors=inputs,
                        optimizer_kwargs={
                            "track_err_history": True,
                            "bakcward_mode": "implicit",
                        },
                    )
                    deformed = verts + outputs["vert_offset"].reshape(*verts.shape)
            loss = ((deformed - target_verts) ** 2).mean()
            print(f"Loss = {loss.detach().cpu().item()}?.")

            with profile_block("backward"):
                loss.backward()
            optimizer.step()

            with torch.no_grad():
                LOGGER.warning(
                    f"Saving animations to: {anim_path / f'step_{step}.obj'}"
                )
                igl.writeOBJ(
                    str(anim_path / f"step_{step}.obj"),
                    deformed.detach().cpu().numpy(),
                    faces.cpu().numpy(),
                )
                print(f"Saving to: {anim_path / f'step_{step}.obj'}.")
                igl.writeOBJ(
                    str(anim_path / f"step_handles_{step}.obj"),
                    handles.clone().detach().cpu().numpy(),
                    np.empty([0, 3]),
                )
    with Path(results_dir, "profile.json").open("w") as f:
        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
    global_profiler.dump(path=Path(results_dir, "profile"))

    peak_rss_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower().startswith("linux"):
        pass
    elif platform.system().lower().startswith("darwin"):
        peak_rss_cpu = peak_rss_cpu / 1024
    peak_rss_gpu = torch.cuda.max_memory_allocated() / 1024

    mem_cpu_str = f"Peak (CPU): {peak_rss_cpu} KB, {peak_rss_cpu / (1024 * 1024)} GB"
    mem_gpu_str = f"Peak (GPU): {peak_rss_gpu} KB, {peak_rss_gpu / (1024 * 1024)} GB"
    print(mem_cpu_str)
    print(mem_gpu_str)

    with Path(results_dir, "memory.txt").open("w") as f:
        f.writelines([mem_cpu_str, mem_gpu_str])


if __name__ == "__main__":
    """
    python -m iskra.apps.inverse_arap data/hand/hand.obj data/hand/hand_deformed_sculpt.obj --handles data/hand/hand_handles.txt --lr 10 --arap_steps 150
    python -m iskra.apps.inverse_arap data/penguin/penguin.obj data/penguin/penguin_deformed.obj --handles data/penguin/penguin_handles.txt --lr 5 --arap_steps 100 --steps 150
    python -m iskra.apps.comparisons.arap ~/Dropbox/Data/iskra-data/arap/armadillo/armadillo.obj  ~/Dropbox/Data/iskra-data/arap/armadillo/armadillo_deformed.obj --handles ~/Dropbox/Data/iskra-data/arap/armadillo/armadillo_handles.txt --lr 10 --arap_steps 200
    python -m iskra.apps.inverse_arap data/springer_rm/springer.obj data/springer_rm/springer_deformed.obj --handles data/springer_rm/springer_handles.txt --lr 5 --arap_steps 100
    """
    print(f"Default num_threads: {torch.get_num_threads()}")

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=Path, help="Source mesh path.")
    parser.add_argument("target_mesh_path", type=Path, help="Target mesh path.")
    parser.add_argument("--handles", type=Path, help="The path of the handles to load.")
    parser.add_argument("--lr", default=5.0, type=float, help="Learning rate.")
    parser.add_argument(
        "--arap_steps", default=100, type=int, help="Num. steps for ARAP."
    )
    parser.add_argument(
        "--steps", default=250, type=int, help="Num. steps for outer loop."
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Mesh data type.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    parser.add_argument(
        "--method",
        type=str,
        default="iskra",
        help="Which method to use in (iskra, cvxpylayers).",
    )
    args = parser.parse_args()
    main(
        args.mesh_path,
        args.target_mesh_path,
        args.handles,
        args.device,
        args.dtype,
        args.method,
        args.lr,
        args.arap_steps,
        args.steps,
    )

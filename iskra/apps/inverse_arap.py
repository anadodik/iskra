# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import numpy as np
import torch

import iskra.sparse as sp
import iskra.sparse_linalg as spla
from iskra.adjoint import make_solver_layer
from iskra.dec import d_01, d_10, laplacian, laplacian_from_weights
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block, profile_fn
from iskra.signed_svd import closest_rot_3x3, polar_3x3, signed_svd
from iskra.topology import (
    boundary,
    edge_to_vertex_adjacency,
    face_index,
    get_subfaces,
    reduce_on_subface,
    vertex_adjacency,
)


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


# def arap_step(deformed, verts, cots, vert_vert, lap, handle_idx, handles):
#     nv = verts.shape[0]

#     # Find covariances
#     lines = face_index(verts, vert_vert)
#     vecs = lines[..., 1, :] - lines[..., 0, :]
#     lines_def = face_index(deformed, vert_vert)
#     vecs_def = lines_def[..., 1, :] - lines_def[..., 0, :]
#     covs = cots[..., None, None] * vecs_def[..., None, :] * vecs[..., :, None]
#     vert_covs = reduce_on_subface(covs, vert_vert[..., 0:1], nv, "sum")

#     # Find closest rotation
#     vert_rot = closest_rot_3x3(vert_covs)

#     # Rotate using closest rotation
#     vert_vert_rot = face_index(vert_rot.mT, vert_vert).mean(1)
#     rotated = cots[:, None] * (vert_vert_rot @ vecs[..., None])[..., 0]
#     rhs = reduce_on_subface(rotated, vert_vert[..., 0:1], nv, "sum")

#     # Solve for deformation
#     _, deformed = spla.min_quadratic_energy(lap, -rhs, handle_idx, handles)
#     return deformed


# def arap_step(
#     verts_deformed: torch.Tensor,
#     verts: torch.Tensor,
#     cots: torch.Tensor,
#     vert_vert: torch.Tensor,
#     lap: torch.Tensor,
#     bc_idx: torch.Tensor,
#     bc_vals: torch.Tensor,
#     solver: CholespySolver | None = None,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     n_vertices = verts.shape[0]

#     lines = face_index(verts, vert_vert)
#     vecs = lines[..., 1, :] - lines[..., 0, :]

#     lines_deformed = face_index(verts_deformed, vert_vert)
#     vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]

#     halfedge_covs = (
#         cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]
#     )

#     vert_covs = reduce_on_subface(halfedge_covs, vert_vert[:, 0:1], n_vertices, "sum")
#     # vert_u, _, vert_vt = signed_svd(vert_covs)
#     # vert_rot = vert_vt.mT @ vert_u.mT
#     vert_rot = closest_rot_3x3(vert_covs).mT
#     # Uncomment to debug SVD:
#     # vert_rot = vert_covs * 0.0 + torch.eye(3, dtype=vert_u.dtype)[None, :, :].expand(
#     #     n_vertices, -1, -1
#     # )
#     assert (torch.linalg.det(vert_rot) > 0).all()

#     # Following lines are energy only:
#     halfedge_vert_rot = face_index(vert_rot, vert_vert)[:, 0, ...]
#     diff = vecs_deformed - (halfedge_vert_rot @ vecs[..., None])[..., 0]
#     weighted_dist = cots * torch.linalg.vector_norm(diff, dim=-1, ord=2) ** 2

#     vert_energy = reduce_on_subface(weighted_dist, vert_vert[:, 0:1], n_vertices, "sum")

#     # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
#     halfedge_rot = face_index(vert_rot, vert_vert).mean(1)
#     rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]

#     rhs = reduce_on_subface(rotated_halfedge_vecs, vert_vert[:, 0:1], n_vertices, "sum")
#     verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals, solver=solver)[1]
#     return verts_deformed, vert_energy


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
        bwd_eps=1e-5,
        verbose=verbose,
    )

    init = verts.clone()
    # TODO: Next line only necessary because of bad gradients with identity matrix?
    init[bc_idx] = bc_vals
    return solver(init, verts, halfedge_weights, vert_vert, lap, bc_idx, bc_vals)


_MESH_HANDLES = {
    "tet": [0, 1, 2],
    "cube": [0, 1, 2, 3],
    "koala": [762, 703, 145, 62],  # , 62, 85, 22, 104, 175, 3225
    "hand_lowres": [762, 703, 145, 62],
    "ogre": [12211, 1262],
    "penguin": [1165, 1243, 135, 2678, 945, 2645, 68, 841, 119, 903, 2467, 1383],
}


def main(
    mesh_path: Path,
    target_mesh_path: Path,
    handles_path: Path,
    lr: float,
    arap_steps: int,
    max_steps: int,
):
    dtype = torch.double
    device = "cpu"

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

    vert_vert = vertex_adjacency(faces)
    weights = cotan_weights(verts, faces, clamp_min=1e-5)
    vert_vert_weights = edge_to_vertex_adjacency(weights)
    lap = laplacian_from_weights(weights, faces)
    unknown_idx = sp.index_complement(mesh.n_vertices, handle_idx)
    lap_uk = spla.quad_energy_mat(lap, unknown_idx)
    lap_factors = spla.default_solver(lap_uk)

    handles = verts[handle_idx]
    handles = handles.requires_grad_(True)
    optimizer = torch.optim.SGD([handles], lr=lr)
    optimizer.zero_grad()
    deformed, energy = arap_solve(
        verts, vert_vert_weights, vert_vert, lap, handle_idx, handles, lap_factors
    )
    loss = ((deformed - target_verts) ** 2).mean()
    loss.backward()

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    with profile_block("igl_arap_precomp"):
        igl.arap_precomputation(
            verts.numpy(), faces.numpy(), 3, handle_idx.numpy(), arap_data_igl
        )
    with profile_block(f"igl_arap_solve_{arap_data_igl.max_iter}_steps"):
        arap_deformed_igl = igl.arap_solve(
            handles.detach().numpy(), arap_data_igl, verts.detach().numpy()
        )

    global_profiler.dump()
    global_profiler.summary_to_json()

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.register_surface_mesh("mesh", verts, faces.numpy(), enabled=False)
        ps_mesh = ps.register_surface_mesh(
            "deformed", deformed.detach().numpy(), faces.numpy()
        )
        ps_target_mesh = ps.register_surface_mesh(
            "target", target_verts.detach().numpy(), faces.numpy()
        )
        ps_mesh_arap = ps.register_surface_mesh(
            "arap_deformed", arap_deformed_igl, faces.numpy(), enabled=False
        )
        ps_edges = ps.register_curve_network(
            "edges",
            deformed.detach().numpy(),
            vert_vert.numpy(),
            enabled=False,
            radius=0.01,
        )
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        ps_cloud = ps.register_point_cloud("bc", handles.detach().numpy(), enabled=True)
        ps_mesh.add_scalar_quantity(
            "energy", energy.detach().numpy(), defined_on="vertices", enabled=True
        )
        assert handles.grad is not None
        ps_cloud.add_vector_quantity(
            "-grad bc", -handles.grad, enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity("cotan", vert_vert_weights, defined_on="edges")

        optimizing = False
        optim_step = 0
        out_path = Path(mesh_path.parent, "animation")
        out_path.mkdir(exist_ok=True, parents=True)

        igl.writeOBJ(
            str(out_path / f"step_handles_{optim_step}.obj"),
            handles.detach().cpu().numpy(),
            np.empty([0, 3]),
        )

        def callback():
            nonlocal optimizing, deformed, optim_step, out_path

            if ps.imgui.Button(
                "Start Optimization" if not optimizing else "Stop Optimizing"
            ):
                optimizing = not optimizing
            if optimizing:
                with profile_block("optim_step"):
                    optimizer.zero_grad()
                    deformed, energy = arap_solve(
                        verts,
                        vert_vert_weights,
                        vert_vert,
                        lap,
                        handle_idx,
                        handles,
                        lap_factors,
                        max_iter=arap_steps,
                    )
                    print(f"ARAP energy: {energy.mean().detach().cpu().item()}")
                    loss = ((deformed - target_verts) ** 2).mean()
                    print(f"Loss = {loss.detach().cpu().item()}.")
                    loss.backward()
                    optimizer.step()

                optim_step += 1
                if optim_step % max_steps == 0:
                    global_profiler.dump()
                    with Path(out_path, f"profile_{optim_step}.json").open("w") as f:
                        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
                    global_profiler.dump(path=Path(out_path, f"profile_{optim_step}"))
                    optimizing = False

                with torch.no_grad():
                    print(f"Step {optim_step}.")
                    ps_cloud.update_point_positions(handles.detach().numpy())
                    arap_deformed_igl = igl.arap_solve(
                        handles.detach().numpy(), arap_data_igl, verts.detach().numpy()
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_{optim_step}.obj"),
                        arap_deformed_igl,
                        faces.cpu().numpy(),
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_handles_{optim_step}.obj"),
                        handles.detach().cpu().numpy(),
                        np.empty([0, 3]),
                    )
                    ps_mesh.update_vertex_positions(deformed.detach().numpy())
                    ps_mesh_arap.update_vertex_positions(arap_deformed_igl)
                    ps_mesh.add_scalar_quantity(
                        "energy",
                        energy.detach().numpy(),
                        defined_on="vertices",
                        enabled=True,
                    )

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    """
    python -m iskra.apps.inverse_arap data/hand/hand.obj data/hand/hand_deformed_sculpt.obj --handles data/hand/hand_handles.txt --lr 10 --arap_steps 150
    python -m iskra.apps.inverse_arap data/penguin/penguin.obj data/penguin/penguin_deformed.obj --handles data/penguin/penguin_handles.txt --lr 5 --arap_steps 100 --steps 150
    python -m iskra.apps.inverse_arap data/armadillo/armadillo.obj data/armadillo/armadillo_deformed.obj --handles data/armadillo/armadillo_handles.txt --lr 10 --arap_steps 200
    python -m iskra.apps.inverse_arap data/springer_rm/springer.obj data/springer_rm/springer_deformed.obj --handles data/springer_rm/springer_handles.txt --lr 5 --arap_steps 100
    """
    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(32)
    torch.set_printoptions(linewidth=200, sci_mode=False)

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
    args = parser.parse_args()
    main(
        args.mesh_path,
        args.target_mesh_path,
        args.handles,
        args.lr,
        args.arap_steps,
        args.steps,
    )

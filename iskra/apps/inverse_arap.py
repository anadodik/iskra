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
from iskra.adjoint import make_solver_layer
from iskra.dec import d_01, d_10, laplacian
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block, profile_fn
from iskra.signed_svd import closest_rot_3x3, polar_3x3, signed_svd
from iskra.sparse_linalg import (
    CholespySolver,
    gmres_solve,
    make_cholespy_solver,
    min_quadratic_energy,
    quad_energy_mat,
)
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


@profile_fn
def arap_step(
    verts_deformed,
    verts_rest,
    cots,
    halfedges,
    lap,
    bc_idx,
    bc_vals,
    solver: CholespySolver | None = None,
):
    n_vertices = verts_rest.shape[0]

    with profile_block("accumulation"):
        lines = face_index(verts_rest, halfedges)
        vecs = lines[..., 1, :] - lines[..., 0, :]
        lines_deformed = face_index(verts_deformed, halfedges)
        vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
        covs = cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]

    with profile_block("svd"):
        vert_covs = reduce_on_subface(covs, halfedges[:, 0:1], n_vertices, "sum")
        vert_rot = closest_rot_3x3(vert_covs)

    with profile_block("rotation"):
        halfedge_rot = face_index(vert_rot.mT, halfedges).mean(1)
        rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]
        rhs = reduce_on_subface(
            rotated_halfedge_vecs, halfedges[:, 0:1], n_vertices, "sum"
        )

    with profile_block("solve"):
        verts_deformed = min_quadratic_energy(
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
#     solver: CholespySolver | None = None,
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
#     verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals, solver=solver)[1]
#     return verts_deformed, vert_energy


@profile_fn
def arap_solve(
    verts: torch.Tensor,
    halfedge_weights: torch.Tensor,
    halfedges: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
    lap_factors: CholespySolver,
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
    return solver(init, verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals)


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
):
    dtype = torch.double
    device = "cpu"

    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    target_mesh, _ = Mesh.from_path(target_mesh_path, dtype=dtype, device=device)
    _, target_verts = target_mesh.topo.faces, target_mesh.geom.vertices

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]

    with Path(handles_path).open("r") as f:
        control_idx = torch.tensor(
            [int(i) for i in f.readline().split(", ")],
            device=device,
            dtype=torch.int64,
        )

    control_verts = verts[control_idx]
    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_vals = torch.cat([bdr_verts, control_verts])

    weights = cotan_weights(verts, faces).clamp_min(1e-5)
    lap, _ = laplacian(verts, faces, clamp_min=1e-5)
    lap_factors = make_cholespy_solver(
        quad_energy_mat(lap, sp.index_complement(mesh.n_vertices, bc_idx))
    )

    edges, _, _ = get_subfaces(faces)
    _, edge_verts, _ = get_subfaces(edges)
    halfedges = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    halfedge_weights = torch.cat([weights, weights], 0)

    bc_vals = bc_vals.requires_grad_(True)
    optimizer = torch.optim.SGD([bc_vals], lr=lr)
    optimizer.zero_grad()
    deformed, energy = arap_solve(
        verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals, lap_factors
    )
    loss = ((deformed - target_verts) ** 2).mean()
    loss.backward()
    global_profiler.dump()

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    igl.arap_precomputation(
        verts.numpy(), faces.numpy(), 3, bc_idx.numpy(), arap_data_igl
    )
    arap_deformed_igl = igl.arap_solve(
        bc_vals.detach().numpy(), arap_data_igl, verts.detach().numpy()
    )
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
            halfedges.numpy(),
            enabled=False,
            radius=0.01,
        )
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        ps_cloud = ps.register_point_cloud("bc", bc_vals.detach().numpy(), enabled=True)
        ps_mesh.add_scalar_quantity(
            "energy", energy.detach().numpy(), defined_on="vertices", enabled=True
        )
        assert bc_vals.grad is not None
        ps_cloud.add_vector_quantity(
            "-grad bc", -bc_vals.grad, enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity("cotan", halfedge_weights, defined_on="edges")

        optimizing = False
        optim_step = 0
        out_path = Path(mesh_path.parent, "animation")
        out_path.mkdir(exist_ok=True, parents=True)

        def callback():
            nonlocal optimizing, deformed, optim_step, out_path

            if ps.imgui.Button(
                "Start Optimization" if not optimizing else "Stop Optimizing"
            ):
                optimizing = not optimizing
            if optimizing:
                optimizer.zero_grad()
                deformed, energy = arap_solve(
                    verts,
                    halfedge_weights,
                    halfedges,
                    lap,
                    bc_idx,
                    bc_vals,
                    lap_factors,
                    max_iter=arap_steps,
                )
                print(f"ARAP energy: {energy.mean().detach().cpu().item()}")
                loss = ((deformed - target_verts) ** 2).mean()
                print(f"Loss = {loss.detach().cpu().item()}.")
                loss.backward()
                optimizer.step()

                optim_step += 1
                if optim_step % 250 == 0:
                    global_profiler.dump()
                    with Path(out_path, f"profile_{optim_step}.json").open("w") as f:
                        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
                    global_profiler.dump(path=Path(out_path, f"profile_{optim_step}"))
                    optimizing = False

                with torch.no_grad():
                    print(f"Step {optim_step}.")
                    ps_cloud.update_point_positions(bc_vals.detach().numpy())
                    arap_deformed_igl = igl.arap_solve(
                        bc_vals.detach().numpy(), arap_data_igl, verts.detach().numpy()
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_{optim_step}.obj"),
                        arap_deformed_igl,
                        faces.cpu().numpy(),
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_handles_{optim_step}.obj"),
                        bc_vals.detach().cpu().numpy(),
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
    python -m iskra.apps.inverse_arap data/penguin/penguin.obj data/penguin/penguin_deformed.obj --handles data/hand/penguin_handles.txt --lr 5 --arap_steps 100
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
    args = parser.parse_args()
    main(args.mesh_path, args.target_mesh_path, args.handles, args.lr, args.arap_steps)

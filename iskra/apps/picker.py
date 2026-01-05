# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
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
from iskra.signed_svd import closest_rot_3x3, polar_3x3, signed_svd
from iskra.sparse_linalg import gmres_solve, min_quadratic_energy
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


def arap_step(verts_deformed, verts_rest, cots, halfedges, lap, bc_idx, bc_vals):
    n_vertices = verts_rest.shape[0]

    lines = face_index(verts_rest, halfedges)
    vecs = lines[..., 1, :] - lines[..., 0, :]
    lines_deformed = face_index(verts_deformed, halfedges)
    vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
    covs = cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]

    vert_covs = reduce_on_subface(covs, halfedges[:, 0:1], n_vertices, "sum")
    vert_rot = closest_rot_3x3(vert_covs)
    # vert_u, _, vert_vt = signed_svd(vert_covs)
    # vert_rot = vert_vt.mT @ vert_u.mT

    halfedge_rot = face_index(vert_rot.mT, halfedges).mean(1)
    rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]

    rhs = reduce_on_subface(rotated_halfedge_vecs, halfedges[:, 0:1], n_vertices, "sum")
    verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals)[1]
    return verts_deformed


def arap_step(
    verts_deformed: torch.Tensor,
    verts: torch.Tensor,
    halfedge_weights: torch.Tensor,
    halfedges: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_vertices = verts.shape[0]

    lines = face_index(verts, halfedges)
    vecs = lines[..., 1, :] - lines[..., 0, :]

    lines_deformed = face_index(verts_deformed, halfedges)
    vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]

    halfedge_covs = (
        halfedge_weights[..., None, None]
        * vecs_deformed[..., None, :]
        * vecs[..., :, None]
    )

    vert_covs = reduce_on_subface(halfedge_covs, halfedges[:, 0:1], n_vertices, "sum")
    # vert_u, _, vert_vt = signed_svd(vert_covs)
    # vert_rot = vert_vt.mT @ vert_u.mT
    vert_rot = closest_rot_3x3(vert_covs).mT
    # Uncomment to debug SVD:
    # vert_rot = vert_covs * 0.0 + torch.eye(3, dtype=vert_u.dtype)[None, :, :].expand(
    #     n_vertices, -1, -1
    # )
    assert (torch.linalg.det(vert_rot) > 0).all()

    # Following lines are energy only:
    halfedge_vert_rot = face_index(vert_rot, halfedges)[:, 0, ...]
    diff = vecs_deformed - (halfedge_vert_rot @ vecs[..., None])[..., 0]
    weighted_dist = (
        halfedge_weights * torch.linalg.vector_norm(diff, dim=-1, ord=2) ** 2
    )
    vert_energy = reduce_on_subface(weighted_dist, halfedges[:, 0:1], n_vertices, "sum")

    # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
    halfedge_rot = face_index(vert_rot, halfedges).mean(1)
    rotated_halfedge_vecs = (
        halfedge_weights[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]
    )

    rhs = reduce_on_subface(rotated_halfedge_vecs, halfedges[:, 0:1], n_vertices, "sum")
    verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals)[1]
    print(vert_energy.sum())
    return verts_deformed, vert_energy


def arap_solve(
    verts: torch.Tensor,
    halfedge_weights: torch.Tensor,
    halfedges: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
    max_iter: int = 50,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    solver = make_solver_layer(
        arap_step,
        [(0, 0)],
        (2, 4, 6),
        fwd_method="fixed-point",
        fwd_max_iter=max_iter,
        fwd_eps=eps,
        bwd_method="gmres",
        bwd_max_iter=200,
        bwd_eps=1e-12,
    )

    init = verts.clone()
    # TODO: Next line only necessary because of bad gradients with identity matrix?
    init[bc_idx] = bc_vals
    return solver(init, verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals)


def main(
    mesh_path: Path,
    handles_path: Path | None = None,
    handles_deformed_path: Path | None = None,
):
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]

    if handles_path is None:
        control_idx = torch.empty([0], device=device, dtype=torch.int64)
    else:
        print(f"Opening handles path: {handles_path}")
        with Path(handles_path).open("r") as f:
            control_idx = torch.tensor(
                [int(i) for i in f.readline().split(", ")],
                device=device,
                dtype=torch.int64,
            )
    if handles_deformed_path is None:
        control_verts = verts[control_idx]
    else:
        control_verts = Mesh.from_path(
            handles_deformed_path, dtype=dtype, device=device
        )[0].vertices

    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_vals = torch.cat([bdr_verts, control_verts])

    weights = cotan_weights(verts, faces)
    lap, _ = laplacian(verts, faces)
    edges, _, _ = get_subfaces(faces)
    _, edge_verts, _ = get_subfaces(edges)
    halfedges = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    halfedge_weights = torch.cat([weights, weights], 0)

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 200
    if control_idx.nelement() == 0:
        deformed = verts
        arap_deformed_igl = verts
    else:
        deformed, energy = arap_solve(
            verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals
        )
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
        ps_mesh_rest = ps.register_surface_mesh(
            "mesh", verts, faces.numpy(), enabled=True
        )
        ps_mesh_rest.set_selection_mode("vertices_only")

        ps_mesh = ps.register_surface_mesh(
            "deformed", deformed.detach().numpy(), faces.numpy()
        )
        ps_igl_mesh = ps.register_surface_mesh(
            "deformed_igl", arap_deformed_igl, faces.numpy()
        )
        ps_cloud = ps.register_point_cloud("bc", bc_vals.detach().numpy(), enabled=True)

        def callback():
            nonlocal ps_cloud, control_idx, control_verts, bc_idx, bc_vals, deformed

            if ps.imgui.Button("Dump"):
                with Path("data", "handles.txt").open("w") as f:
                    f.write(", ".join([str(i) for i in bc_idx.cpu().numpy().tolist()]))
                igl.writeOBJ("data/pc.obj", bc_vals.numpy(), np.empty([0, 3]))
                igl.writeOBJ(
                    "data/deformed.obj", deformed.cpu().numpy(), faces.cpu().numpy()
                )
            io = ps.imgui.GetIO()
            if io.MouseClicked[1]:  # if clicked
                screen_coords = io.MousePos
                pick_result = ps.pick(screen_coords=screen_coords)

                # check out pick_result.is_hit, pick_result.structureName, pick_result.depth, etc

                if pick_result.is_hit and pick_result.structure_name == "mesh":
                    print(pick_result.structure_data)
                    control_idx = torch.cat(
                        [
                            control_idx,
                            torch.tensor(
                                [pick_result.structure_data["index"]],
                                device=device,
                                dtype=torch.int64,
                            ),
                        ]
                    )
                    control_verts = verts[control_idx]
                    bc_idx = torch.cat([bdr_idx, control_idx])
                    bc_vals = torch.cat([bdr_verts, control_verts])
                    ps_cloud.remove()
                    ps_cloud = ps.register_point_cloud(
                        "bc", bc_vals.detach().numpy(), enabled=True
                    )
                elif pick_result.is_hit and pick_result.structure_name == "bc":
                    print(pick_result.structure_data)
                    bc_idx = torch.cat([bdr_idx, control_idx])
                    bc_vals = torch.cat([bdr_verts, control_verts])

                    deformed, energy = arap_solve(
                        verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals
                    )
                    print(energy.mean())
                    ps_mesh.update_vertex_positions(deformed.detach().numpy())
                    # additional dictionary of element type, coords, etc.

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(8)
    torch.set_printoptions(linewidth=200, sci_mode=False)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=Path, help="The path of the mesh to load.")
    parser.add_argument(
        "--handles",
        default=None,
        type=Path,
        help="The path of the handles to load.",
    )
    parser.add_argument(
        "--handles_deformed",
        default=None,
        type=Path,
        help="The path of the handles to load.",
    )
    args = parser.parse_args()
    main(args.mesh_path, args.handles, args.handles_deformed)

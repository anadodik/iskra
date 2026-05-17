# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import numpy as np
import torch

import iskra.sparse as sp
from iskra.dec import d_01, d_10, laplacian
from iskra.deformation import arap_precompute, arap_solve
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.signed_svd import closest_rot_3x3, polar_3x3, signed_svd
from iskra.sparse_linalg import gmres_solve, min_quadratic_energy
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


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
        control_idx = torch.tensor([0, 1, 2], device=device, dtype=torch.int64)
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

    halfedges, halfedge_weights, lap, lap_factors = arap_precompute(
        verts, faces, bc_idx
    )

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    if control_idx.nelement() == 0:
        deformed = verts
        arap_deformed_igl = verts
    else:
        deformed, energy = arap_solve(
            verts, bc_idx, bc_vals, halfedges, halfedge_weights, lap, lap_factors
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
                        verts,
                        bc_idx,
                        bc_vals,
                        halfedges,
                        halfedge_weights,
                        lap,
                        lap_factors,
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

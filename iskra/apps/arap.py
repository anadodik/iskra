# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path

import igl
import torch

from iskra.adjoint import compute_numerical_jacobian
from iskra.deformation import arap_precompute, arap_solve
from iskra.mesh import Mesh
from iskra.topology import boundary

_MESH_HANDLES = {
    "tet": [0, 1, 2],
    "cube": [0, 1, 2, 3],
    "koala_flat_vt": [762, 703, 145, 62, 85, 22, 104, 175, 3225, 3511],
    "hand_lowres": [762, 703, 145, 62],
    "ogre": [12211, 1262],
}


def main(
    mesh_path: Path, handles_path: Path | None, handles_deformed_path: Path | None
) -> None:
    # TODO: we should differentiate into cot weights to get artistic control
    # over deformations!
    # TODO: inverse rendering + video diffusion guidance

    global optimizing, optim_step, deformed

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    if handles_path is not None:
        with Path(handles_path).open("r") as f:
            control_idx = torch.tensor(
                [int(i) for i in f.readline().split(", ")],
                device=device,
                dtype=torch.int64,
            )
            print("Handles loaded.")
        control_verts = Mesh.from_path(handles_deformed_path, device="cpu")[
            0
        ].geom.vertices
        print(control_verts.shape, control_idx.shape)
    else:
        control_idx = _MESH_HANDLES.get(mesh_path.stem, [0, 1, 2])
        control_idx = torch.tensor(control_idx, device=device, dtype=torch.int64)
        control_verts = verts[control_idx] - 1
        control_verts[-1, -1] += 0.1

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]
    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_vals = torch.cat([bdr_verts, control_verts])
    halfedges, halfedge_weights, lap, lap_factors = arap_precompute(
        verts, faces, bc_idx, None
    )

    bc_vals = bc_vals.requires_grad_(True)
    deformed, energy = arap_solve(
        verts,
        bc_idx,
        bc_vals,
        halfedges,
        halfedge_weights,
        lap,
        lap_factors,
        fwd_max_iter=1000,
        bwd_max_iter=1000,
        verbose=True,
    )

    grad_deformed = torch.zeros_like(verts)
    grad_deformed[3] += -0.1
    deformed.backward(grad_deformed)

    num_grad = None
    num_jac = compute_numerical_jacobian(
        lambda *args, **kwargs: arap_solve(*args, **kwargs)[0],
        0,
        2,
        1e-8,
        verts,
        bc_idx,
        bc_vals,
        halfedges,
        halfedge_weights,
        lap,
        fwd_max_iter=1000,
        bwd_max_iter=1000,
        lap_factors=lap_factors,
    )
    num_grad = (grad_deformed.flatten() @ num_jac).reshape(*bc_vals.shape)

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    igl.arap_precomputation(
        verts.numpy(), faces.numpy(), 3, bc_idx.numpy(), arap_data_igl
    )
    init = verts.clone()
    init[bc_idx] = bc_vals
    arap_deformed_igl = igl.arap_solve(
        bc_vals.detach().numpy(), arap_data_igl, init.detach().numpy()
    )

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh_rest = ps.register_surface_mesh(
            "mesh", verts, faces.numpy(), enabled=False
        )
        ps_mesh_rest.set_selection_mode("vertices_only")
        ps_mesh = ps.register_surface_mesh(
            "deformed", deformed.detach().numpy(), faces.numpy()
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
        ps_mesh.add_vector_quantity(
            "grad_deformed", grad_deformed, length=0.15, enabled=True
        )
        _ = ps.register_point_cloud(
            "bc_rest", verts[bc_idx].detach().numpy(), enabled=True
        )
        if num_grad is not None:
            ps_cloud.add_vector_quantity(
                "num_grad_bc", num_grad, enabled=True, length=0.15
            )
        ps_cloud.add_vector_quantity(
            "backward_grad_bc", bc_vals.grad, enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity("cotan", halfedge_weights, defined_on="edges")
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
    parser.add_argument("mesh_path", type=Path, help="Path of the mesh to load.")
    parser.add_argument("--handles", type=Path, help="Path of the handles to load.")
    parser.add_argument(
        "--handles_deformed", type=Path, help="Path of the deformed handles."
    )
    args = parser.parse_args()
    main(args.mesh_path, args.handles, args.handles_deformed)

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from typing import Literal

import igl
import numpy as np
import torch

from iskra.dec import laplacian
from iskra.geometry import cotan_weights, triangle_areas, triangle_coordinate_system
from iskra.mesh import Mesh
from iskra.signed_svd import signed_svdvals
from iskra.sparse_linalg import CholeskySolver, min_quadratic_energy
from iskra.topology import (
    boundary,
    face_index,
    get_subfaces,
    ordered_boundary_edges,
    reduce_on_subface,
)


def triangle_to_local(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = face_index(verts, faces)
    _, t, b = triangle_coordinate_system(triangles)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    world_to_local = torch.stack([t, b], -2)
    local = world_to_local @ edge_vecs.mT
    return local


if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, fdtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]

    mesh_center = verts.mean(0, keepdim=True)
    # OGRE:
    # control_idx = torch.tensor([12211, 1262], device=device, dtype=torch.int64)
    # control_verts = 1.2 * (verts[control_idx] - mesh_center) + mesh_center
    # HAND_LOWRES
    control_idx = torch.tensor([762, 703], device=device, dtype=torch.int64)
    control_verts = 1.5 * (verts[control_idx] - mesh_center) + mesh_center
    # TET:
    # control_idx = torch.tensor([1], device=device, dtype=torch.int64)
    # control_verts = verts[control_idx] - 0.1

    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_verts = torch.cat([bdr_verts, control_verts])

    edges, face_edges, face_edge_sign = get_subfaces(faces)
    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))

    lap, mass = laplacian(verts, faces, clamp_min=0.0)
    rhs = torch.zeros([verts.shape[0], 2], dtype=dtype, device=device)
    deformed = verts.clone()
    deformed[bc_idx] = bc_verts

    edges, face_edge, edge_signs = get_subfaces(faces)
    _, edge_verts, vert_signs = get_subfaces(edges)
    half_edge_verts = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    # TODO: to_oriented? to_half_edge?

    weights = cotan_weights(verts, faces)
    # half_edge_weights = torch.ones_like(torch.cat([weights, weights], 0))
    half_edge_weights = torch.cat([weights, weights], 0)

    def step_fn(
        rest: torch.Tensor,
        deformed: torch.Tensor,
        half_edge_weights: torch.Tensor,
        half_edge_verts: torch.Tensor,
    ):
        lines = face_index(rest, half_edge_verts)
        half_edge_vecs = lines[..., 1, :] - lines[..., 0, :]

        deformed_lines = face_index(deformed, half_edge_verts)
        deformed_half_edge_vecs = deformed_lines[..., 1, :] - deformed_lines[..., 0, :]

        covs = (
            half_edge_weights[..., None, None]
            * half_edge_vecs[..., None, :]
            * deformed_half_edge_vecs[..., :, None]
        )

        # TODO: How to nicely reduce half-edges?
        vert_covs = reduce_on_subface(
            covs, half_edge_verts[:, 0:1], mesh.n_vertices, "sum"
        )
        vert_u, _, vert_vt = signed_svdvals(vert_covs)
        vert_rot = vert_u @ vert_vt
        assert (torch.linalg.det(vert_rot) > 0).all()

        half_edge_vert_rot = face_index(vert_rot, half_edge_verts)[:, 0, ...]
        diff = (
            deformed_half_edge_vecs
            - (half_edge_vert_rot @ half_edge_vecs[..., None])[..., 0]
        )
        weighted_dist = (
            half_edge_weights * torch.linalg.vector_norm(diff, dim=-1, ord=2) ** 2
        )

        vert_energy = reduce_on_subface(
            weighted_dist, half_edge_verts[:, 0:1], mesh.n_vertices, "sum"
        )

        # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
        half_edge_vert_rot = face_index(vert_rot, half_edge_verts)
        half_edge_rot = half_edge_vert_rot.mean(1)

        rotated_signed_edge_vecs = (
            half_edge_weights[:, None]
            * (half_edge_rot @ half_edge_vecs[..., None])[..., 0]
        )
        rhs = reduce_on_subface(
            rotated_signed_edge_vecs, half_edge_verts[:, 0:1], mesh.n_vertices, "sum"
        )

        deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_verts)
        return vert_energy, rhs, deformed

    vert_energy, rhs, _ = step_fn(verts, deformed, half_edge_weights, half_edge_verts)

    arap_data = igl.ARAPData()
    arap_data.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap = igl.arap_precomputation(
        verts.numpy(), faces.numpy(), 3, bc_idx.numpy(), arap_data
    )
    arap_deformed = igl.arap_solve(bc_verts.numpy(), arap_data, deformed.numpy())

    try:
        import polyscope as ps

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", arap_deformed, faces.numpy())
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        # ps_mesh.set_enabled(False)
        # ps_edges = ps.register_curve_network("edges", verts, edges, radius=0.001)
        ps_mesh.add_vector_quantity("rhs", rhs, defined_on="vertices")
        # ps_deformed_edges = ps.register_curve_network(
        #     "deformed_edges", deformed, edges, radius=0.001
        # )
        # ps_edges.add_vector_quantity(
        #     "edge_vec",
        #     signed_edge_vecs[:, 0, :],
        #     defined_on="edges",
        #     enabled=True,
        #     radius=0.005,
        #     length=0.1,
        # )
        # ps_edges.add_vector_quantity(
        #     "deformed_edge_vec",
        #     signed_deformed_edge_vecs[:, 0, :],
        #     defined_on="edges",
        #     enabled=True,
        #     radius=0.005,
        #     length=0.1,
        # )
        # ps_edges.add_scalar_quantity(
        #     "cov_magnitude",
        #     torch.linalg.matrix_norm(covs[:, 0, :]),
        #     defined_on="edges",
        #     enabled=True,
        # )
        # ps_mesh.add_scalar_quantity(
        #     "vert_jac",
        #     torch.linalg.matrix_norm(vert_jac).numpy(),
        #     defined_on="vertices",
        #     enabled=True,
        # )
        ps_mesh.add_scalar_quantity(
            "vert_energy", vert_energy.numpy(), defined_on="vertices", enabled=True
        )

        # ps_boundary = ps.register_curve_network(
        #     "boundary", bdr_verts.numpy(), edges="loop"
        # )

        optimizing = False
        optim_step = 0
        impl: Literal["iskra", "libigl"] = "iskra"

        def callback():
            global optimizing, deformed, optim_step

            if ps.imgui.Button(
                "Start Optimization" if not optimizing else "Stop Optimizing"
            ):
                optimizing = not optimizing
            if optimizing:
                if impl == "libgil":
                    arap_data.max_iter = optim_step
                    optim_step += 1
                    arap_deformed = igl.arap_solve(
                        bc_verts.numpy(), arap_data, deformed.numpy()
                    )
                    ps_mesh.update_vertex_positions(arap_deformed)
                else:
                    vert_energy, rhs, deformed = step_fn(
                        verts, deformed, half_edge_weights, half_edge_verts
                    )
                    ps_mesh.update_vertex_positions(deformed.detach().numpy())
                    ps_mesh.add_scalar_quantity(
                        "vert_energy",
                        vert_energy.numpy(),
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

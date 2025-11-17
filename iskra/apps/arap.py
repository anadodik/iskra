# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import torch

import iskra.sparse as sp
from iskra.adjoint import (
    compute_jacobians,
    compute_numerical_jacobian,
    make_solver_layer,
    make_vjp,
)
from iskra.dec import d_01, d_10, laplacian
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.signed_svd import signed_svd
from iskra.sparse_linalg import gmres_solve, min_quadratic_energy
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


def arap_step(verts_deformed, verts_rest, cots, halfedges, lap, bc_idx, bc_vals):
    n_vertices = verts_rest.shape[0]

    lines = face_index(verts_rest, halfedges)
    vecs = lines[..., 1, :] - lines[..., 0, :]
    lines_deformed = face_index(verts_deformed, halfedges)
    vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
    covs = cots[..., None, None] * vecs[..., None, :] * vecs_deformed[..., :, None]

    vert_covs = reduce_on_subface(covs, halfedges[:, 0:1], n_vertices, "sum")
    vert_u, _, vert_vt = signed_svd(vert_covs)
    vert_rot = vert_vt.mT @ vert_u.mT

    halfedge_rot = face_index(vert_rot, halfedges).mean(1)
    rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]

    rhs = reduce_on_subface(rotated_halfedge_vecs, halfedges[:, 0:1], n_vertices, "sum")
    verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals)
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
        * vecs[..., None, :]
        * vecs_deformed[..., :, None]
    )

    vert_covs = reduce_on_subface(halfedge_covs, halfedges[:, 0:1], n_vertices, "sum")
    vert_u, _, vert_vt = signed_svd(vert_covs)
    vert_rot = vert_vt.mT @ vert_u.mT
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
    verts_deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_vals)
    return verts_deformed, vert_energy


def arap_solve(
    verts: torch.Tensor,
    halfedge_weights: torch.Tensor,
    halfedges: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
    max_iter: int = 80,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    solver = make_solver_layer(
        arap_step, 0, 0, (2, 4, 6), max_iter=max_iter, eps=eps, bwrd_method="gmres"
    )

    init = verts.clone()
    # TODO: Next line only necessary because of bad gradients with identity matrix?
    init[bc_idx] = bc_vals
    return solver(init, verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals)


def make_arap_vjp(
    deformed: torch.Tensor,
    rest: torch.Tensor,
    halfedge_weights: torch.Tensor,
    halfedges: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    vjp_deformed, vjp_bc_verts = make_vjp(
        arap_step,
        0,
        0,
        (-1,),
        deformed,
        rest,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )
    return vjp_deformed, vjp_bc_verts


_MESH_HANDLES = {
    "tet": [0, 1, 2],
    "cube": [0, 1, 2, 3],
    "koala": [762, 703, 145, 62],  # , 62, 85, 22, 104, 175, 3225
    "hand_lowres": [762, 703, 145, 62],
    "ogre": [12211, 1262],
}


def main(mesh_path: Path):
    # TODO: we should differentiate into cot weights to get artistic control
    # over deformations!
    # TODO: to_oriented? to_half_edge?
    # TODO: How to nicely reduce half-edges?
    # TODO: These cannot be actual half-edges from faces because of boundaries:
    # TODO: Turn into tests:

    global optimizing, optim_step, deformed

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, fdtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]

    control_idx = _MESH_HANDLES.get(mesh_path.stem, [0, 1, 2])
    control_idx = torch.tensor(control_idx, device=device, dtype=torch.int64)
    control_verts = verts[control_idx] - 1

    grad_deformed = torch.zeros_like(verts)
    grad_deformed[3] += -0.1

    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_vals = torch.cat([bdr_verts, control_verts])

    weights = cotan_weights(verts, faces)
    lap, _ = laplacian(verts, faces)

    edges, _, _ = get_subfaces(faces)
    _, edge_verts, _ = get_subfaces(edges)
    halfedges = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    halfedge_weights = torch.cat([weights, weights], 0)

    bc_vals = bc_vals.requires_grad_(True)
    deformed, energy = arap_solve(
        verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals
    )
    deformed.backward(grad_deformed)

    num_jac = compute_numerical_jacobian(
        arap_solve,
        0,
        -1,
        1e-8,
        verts,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )
    num_grad = (grad_deformed.flatten() @ num_jac).reshape(*bc_vals.shape)
    print("NUM JACOBIAN:\n", num_jac)

    jac_verts, jac_bc = compute_jacobians(
        arap_step,
        0,
        0,
        6,
        deformed,
        verts,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )

    print("JACOBIAN VERTS:\n", jac_verts)
    print("JACOBIAN BC:\n", jac_bc)
    print("JACOBIAN FULL:\n", -torch.linalg.solve(jac_verts, jac_bc))

    vjp_deformed, vjp_bc_verts = make_arap_vjp(
        deformed, verts, halfedge_weights, halfedges, lap, bc_idx, bc_vals
    )
    init = torch.randn_like(deformed)
    dl_df = -gmres_solve(
        lambda z: vjp_deformed(z)[0], grad_deformed, init, maxiter=200, tol=1e-12
    )
    print(
        "AHHHH", torch.norm((grad_deformed - vjp_deformed(-dl_df)[0]).flatten()).item()
    )
    grad_bc = vjp_bc_verts(dl_df)[0]

    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    igl.arap_precomputation(
        verts.numpy(), faces.numpy(), 3, bc_idx.numpy(), arap_data_igl
    )
    arap_deformed_igl = igl.arap_solve(
        bc_vals.detach().numpy(), arap_data_igl, deformed.detach().numpy()
    )

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh_rest = ps.register_surface_mesh(
            "mesh", verts, faces.numpy(), enabled=False
        )
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
        ps_mesh.add_vector_quantity("dl_df", dl_df, enabled=False, length=0.15)
        ps_cloud.add_vector_quantity("grad_bc", grad_bc, enabled=True, length=0.15)
        ps_cloud.add_vector_quantity("num_grad_bc", num_grad, enabled=True, length=0.15)
        ps_cloud.add_vector_quantity(
            "backward_grad_bc", bc_vals.grad, enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity("cotan", halfedge_weights, defined_on="edges")

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
                        bc_vals.numpy(), arap_data, deformed.numpy()
                    )
                    ps_mesh.update_vertex_positions(arap_deformed)
                else:
                    energy, deformed = arap_step(
                        verts,
                        deformed,
                        halfedge_weights,
                        halfedges,
                        lap,
                        bc_idx,
                        bc_vals,
                    )
                    ps_mesh.update_vertex_positions(deformed.detach().numpy())
                    ps_mesh.add_scalar_quantity(
                        "energy",
                        energy.numpy(),
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
    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(8)
    torch.set_printoptions(linewidth=200, sci_mode=False)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=Path, help="The path of the mesh to load.")
    args = parser.parse_args()
    main(args.mesh_path)

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from typing import Callable, Literal

import igl
import torch

import iskra.sparse as sp
from iskra.dec import d_01, d_10, laplacian
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.signed_svd import signed_svd
from iskra.sparse_linalg import gmres_solve, min_quadratic_energy
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


def arap_step(
    rest: torch.Tensor,
    deformed: torch.Tensor,
    half_edge_weights: torch.Tensor,
    half_edge_verts: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_verts: torch.Tensor,
):
    n_vertices = rest.shape[0]
    lines = face_index(rest, half_edge_verts)
    half_edge_vecs = lines[..., 1, :] - lines[..., 0, :]

    deformed_lines = face_index(deformed, half_edge_verts)
    deformed_half_edge_vecs = deformed_lines[..., 1, :] - deformed_lines[..., 0, :]

    covs = (
        half_edge_weights[..., None, None]
        * half_edge_vecs[..., None, :]
        * deformed_half_edge_vecs[..., :, None]
    )

    vert_covs = reduce_on_subface(covs, half_edge_verts[:, 0:1], n_vertices, "sum")
    vert_u, _, vert_vt = signed_svd(vert_covs)
    vert_rot = vert_vt.mT @ vert_u.mT
    # Uncomment to debug SVD:
    # vert_rot = vert_covs * 0.0 + torch.eye(3, dtype=vert_u.dtype)[None, :, :].expand(
    #     n_vertices, -1, -1
    # )
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
        weighted_dist, half_edge_verts[:, 0:1], n_vertices, "sum"
    )

    # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
    half_edge_vert_rot = face_index(vert_rot, half_edge_verts)
    half_edge_rot = half_edge_vert_rot.mean(1)

    rotated_signed_edge_vecs = (
        half_edge_weights[:, None] * (half_edge_rot @ half_edge_vecs[..., None])[..., 0]
    )
    rhs = reduce_on_subface(
        rotated_signed_edge_vecs, half_edge_verts[:, 0:1], n_vertices, "sum"
    )

    deformed = min_quadratic_energy(lap, -rhs, bc_idx, bc_verts)
    return vert_energy, deformed


def arap_solve(
    rest: torch.Tensor,
    init: torch.Tensor,
    half_edge_weights: torch.Tensor,
    half_edge_verts: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_verts: torch.Tensor,
    max_iter: int = 100,
    eps: float = 1e-10,
):
    deformed = init
    for _ in range(max_iter):
        energy, deformed_new = arap_step(
            rest,
            deformed,
            half_edge_weights,
            half_edge_verts,
            lap,
            mass,
            bc_idx,
            bc_verts,
        )
        print(
            "ARAP: ", torch.linalg.vector_norm(deformed - deformed_new, axis=-1).sum()
        )
        deformed = deformed_new
        if energy.detach().max().cpu().item() < eps:
            break
    return energy, deformed


def make_arap_vjp(
    rest: torch.Tensor,
    deformed: torch.Tensor,
    half_edge_weights: torch.Tensor,
    half_edge_verts: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_verts: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    with torch.enable_grad():
        bc_verts_g = bc_verts.clone().requires_grad_(True)
        deformed_g = deformed.clone().requires_grad_(True)
        energy, deformed_out = arap_step(
            rest,
            deformed_g,
            half_edge_weights,
            half_edge_verts,
            lap,
            mass,
            bc_idx,
            bc_verts_g,
        )
        deformed_out = deformed_out - deformed_g

        def vjp_deformed(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (deformed_out,),
                (deformed_g),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

        def vjp_bc_verts(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (deformed_out,),
                (bc_verts_g),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

    return vjp_deformed, vjp_bc_verts


def main(mesh_path):
    global optimizing, optim_step, deformed

    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(8)
    torch.set_printoptions(linewidth=200, sci_mode=False)
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, fdtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    bdr_idx = boundary(faces)[:, 0]
    bdr_verts = verts[bdr_idx]

    mesh_center = verts.mean(0, keepdim=True)
    # TODO: fix API here:
    # OGRE:
    # control_idx = torch.tensor([12211, 1262], device=device, dtype=torch.int64)
    # control_verts = 1.2 * (verts[control_idx] - mesh_center) + mesh_center
    # HAND_LOWRES/KOALA
    # control_idx = torch.tensor(
    #     [762, 703, 145, 62],  # , 62, 85, 22, 104, 175, 3225
    #     device=device,
    #     dtype=torch.int64,
    # )
    # control_verts = verts[control_idx]
    # 1.5 * (verts[control_idx] - mesh_center) + mesh_center
    # TET:
    control_idx = torch.tensor([0, 1, 2], device=device, dtype=torch.int64)
    control_verts = verts[control_idx] - 0.1
    # CUBE:
    # control_idx = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64)
    # control_verts = verts[control_idx] - 0.1

    grad_deformed = torch.zeros_like(verts)
    offset = -0.1
    grad_deformed[3] += offset
    # grad_deformed += offset

    bc_idx = torch.cat([bdr_idx, control_idx])
    bc_verts = torch.cat([bdr_verts, control_verts])

    weights = cotan_weights(verts, faces)
    lap, mass = laplacian(verts, faces)
    hodge_1 = sp.diag(weights)
    lap = d_10(faces, dtype=verts.dtype) @ hodge_1 @ d_01(faces, dtype=verts.dtype)
    print("MASS:", mass.to_dense())
    print("COT:", weights)
    print("LAP:", lap.to_dense())
    print("LAP_SUM:", lap.to_dense().sum(-1))

    edges, face_edge, edge_signs = get_subfaces(faces)
    _, edge_verts, vert_signs = get_subfaces(edges)
    # These cannot be actual half-edges from faces because of boundaries:
    half_edge_verts = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    # TODO: to_oriented? to_half_edge?
    # TODO: How to nicely reduce half-edges?
    mass_diag = sp.get_diag(mass).to_dense()

    print(
        mass_diag.shape, half_edge_verts.shape, mass_diag[half_edge_verts[:, 0]].shape
    )
    half_edge_weights = torch.cat([weights, weights], 0)
    # half_edge_mass = 1 / mass_diag[half_edge_verts[:, 0]]
    # half_edge_weights = half_edge_mass * half_edge_weights

    init = verts.clone()
    init[bc_idx] = bc_verts
    energy, deformed = arap_solve(
        verts,
        init,
        half_edge_weights,
        half_edge_verts,
        lap,
        mass,
        bc_idx,
        bc_verts,
    )

    num_eps = 1e-8
    num_grad = torch.zeros_like(init)
    num_jac = torch.zeros([*verts.shape, *bc_verts.shape], dtype=dtype, device=device)
    for i in range(bc_verts.shape[0]):
        for j in range(bc_verts.shape[1]):
            offset = torch.zeros_like(bc_verts)
            offset[i, j] = num_eps
            _, deformed_plus = arap_solve(
                verts,
                init,
                half_edge_weights,
                half_edge_verts,
                lap,
                mass,
                bc_idx,
                bc_verts + offset,
            )
            _, deformed_minus = arap_solve(
                verts,
                init,
                half_edge_weights,
                half_edge_verts,
                lap,
                mass,
                bc_idx,
                bc_verts - offset,
            )
            num_jac[:, :, i, j] = (deformed_plus - deformed_minus) / (2 * num_eps)
    num_jac = num_jac.reshape(verts.numel(), bc_verts.numel())
    num_grad = (grad_deformed.flatten() @ num_jac).reshape(*bc_verts.shape)

    # TODO: Turn into tests:
    vjp_deformed, vjp_bc_verts = make_arap_vjp(
        verts, deformed, half_edge_weights, half_edge_verts, lap, mass, bc_idx, bc_verts
    )
    basis = torch.eye(mesh.n_vertices * 3, device=device, dtype=dtype)
    jac_rows = []
    for i in range(mesh.n_vertices * 3):
        jac_rows.append(vjp_deformed(basis[i].reshape(mesh.n_vertices, 3))[0].flatten())
    jac_verts = torch.stack(jac_rows, 0)
    print("JACOBIAN:\n", jac_verts)

    jac_bc_rows = []
    for i in range(mesh.n_vertices * 3):
        jac_bc_rows.append(
            vjp_bc_verts(basis[i].reshape(mesh.n_vertices, 3))[0].flatten()
        )
    jac_bc = torch.stack(jac_bc_rows, 0)
    print("JACOBIAN:\n", jac_bc)

    print("JACOBIAN FULL:\n", -torch.linalg.solve(jac_verts, jac_bc))

    print("NUM JACOBIAN:\n", num_jac)

    init = torch.randn_like(deformed)
    dl_df = -gmres_solve(
        lambda z: vjp_deformed(z)[0], grad_deformed, init, maxiter=200, tol=1e-12
    )
    # dl_df = fixed_point_solver(
    #     lambda z: grad_deformed + vjp_deformed(z)[0], init, 1000, 0
    # )
    print(
        "AHHHH",
        torch.norm((grad_deformed - vjp_deformed(-dl_df)[0]).flatten()).item(),
        # torch.linalg.vector_norm(, axis=-1).mean(),
    )
    grad_bc = vjp_bc_verts(dl_df)[0]
    print(grad_bc.shape)

    arap_data = igl.ARAPData()
    arap_data.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data.max_iter = 100
    arap = igl.arap_precomputation(
        verts.numpy(), faces.numpy(), 3, bc_idx.numpy(), arap_data
    )
    arap_deformed = igl.arap_solve(bc_verts.numpy(), arap_data, deformed.numpy())

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh_rest = ps.register_surface_mesh(
            "mesh", verts, faces.numpy(), enabled=False
        )
        ps_mesh = ps.register_surface_mesh("deformed", deformed, faces.numpy())
        ps_mesh_arap = ps.register_surface_mesh(
            "arap_deformed", arap_deformed, faces.numpy(), enabled=False
        )
        ps_edges = ps.register_curve_network(
            "edges", deformed, half_edge_verts.numpy(), enabled=False, radius=0.01
        )
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        ps_cloud = ps.register_point_cloud("bc", bc_verts, enabled=True)
        ps_mesh.add_scalar_quantity(
            "energy", energy.numpy(), defined_on="vertices", enabled=True
        )
        ps_mesh.add_vector_quantity(
            "grad_deformed", grad_deformed, length=0.15, enabled=True
        )
        ps_mesh.add_vector_quantity("dl_df", dl_df, enabled=False, length=0.15)
        ps_cloud.add_vector_quantity("grad_bc", grad_bc, enabled=True, length=0.15)
        ps_cloud.add_vector_quantity("num_grad_bc", num_grad, enabled=True, length=0.15)
        ps_edges.add_scalar_quantity("cotan", half_edge_weights, defined_on="edges")

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
                    energy, deformed = arap_step(
                        verts,
                        deformed,
                        half_edge_weights,
                        half_edge_verts,
                        lap,
                        mass,
                        bc_idx,
                        bc_verts,
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
    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()
    main(args.mesh_path)

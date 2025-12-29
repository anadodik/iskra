# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from functools import partial

import torch

import iskra.sparse as sp
from iskra.adjoint import compute_numerical_jacobian, make_solver_layer
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry import triangle_areas
from iskra.mesh import Mesh
from iskra.sparse_linalg import cholespy_factor_and_solve
from iskra.topology import face_index, reduce_on_subface


def rdg_step(
    y: torch.Tensor,
    z: torch.Tensor,
    vert_areas: torch.Tensor,
    grad: torch.Tensor,
    div: torch.Tensor,
    lap: torch.Tensor,
    alpha: torch.Tensor,
    rho: torch.Tensor,
    alphak: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # step 1: u-minimization
    b = vert_areas - sp.matmul(div, y.flatten()) + rho * sp.matmul(div, z.flatten())
    u = cholespy_factor_and_solve(lap, b) / (alpha + rho)

    # step 2: z-minimization
    grad_u = sp.matmul(grad, u).reshape(*z.shape)
    z_new = (1 / rho) * y + grad_u
    norm_z = torch.linalg.vector_norm(z_new, axis=0)
    z_new = z_new / torch.where(norm_z >= 1, norm_z, 1)

    # step 3: dual update
    y_new = y + rho * (alphak * grad_u + (1 - alphak) * z - z_new)

    return y_new, z_new, u


def rdg_solve(
    verts: torch.Tensor,
    faces: torch.Tensor,
    bc_idx: torch.Tensor,
    alpha_hat: float = 0.1,
    alphak: float = 1.7,
    max_iter: int = 200,
) -> torch.Tensor:
    n_vertices = verts.shape[0]
    n_faces = faces.shape[0]
    dtype = verts.dtype
    device = verts.device

    tri_areas = triangle_areas(face_index(verts, faces))
    vert_areas = reduce_on_subface(tri_areas / 3, faces, n_vertices, "sum")
    g_x, g_y, g_z = grad(verts, faces)
    g = sp.cat([g_x, g_y, g_z], 0)
    lap, _ = laplacian(verts, faces)

    alpha = alpha_hat * torch.sqrt(torch.sum(vert_areas))
    rho = 2 * torch.sqrt(torch.sum(vert_areas))
    solver = make_solver_layer(
        partial(rdg_step, alphak=alphak),
        [(0, 0), (1, 1)],
        (2, 3, 4, 5, 6, 7),
        fwd_method="fixed-point",
        fwd_max_iter=max_iter,
        fwd_eps=1e-12,
        bwd_method="gmres",
        bwd_max_iter=max_iter,
        bwd_eps=1e-12,
    )

    unknown_mask = torch.ones([n_vertices], dtype=torch.bool, device=verts.device)
    unknown_mask[bc_idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    vert_areas_unknown = vert_areas[unknown_idx]

    g_unknown = sp.get_slice(g, slice(0, 3 * n_faces), unknown_idx)
    lap_unknown = sp.get_slice(lap, unknown_idx, unknown_idx)
    div_unknown = sp.mul(torch.cat(3 * [tri_areas])[None, :], g_unknown.mT.coalesce())

    u_unknown = torch.zeros([unknown_idx.shape[0]], device=device, dtype=dtype)
    y = torch.zeros([3, n_faces], device=device, dtype=dtype)
    z = torch.zeros([3, n_faces], device=device, dtype=dtype)

    y, z, u_unknown = solver(
        y,
        z,
        vert_areas_unknown,
        g_unknown,
        div_unknown,
        lap_unknown,
        alpha,
        rho,
    )

    b = (
        vert_areas_unknown
        - sp.matmul(div_unknown, y.flatten())
        + rho * sp.matmul(div_unknown, z.flatten())
    )
    u_unknown = cholespy_factor_and_solve(lap_unknown, b) / (alpha + rho)

    u = torch.zeros([n_vertices], device=device, dtype=dtype)
    u[unknown_idx] = u_unknown
    return u


def main(mesh_path):
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, fdtype=dtype, device=device)
    # mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices.to(torch.float64)
    bc_idx = torch.tensor([0], device=device, dtype=torch.int64)

    verts = verts.requires_grad_(True)
    dist = rdg_solve(verts, faces, bc_idx)
    grad_dist = torch.full_like(dist, 2)
    dist.backward(grad_dist)

    num_grad = None
    if verts.shape[0] < 100:
        with torch.no_grad():
            num_jac = compute_numerical_jacobian(
                rdg_solve, 0, 0, 1e-8, verts, faces, bc_idx
            )
            num_grad = (grad_dist.flatten() @ num_jac).reshape(*verts.shape)

    # for i in range(1):
    #     if verts.grad is not None:
    #         verts.grad.zero_()
    #     dist = rdg_admm(verts, faces, bc_idx)
    #     dist.backward(torch.full_like(dist, -2))
    #     verts.data -= 1e-4 * verts.grad

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh = ps.register_surface_mesh(
            "mesh",
            verts.detach().numpy(),
            faces.numpy(),
            transparency=0.4,
            enabled=True,
        )
        ps_mesh.add_scalar_quantity(
            "dist",
            dist.detach().numpy(),
            isolines_enabled=True,
            defined_on="vertices",
            enabled=True,
        )
        ps_verts = ps.register_point_cloud(
            "verts", verts.detach().numpy(), enabled=True, radius=0.001
        )
        ps_verts.add_vector_quantity(
            "grad", verts.grad.numpy(), enabled=True, length=0.15
        )
        if num_grad is not None:
            ps_verts.add_vector_quantity(
                "num_grad", num_grad.numpy(), length=0.15, enabled=True
            )
        ps.register_point_cloud("bc", verts[bc_idx].detach().numpy(), enabled=True)

        def callback():
            if ps.imgui.Button("Compute"):
                pass

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(16)
    torch.set_printoptions(linewidth=200, sci_mode=False)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()
    main(args.mesh_path)

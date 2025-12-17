# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.sparse.linalg import splu

import iskra.sparse as sp
from iskra.adjoint import (
    compute_jacobians,
    compute_numerical_jacobian,
    make_solver_layer,
    make_vjp,
)
from iskra.dec import d_01, d_10, laplacian
from iskra.fem import grad
from iskra.geometry import cotan_weights, triangle_areas
from iskra.mesh import Mesh
from iskra.signed_svd import signed_svd
from iskra.sparse_linalg import (
    cholespy_factor_and_solve,
    gmres_solve,
    min_quadratic_energy,
)
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


def rdg_step(
    y: torch.Tensor,
    z: torch.Tensor,
    u: torch.Tensor,
    vert_areas: torch.Tensor,
    grad: torch.Tensor,
    div: torch.Tensor,
    lap: torch.Tensor,
    rho: float,
    alpha: float,
    alphak: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # step 1: u-minimization
    b = vert_areas - div @ y.flatten() + rho * div @ z.flatten()
    u_new = cholespy_factor_and_solve(lap, b) / (alpha + rho)

    # step 2: z-minimization
    grad_u = (grad @ u_new).reshape(*z.shape)
    z_new = (1 / rho) * y + grad_u
    norm_z = torch.linalg.vector_norm(z_new, axis=0)
    z_new = z_new / torch.where(norm_z >= 1, norm_z, 1)

    # step 3: dual update
    y_new = y + rho * (alphak * grad_u + (1 - alphak) * z - z_new)

    return y_new, z_new, u_new


def rdg_admm(
    verts: torch.Tensor,
    faces: torch.Tensor,
    bc_idx: torch.Tensor,
    alpha_hat: float = 0.1,
    alphak: float = 1.7,
    max_iter: int = 100,
    abstol: float = 1e-5 / 2,
    reltol: float = 1e-2,
) -> torch.Tensor:
    n_vertices = verts.shape[0]
    n_faces = faces.shape[0]
    dtype = verts.dtype
    device = verts.device

    tri_areas = triangle_areas(face_index(verts, faces))
    vert_areas = reduce_on_subface(tri_areas / 3, faces, n_vertices, "sum")
    g_x, g_y, g_z = grad(verts, faces)
    g = torch.vstack([g_x, g_y, g_z]).coalesce()
    lap, _ = laplacian(verts, faces)

    # ADMM parameters
    rho: float = 2 * torch.sqrt(torch.sum(vert_areas)).item()
    alpha: float = alpha_hat * torch.sqrt(torch.sum(vert_areas)).item()

    solver = make_solver_layer(
        partial(rdg_step, rho=rho, alpha=alpha, alphak=alphak),
        [(0, 0), (1, 1)],
        (3,),
        fwd_method="fixed-point",
        fwd_max_iter=max_iter,
        fwd_eps=1e-12,
        bwd_method="gmres",
        bwd_max_iter=200,
        bwd_eps=1e-12,
    )

    unknown_mask = torch.ones([n_vertices], dtype=torch.bool, device=verts.device)
    unknown_mask[bc_idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    vert_areas_unknown = vert_areas[unknown_idx]
    lap_unknown = sp.get_slice(lap, unknown_idx, unknown_idx)
    g_unknown = sp.get_slice(g, slice(0, 3 * n_faces), unknown_idx)
    div_unknown = (torch.cat(3 * [tri_areas])[:, None] * g_unknown).mT

    u_unknown = torch.zeros([unknown_idx.shape[0]], device=device, dtype=dtype)
    y = torch.zeros([3, n_faces], device=device, dtype=dtype)
    z = torch.zeros([3, n_faces], device=device, dtype=dtype)

    y, z, u_unknown = solver(
        y, z, u_unknown, vert_areas_unknown, g_unknown, div_unknown, lap_unknown
    )

    u = torch.zeros([n_vertices], device=device, dtype=dtype)
    u[unknown_idx] = u_unknown
    return u


def main(mesh_path):
    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(mesh_path, fdtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices.to(torch.float64)
    bc_idx = torch.tensor([0], device=device, dtype=torch.int64)

    dist = rdg_admm(verts, faces, bc_idx)

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh = ps.register_surface_mesh(
            "mesh", verts.numpy(), faces.numpy(), enabled=True
        )
        ps_mesh.add_scalar_quantity(
            "dist", dist, defined_on="vertices", enabled=True, isolines_enabled=True
        )

        ps_mesh.add_scalar_quantity(
            "dist", dist, defined_on="vertices", enabled=True, isolines_enabled=True
        )
        ps.register_point_cloud("bc", verts[bc_idx].numpy(), enabled=True)

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
    torch.set_num_threads(8)
    torch.set_printoptions(linewidth=200, sci_mode=False)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()
    main(args.mesh_path)

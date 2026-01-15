# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from argparse import ArgumentParser
from functools import partial
from math import sqrt
from typing import TYPE_CHECKING

import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)

import iskra.sparse as sp
from iskra.adjoint import compute_numerical_jacobian, make_solver_layer
from iskra.dec import laplacian
from iskra.fem import grad, grad_to_div
from iskra.geometry import triangle_areas
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_fn
from iskra.sparse_linalg import (
    CholmodSolver,
    SolverT,
    default_solver,
    linear_solve,
    min_quadratic_energy,
)
from iskra.topology import face_index, reduce_on_subface


@profile_fn(name="rgd_step")
def rdg_step(
    y: torch.Tensor,
    z: torch.Tensor,
    tri_areas_sqrt: torch.Tensor,
    vert_areas: torch.Tensor,
    grad: torch.Tensor,
    div: torch.Tensor,
    lap: torch.Tensor,
    alpha: torch.Tensor,
    rho: torch.Tensor,
    alphak: float,
    lap_solver: SolverT | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # step 1: u-minimization
    div_y = sp.matmul(div, y.flatten())
    div_z = sp.matmul(div, z.flatten())
    b = vert_areas - div_y + rho * div_z
    u = linear_solve(lap, b, solver_fn=lap_solver)[1] / (alpha + rho)

    # step 2: z-minimization
    grad_u = sp.matmul(grad, u).reshape(*z.shape)
    z_new = (1 / rho) * y + grad_u
    norm_z = torch.linalg.vector_norm(z_new, axis=0)
    z_new = z_new / torch.where(norm_z >= 1, norm_z, 1)
    div_z_new = sp.matmul(div, z_new.flatten())

    # step 3: dual update
    y_new = y + rho * (alphak * grad_u + (1 - alphak) * z - z_new)
    # div_y_new = sp.matmul(div, y_new.flatten())

    tri_areas_sqrt_grad_u = tri_areas_sqrt[None, :] * grad_u
    tri_areas_sqrt_z_new = tri_areas_sqrt[None, :] * z_new
    r_norm = torch.linalg.norm(tri_areas_sqrt_grad_u - tri_areas_sqrt_z_new, ord="fro")
    s_norm = rho * torch.linalg.norm((div_z - div_z_new)[:, None], ord="fro")

    # thresh1 = (
    #     sqrt(3 * tri_areas_sqrt.shape[0]) * 5e-6 * torch.sqrt(torch.sum(vert_areas))
    # )
    # thresh2 = sqrt(vert_areas.shape[0]) * 5e-6 * (torch.sum(vert_areas))

    # eps_pri = thresh1 + 1e-2 * torch.maximum(
    #     torch.linalg.norm(tri_areas_sqrt_grad_u, "fro"),
    #     torch.linalg.norm(tri_areas_sqrt_z_new, "fro"),
    # )
    # eps_dual = thresh2 + 1e-2 * torch.linalg.norm(div_y_new[:, None], "fro")
    # print(f"Primal: {eps_pri.item()}\tDual: {eps_dual.item()}")
    mu = 10.0
    if r_norm > mu * s_norm:
        rho *= 2
        print("Rho larger")
    elif s_norm > mu * r_norm:
        rho *= 0.5
        print("Rho smaller")
    else:
        rho = 1.0 * rho
    return y_new, z_new, u, rho


@profile_fn(name="rdg_solve")
def rdg_solve(
    verts: torch.Tensor,
    faces: torch.Tensor,
    bc_idx: torch.Tensor,
    alpha_hat: float = 0.05,
    alphak: float = 1.7,
    fwd_max_iter: int = 2_000,
    bwd_max_iter: int = 600,
) -> torch.Tensor:
    n_vertices = verts.shape[0]
    n_faces = faces.shape[0]
    dtype = verts.dtype
    device = verts.device

    tri_areas = triangle_areas(face_index(verts, faces))
    vert_areas = reduce_on_subface(tri_areas / 3, faces, n_vertices, "sum")
    g: torch.Tensor = grad(verts, faces, stack=True)  # type: ignore
    lap, _ = laplacian(verts, faces)

    alpha = alpha_hat * torch.sqrt(torch.sum(vert_areas))
    rho = 2 * torch.sqrt(torch.sum(vert_areas))

    unknown_idx = sp.index_complement(n_vertices, bc_idx)
    vert_areas_unknown = vert_areas[unknown_idx]
    g_unknown = sp.get_slice(g, None, unknown_idx)
    lap_unknown = sp.get_slice(lap, unknown_idx, unknown_idx)
    div_unknown = grad_to_div(g_unknown, tri_areas)
    g_unknown = g_unknown.to_sparse_csr()
    div_unknown = div_unknown.to_sparse_csr()
    # chol = default_solver(lap_unknown)
    chol = CholmodSolver(lap_unknown)

    solver = make_solver_layer(
        partial(rdg_step, alphak=alphak, lap_solver=chol),
        [(0, 0), (1, 1), (8, 3)],
        (2, 3, 4, 5, 6, 7),
        fwd_method="fixed-point",
        fwd_max_iter=fwd_max_iter,
        fwd_eps=1e-12,
        bwd_method="gmres",
        bwd_max_iter=bwd_max_iter,
        bwd_eps=1e-12,
    )

    u_unknown = torch.zeros([unknown_idx.shape[0]], device=device, dtype=dtype)
    y = torch.zeros([3, n_faces], device=device, dtype=dtype)
    z = torch.zeros([3, n_faces], device=device, dtype=dtype)

    y, z, u_unknown, _ = solver(
        y,
        z,
        torch.sqrt(tri_areas),
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
    u_unknown = linear_solve(lap_unknown, b, solver_fn=chol)[1] / (alpha + rho)

    u = torch.zeros([n_vertices], device=device, dtype=dtype)
    u[unknown_idx] = u_unknown
    return u


class Heightfield(torch.nn.Module):
    def __init__(self, xy: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("xy", xy)
        self.z = torch.nn.Parameter(
            1e-3 * -((xy[:, 0:1] - 0.5) ** 2 + (xy[:, 1:2] - 0.5) ** 2)
        )

        if TYPE_CHECKING:
            self.xy: torch.Tensor

    def forward(self) -> torch.Tensor:
        return torch.cat([self.xy[:, :1], self.z, self.xy[:, 1:]], -1)


class Verts(torch.nn.Module):
    def __init__(self, xyz: torch.Tensor) -> None:
        super().__init__()
        self.xyz = torch.nn.Parameter(xyz)

    def forward(self) -> torch.Tensor:
        return self.xyz


def main(mesh_path):
    dtype = torch.double
    device = "cuda"
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    # mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices.to(torch.float64)
    terrain = Heightfield(verts[:, (0, 2)])
    # terrain = Verts(verts)
    terrain.train()

    start = 0
    # start = 2272
    end = 288

    # side_1 = 78
    # side_2 = 274
    # middle = 176
    centerline = torch.tensor(
        [
            1,
            1044,
            2,
            1074,
            846,
            78,
            92,
            859,
            106,
            872,
            120,
            885,
            134,
            898,
            148,
            911,
            162,
            924,
            176,
            274,
            1015,
            260,
            1002,
            246,
            989,
            232,
            976,
            218,
            963,
            204,
            950,
            190,
            937,
        ],
        device=device,
        dtype=torch.int64,
    )
    target_dist = 1.89953
    lr = 500
    optimizer = torch.optim.SGD(terrain.parameters(), lr=lr)

    with torch.no_grad():
        bc_idx = torch.tensor([start], device=device, dtype=torch.int64)
        start_dist_init = rdg_solve(terrain(), faces, bc_idx)

    global_profiler.dump()

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh = ps.register_surface_mesh(
            "mesh", terrain().detach().cpu().numpy(), faces.cpu().numpy(), enabled=True
        )
        ps_mesh.add_scalar_quantity(
            "dist",
            start_dist_init.detach().cpu().numpy(),
            isolines_enabled=True,
            defined_on="vertices",
            enabled=True,
        )
        ps_verts = ps.register_point_cloud(
            "verts", terrain().detach().cpu().numpy(), enabled=True, radius=0.001
        )
        # ps_verts.add_vector_quantity(
        #     "grad", verts.grad.numpy(), enabled=True, length=0.15
        # )
        ps.register_point_cloud(
            "bc", verts[start : start + 1].detach().cpu().numpy(), enabled=True
        )
        is_optimizing = False
        iteration = 0
        sobolev_factor = 20.0
        smoothing = 0.25
        max_iter = 2

        solver = None

        def callback():
            nonlocal \
                optimizer, \
                terrain, \
                is_optimizing, \
                iteration, \
                sobolev_factor, \
                smoothing, \
                solver
            _, sobolev_factor = ps.imgui.SliderFloat(
                "Curr Frame", sobolev_factor, 0, 100
            )
            if ps.imgui.Button("Compute"):
                is_optimizing = not is_optimizing
            if is_optimizing:
                print(f"Optimizing, iteration {iteration}.")
                iteration += 1
                if iteration > max_iter:
                    global_profiler.dump()
                    ps.unshow()
                    return
                if iteration % 250 == 0:
                    is_optimizing = False
                for _ in range(1):
                    time_start = time.perf_counter()
                    optimizer.zero_grad()
                    bc_idx = torch.tensor([start], device=device, dtype=torch.int64)
                    dist = rdg_solve(terrain(), faces, bc_idx)
                    dist_loss = ((dist[centerline] - target_dist) ** 2).sum()
                    lap, mass = laplacian(terrain(), faces)
                    smooth_loss = sp.matmul(terrain.z.mT, sp.matmul(lap, terrain.z))
                    nonneg_loss = torch.relu(-torch.log(terrain.z + 1)).sum()
                    loss = dist_loss + smoothing * smooth_loss + 0.1 * nonneg_loss
                    print(
                        f"Loss={loss.item():.6f} "
                        f"(dist_loss={dist_loss.item():.6f}, "
                        f"smooth_loss={smooth_loss.item():.6f}, "
                        f"nonneg_loss={nonneg_loss.item():.6f})."
                    )
                    time_end = time.perf_counter()
                    print(f"Forward took: {time_end - time_start}s.")
                    time_start = time.perf_counter()
                    loss.backward()
                    time_end = time.perf_counter()
                    print(f"Backward took: {time_end - time_start}s.")

                    with torch.no_grad():
                        mat = mass + sobolev_factor * lap
                        assert terrain.z.grad is not None
                        solver, z_grad = min_quadratic_energy(
                            mat,
                            sp.matmul(mass, terrain.z.grad),
                            torch.tensor([start], device=device),
                            torch.tensor([[0.0]], dtype=dtype, device=device),
                            solver=solver,
                        )

                        new_lr = lr
                        max_z_grad = z_grad.abs().max()
                        while new_lr * max_z_grad > 0.1:
                            new_lr *= 0.5
                            print(new_lr)
                        for g in optimizer.param_groups:
                            g["lr"] = new_lr
                        # z_grad = torch.clamp(z_grad, -0.1 / lr, 0.1 / lr)
                        # z_grad -= z_grad.mean(0, keepdim=True)
                        terrain.z.grad = z_grad
                    optimizer.step()

                ps_mesh.update_vertex_positions(terrain().detach().cpu().numpy())
                ps_verts.update_point_positions(terrain().detach().cpu().numpy())
                ps_mesh.add_scalar_quantity(
                    "dist",
                    dist.detach().cpu().numpy(),
                    isolines_enabled=True,
                    defined_on="vertices",
                    enabled=True,
                )
                ps_mesh.add_scalar_quantity(
                    "-grad dist",
                    -z_grad.flatten().detach().cpu().numpy(),
                    isolines_enabled=True,
                    defined_on="vertices",
                    enabled=False,
                )
                import igl

                igl.write_triangle_mesh(
                    f"results/terrain_smooth={smoothing}.obj",
                    terrain().detach().cpu().cpu().numpy(),
                    mesh.faces.cpu().cpu().numpy(),
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
    torch.set_printoptions(linewidth=200, sci_mode=False, precision=6)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()
    main(args.mesh_path)

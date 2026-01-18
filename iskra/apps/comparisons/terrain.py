# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
from argparse import ArgumentParser, Namespace
from functools import partial
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import torch
from torch.linalg import norm

torch.set_num_threads(32)
torch.set_num_interop_threads(32)

import cvxpy as cp
import numpy as np

# import polyscope as ps
import sparse_solver
import theseus as th
from cvxpylayers.torch import CvxpyLayer
from networkx import center

import iskra.sparse as sp
from iskra import dec
from iskra.adjoint import make_solver_layer
from iskra.fem import grad, grad_to_div
from iskra.geometry.volume import triangle_areas
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block, profile_fn
from iskra.sparse_linalg import (
    CholmodSolver,
    CUDSSSolver,
    SolverT,
    default_solver,
    linear_solve,
    min_quadratic_energy,
)
from iskra.topology import face_index, reduce_on_subface


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
    lap_solver: SolverT | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    alphak = 1.7
    abstol = 1e-6
    reltol = 1e-6
    nv, nf = vert_areas.shape[0], tri_areas_sqrt.shape[0]
    va_sum = vert_areas.sum()
    thresh1 = sqrt(3 * nf) * abstol * va_sum.sqrt()
    thresh2 = sqrt(nv) * abstol * va_sum
    rho = 1.0 * rho

    # step 1: u-minimization
    div_y = sp.matmul(div, y.flatten())
    div_z = sp.matmul(div, z.flatten())
    b = vert_areas - div_y + rho * div_z
    u = linear_solve(lap, b, solver_fn=lap_solver)[1] / (alpha + rho)

    # step 2: z-minimization
    grad_u = sp.matmul(grad, u).reshape(*z.shape)
    z_new = (1 / rho) * y + grad_u
    norm_z = norm(z_new, axis=0)
    z_new = z_new / torch.where(norm_z >= 1, norm_z, 1)

    # step 3: dual update
    y_new = y + rho * (alphak * grad_u + (1 - alphak) * z - z_new)

    div_y_new = sp.matmul(div, y_new.flatten())
    div_z_new = sp.matmul(div, z_new.flatten())
    tas_grad_u = tri_areas_sqrt[None, :] * grad_u
    tas_z_new = tri_areas_sqrt[None, :] * z_new
    r_norm = norm(tas_grad_u - tas_z_new)
    s_norm = rho * norm(div_z - div_z_new)
    error = torch.stack([r_norm, s_norm])
    # print(f"Primal: {r_norm.item()}\tDual: {s_norm.item()}.")

    eps_pri = thresh1 + reltol * torch.max(norm(tas_grad_u), norm(tas_z_new))
    eps_dual = thresh2 + reltol * norm(div_y_new)
    if r_norm < eps_pri and s_norm < eps_dual:
        # print(f"eps_pri: {eps_pri.item()}\teps_dual: {eps_dual.item()}.")
        error = torch.tensor(0)
    if r_norm > 10.0 * s_norm:
        rho *= 2
        # print("Increasing rho.")
    elif s_norm > 10.0 * r_norm:
        rho *= 0.5
        # print("Decreasing rho.")
    return y_new, z_new, u, rho, error


def extract_grad_coeffs(grad_sparse, faces):
    """Given a sparse gradient matrix (nf, nv) with 3 nnz per row, extract coefficients as a dense (nf, 3) array aligned with face vertex order.

    Instead of storing g as a sparse matrix, store the gradient coefficients
    gx, gy, gz each have 3 non-zeros per row (one per vertex of each face)
    We can represent this as dense arrays of shape (nf, 3) plus the face indices

    The face indices already tell us which vertices contribute to each face
    faces: (nf, 3) - indices of vertices for each face
    For the gradient, we need the coefficients. Let's extract them:
    """
    grad_sparse = grad_sparse.coalesce()
    nf = faces.shape[0]
    coeffs = torch.zeros(nf, 3, dtype=grad_sparse.dtype, device=grad_sparse.device)

    indices = grad_sparse.indices()  # (2, nnz)
    values = grad_sparse.values()  # (nnz,)

    rows = indices[0]
    cols = indices[1]

    for i in range(3):
        vertex_ids = faces[:, i]  # (nf,) - the i-th vertex of each face
        # Find entries where col matches vertex_ids[row]
        mask = cols == vertex_ids[rows]
        coeffs[rows[mask], i] = values[mask]

    return coeffs


def iskra_setup(
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
) -> SolverT:
    return None


gmres_init = None


def iskra_forward(
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    solver: SolverT,
) -> torch.Tensor:
    nv, nf = verts.shape[0], faces.shape[0]
    dtype, device = verts.dtype, verts.device
    tri_areas = triangle_areas(face_index(verts, faces))
    vert_areas = reduce_on_subface(tri_areas / 3, faces, nv, "sum")
    alpha = 0.05 * torch.sqrt(torch.sum(vert_areas))
    rho = 2 * torch.sqrt(torch.sum(vert_areas))

    g: torch.Tensor = grad(verts, faces, stack=True)  # type: ignore

    free_idx = sp.index_complement(nv, start_idcs)
    vert_areas_free = vert_areas[free_idx]
    g_free = sp.get_slice(g, None, free_idx)
    lap_free = sp.get_slice(lap, free_idx, free_idx)
    div_free = grad_to_div(g_free, tri_areas).to_sparse_csr()
    g_free = g_free.to_sparse_csr()

    if lap_free.is_cpu:
        solver = CholmodSolver(lap_free)
    else:
        solver = default_solver(lap_free)

    # solver.refactor_numeric(mat)
    # faired = linear_solve(mat, sp.matmul(mass, verts), solver_fn=solver)[1]
    def callback_gmres_sol(sol):
        global gmres_init
        gmres_init = sol

    geodesic_layer = make_solver_layer(
        partial(rdg_step, lap_solver=solver),
        [(0, 0), (1, 1), (8, 3)],
        (2, 3, 4, 5, 6, 7),
        fwd_method="fixed-point",
        fwd_max_iter=1_000,
        fwd_eps=1e-12,
        fwd_error_arg=4,
        fwd_error_tol=1e-12,
        bwd_method="gmres",
        gmres_init=gmres_init,
        # callback_gmres_sol=callback_gmres_sol,
        bwd_max_iter=600,
        bwd_eps=1e-4,
    )
    u_unknown = torch.zeros([free_idx.shape[0]], device=device, dtype=dtype)
    y = torch.zeros([3, nf], device=device, dtype=dtype)
    z = torch.zeros([3, nf], device=device, dtype=dtype)

    y, z, u_unknown, _, _ = geodesic_layer(
        y, z, tri_areas.sqrt(), vert_areas_free, g_free, div_free, lap_free, alpha, rho
    )

    # Only necessary because of gradients:
    b = (
        vert_areas_free
        - sp.matmul(div_free, y.flatten())
        + rho * sp.matmul(div_free, z.flatten())
    )
    u_unknown = linear_solve(lap_free, b, solver_fn=solver)[1] / (alpha + rho)

    u = torch.zeros([nv], device=device, dtype=dtype)
    u[free_idx] = u_unknown
    return u


def cvxpylayers_setup(
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
) -> CvxpyLayer:
    with profile_block("making_layer"):
        nv, nf = verts.shape[0], faces.shape[0]
        u = cp.Variable(nv)
        va_param = cp.Parameter(nv)
        gx_param = cp.Parameter((nf, 3))
        gy_param = cp.Parameter((nf, 3))
        gz_param = cp.Parameter((nf, 3))

        lap_x_sqrt_param = cp.Parameter((nf, 3))
        lap_y_sqrt_param = cp.Parameter((nf, 3))
        lap_z_sqrt_param = cp.Parameter((nf, 3))

        u_faces = u[faces]
        grad_x_expr = cp.sum(cp.multiply(gx_param, u_faces), axis=1)
        grad_y_expr = cp.sum(cp.multiply(gy_param, u_faces), axis=1)
        grad_z_expr = cp.sum(cp.multiply(gz_param, u_faces), axis=1)
        grad_vec = cp.bmat([[grad_x_expr], [grad_y_expr], [grad_z_expr]]).T

        lap_x_expr = cp.sum(cp.multiply(lap_x_sqrt_param, u_faces), axis=1)
        lap_y_expr = cp.sum(cp.multiply(lap_y_sqrt_param, u_faces), axis=1)
        lap_z_expr = cp.sum(cp.multiply(lap_z_sqrt_param, u_faces), axis=1)

        # ta_param *
        reg = (
            cp.sum_squares(lap_x_expr)
            + cp.sum_squares(lap_y_expr)
            + cp.sum_squares(lap_z_expr)
        )
        obj = cp.Maximize(va_param @ u - 0.5 * cp.sum(reg))

        constraints = [
            u[start_idcs] == 0,
            cp.pnorm(grad_vec, p=2, axis=1) <= 1,
        ]

        problem = cp.Problem(obj, constraints)

        layer = CvxpyLayer(
            problem,
            parameters=[
                va_param,
                gx_param,
                gy_param,
                gz_param,
                lap_x_sqrt_param,
                lap_y_sqrt_param,
                lap_z_sqrt_param,
            ],
            variables=[u],
        )
    return layer


def cvxpylayers_forward(
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    layer: CvxpyLayer,
) -> torch.Tensor:
    nv, nf = verts.shape[0], faces.shape[0]
    dtype, device = verts.dtype, verts.device

    # va_param.value = vert_areas.detach().numpy()
    # alpha_param.value = alpha.detach().numpy()
    # gx_param.value = gx_coeffs.detach().numpy()
    # gy_param.value = gy_coeffs.detach().numpy()
    # gz_param.value = gz_coeffs.detach().numpy()
    # problem.solve(
    #     verbose=True,
    # )
    # solver_args={
    #     "max_iter": 2_000,
    #     "tol_feas": 0,
    #     "tol_gap_abs": 0,
    #     "tol_gap_rel": 0,
    #     "verbose": True,
    # },
    with profile_block("calling_layer"):
        tri_areas = triangle_areas(face_index(verts, faces))
        tri_areas_sqrt = tri_areas.sqrt()
        vert_areas = reduce_on_subface(tri_areas / 3, faces, nv, "sum")

        alpha = 0.05 * torch.sqrt(torch.sum(vert_areas))
        alpha_sqrt = alpha.sqrt()
        gx, gy, gz = grad(verts, faces)  # type: ignore
        gx_coeffs = extract_grad_coeffs(gx, faces)
        gy_coeffs = extract_grad_coeffs(gy, faces)
        gz_coeffs = extract_grad_coeffs(gz, faces)
        (u,) = layer(
            vert_areas,
            gx_coeffs,
            gy_coeffs,
            gz_coeffs,
            alpha_sqrt * tri_areas_sqrt[:, None] * gx_coeffs,
            alpha_sqrt * tri_areas_sqrt[:, None] * gy_coeffs,
            alpha_sqrt * tri_areas_sqrt[:, None] * gz_coeffs,
            # solver_args={"eps_abs": 1e-10, "eps_rel": 1e-10, "verbose": True},
        )

    return u


FN_MAP = {
    "iskra": [iskra_setup, iskra_forward],
    "cvxpylayers": [cvxpylayers_setup, cvxpylayers_forward],
}


def objective(
    u: torch.Tensor, alpha: torch.Tensor, mass: torch.Tensor, lap: torch.Tensor
):
    objective = -mass @ u + 0.5 * alpha * sp.matmul(sp.matmul(u.mT, lap), u)
    return objective


def optimize(
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    centerline_idcs: torch.Tensor,
    target_dist: float,
    sigma: float,
    lr: float,
    method: str,
):
    terrain = Heightfield(verts[:, (0, 2)])
    terrain.train()
    optim = torch.optim.SGD(terrain.parameters(), lr=lr)
    lap, mass = dec.laplacian(verts, faces)
    d_01 = dec.d_01(faces, dtype=verts.dtype)
    eye = sp.eye(verts.shape[0], device=verts.device, dtype=verts.dtype)
    h1_mat = (mass + 20.0 * lap).coalesce()
    free_idx = sp.index_complement(verts.shape[0], start_idcs)
    h1_free = sp.get_slice(h1_mat, free_idx, free_idx)
    h1_solver = default_solver(h1_free)
    start_val = torch.zeros(
        [start_idcs.shape[0], 1], dtype=verts.dtype, device=verts.device
    )

    setup_fn, forward_fn = FN_MAP[method]
    with profile_block(method):
        with profile_block("setup"):
            data = setup_fn(verts, faces, start_idcs, lap, mass)
        for _ in range(500):
            optim.zero_grad()
            with profile_block("forward"):
                verts_param = terrain()
                with profile_block("operators"):
                    cots = dec.hodge_1(verts_param, faces, clamp_min=1e-4)
                    lap = sp.matmul(d_01.mT, sp.matmul(cots, d_01))
                    lap = (lap + 1e-7 * eye).coalesce()
                    mass = dec.hodge_0(verts_param, faces)

                dist = forward_fn(verts_param, faces, start_idcs, lap, mass, data)
                dist_term = ((dist[centerline_idcs] - target_dist) ** 2).sum()
                lap = lap.to_sparse_csr()
                smooth_term = sp.matmul(terrain.z.mT, sp.matmul(lap, terrain.z))
                nonneg_term = torch.relu(-torch.log(terrain.z + 1)).sum()
                loss = dist_term + sigma * smooth_term + 0.1 * nonneg_term
                print(f"Loss: {loss.detach().cpu().item()}")

                # vert_areas = sp.get_diag(mass)
                # alpha = 0.05 * torch.sqrt(torch.sum(vert_areas))
                # obj = objective(dist[:, None], alpha, sp.get_diag(mass).to_dense(), lap)
                # print(obj.item())
            with profile_block("backward"):
                loss.backward()
            if terrain.z.grad is None:
                raise RuntimeError("terrain.z.grad is None!")
            with profile_block("h1"):
                with torch.no_grad():
                    rhs = sp.matmul(mass, terrain.z.grad)
                    _, z_grad = min_quadratic_energy(
                        h1_mat, rhs, start_idcs, start_val, solver=h1_solver
                    )
                    terrain.z.grad = z_grad
                    new_lr = lr
                    max_z_grad = z_grad.abs().max()
                    while new_lr * max_z_grad > 0.1:
                        new_lr *= 0.5
                    for g in optim.param_groups:
                        g["lr"] = new_lr
            optim.step()
    return terrain, dist


def read_numbers(
    path: Path, device: str | torch.device, dtype: torch.dtype
) -> torch.Tensor:
    with path.open("r") as f:
        idcs = torch.tensor(
            [float(i) for i in f.readline().split(", ")], device=device, dtype=dtype
        )
    return idcs


def loop_subdivide(
    verts: torch.Tensor, faces: torch.Tensor, iterations: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = verts.dtype, verts.device
    import igl

    if iterations < 0:
        raise ValueError(
            f"Cannot subdivide a negative number of iterations: {iterations}."
        )
    faces_np = faces.cpu().numpy()
    verts_np = verts.cpu().numpy()
    for _ in range(0, iterations):
        subdiv_matrix_new, faces_np = igl.loop_matrix(faces_np, verts_np.shape[0])
        verts_np = subdiv_matrix_new @ verts_np
    verts = torch.tensor(verts_np, dtype=dtype, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int64, device=device)
    return verts, faces


@torch.no_grad()
def plot_optimized(
    results_dir,
    verts: torch.Tensor,
    faces: torch.Tensor,
    start_idcs: torch.Tensor,
    centerline_idcs: torch.Tensor,
    target_dist: torch.Tensor,
    method: str,
    sigma: float,
):
    lap, mass = dec.laplacian(verts, faces)
    d_01 = dec.d_01(faces, dtype=verts.dtype)
    eye = sp.eye(verts.shape[0], device=verts.device, dtype=verts.dtype)
    setup_fn, forward_fn = FN_MAP[method]
    data = setup_fn(verts, faces, start_idcs, lap, mass)
    cots = dec.hodge_1(verts, faces, clamp_min=1e-4)
    lap = sp.matmul(d_01.mT, sp.matmul(cots, d_01))
    lap = (lap + 1e-7 * eye).coalesce()
    mass = dec.hodge_0(verts, faces)
    dist = forward_fn(verts, faces, start_idcs, lap, mass, data)
    from gemvis import Plot

    from iskra.mesh import BBox

    verts = verts[:, (0, 2)].to(torch.float32)
    bbox = BBox.compute(verts)
    bbox.min -= 0.025
    bbox.max += 0.025
    # max_extent = torch.max(bbox.extent)
    # verts = (verts - bbox.min) / max_extent
    # print(verts)

    plot = Plot(
        (1440, 1440), headless=True, projection="ortho", ground=False, bbox=bbox
    )
    plot.camera.rigid = plot.camera_views[0]
    plot.set_background_color(1.0, 1.0, 1.0, 0.0)
    n_contours = 14
    height_plot = plot.plot_mesh(verts, faces)
    height_plot.set_lighting(False)
    height_plot.set_cmap("teal_rose")
    height_plot.set_contour(True)
    height_plot.set_contour_levels(n_contours)
    height_plot.cull_face(False)
    height_plot.depth_test(False)
    varnorm = mpl.colors.CenteredNorm(halfrange=target_dist.detach().cpu().numpy())
    center_norm_variation = varnorm((dist - target_dist).detach().cpu().numpy())
    height_plot.set_values(center_norm_variation, normalize=False)

    start_plot = plot.plot_points(verts[start_idcs])
    start_plot.depth_test(False)
    start_plot.cull_face(False)
    start_plot.set_colors(start_plot.qualitative_cmap[-2])
    start_plot.set_point_scale(0.04)

    start_plot = plot.plot_points(verts[centerline_idcs])
    start_plot.depth_test(False)
    start_plot.cull_face(False)
    start_plot.set_colors(start_plot.qualitative_cmap[1])
    start_plot.set_point_scale(0.02)

    plot.render()
    plot.save_screenshot(results_dir / f"distances_{sigma}.png")


def plot_main(sigma: float):
    device = "cpu"
    dtype = torch.float32
    input_dir = Path().home() / "Dropbox" / "Data" / "iskra-data" / "terrain"
    results_dir = Path().home() / "Dropbox" / "Results" / "iskra" / "terrain"
    # mesh_path = results_dir / f"terrain_smooth={sigma}.obj"
    mesh_path = (
        "/home/anadodik/experiments/iskra/terrain/s0_iskra_cpu_float32/result.obj"
    )
    mesh, _ = Mesh.from_path(mesh_path, device=device)
    # mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices
    # start_idcs = read_numbers(input_dir / "start.txt", device, torch.int64)
    # centerline_idcs = read_numbers(input_dir / "centerline.txt", device, torch.int64)
    # target_dist = read_numbers(input_dir / "target_dist.txt", device, dtype)

    verts_2d = verts[:, (0, 2)]
    eps = 0.012
    start_idcs = torch.where((verts_2d < eps).all(-1))[0]
    centerline_idcs = torch.where(((verts_2d.sum(-1) - 1).abs() < eps))[0]
    target_dist = torch.cdist(verts_2d[start_idcs], verts_2d[centerline_idcs]).max()
    plot_optimized(
        results_dir,
        verts,
        faces,
        start_idcs,
        centerline_idcs,
        target_dist,
        "iskra",
        sigma,
    )


def main(
    subdiv: int,
    input_path: Path,
    results_dir: Path,
    method: str,
    sigma: float,
    device_name: str,
    dtype_name: str,
):
    torch.set_num_threads(16)
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    mesh, _ = Mesh.from_path(input_path / "plane.obj", device=device, dtype=dtype)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices
    verts, faces = loop_subdivide(verts, faces, subdiv)

    verts_2d = verts[:, (0, 2)]
    eps = 0.012
    start_idcs = torch.where((verts_2d < eps).all(-1))[0]
    centerline_idcs = torch.where(((verts_2d.sum(-1) - 1).abs() < eps))[0]
    furthest = (
        torch.cdist(verts_2d[start_idcs], verts_2d[centerline_idcs])
        .max(dim=0)[0]
        .argmax()
    )
    # target_dist = torch.cdist(verts_2d[start_idcs], verts_2d[centerline_idcs]).max()
    lap, mass = dec.laplacian(verts, faces)
    eye = sp.eye(verts.shape[0], device=verts.device, dtype=verts.dtype)
    lap = (lap + 1e-7 * eye).coalesce()
    dist = iskra_forward(verts, faces, start_idcs, lap, mass, None)
    target_dist = dist[centerline_idcs].max()
    print(start_idcs)
    print(centerline_idcs)
    print(target_dist, furthest)

    with Path(results_dir, "profile.json").open("w") as f:
        f.write("{}")

    terrain, dist = optimize(
        verts, faces, start_idcs, centerline_idcs, target_dist, sigma, 500, method
    )

    if method == "iskra":
        plot_optimized(
            results_dir,
            terrain(),
            faces,
            start_idcs,
            centerline_idcs,
            target_dist,
            method,
            sigma,
        )

    with Path(results_dir, "profile.json").open("w") as f:
        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
    global_profiler.dump(path=Path(results_dir, "profile"))

    # COMPUTE OUTPUT
    import igl

    igl.writeOBJ(
        results_dir / "result.obj",
        terrain().detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
    )
    return

    try:
        ps.init()
        ps.register_surface_mesh(
            "init", verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
        )
        ps_mesh = ps.register_surface_mesh(
            "terrain",
            terrain().detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )
        ps_mesh.add_scalar_quantity(
            "dist",
            dist.detach().cpu().numpy(),
            isolines_enabled=True,
            defined_on="vertices",
            enabled=True,
        )
        ps.register_point_cloud(
            "start_idcs",
            verts[start_idcs].detach().cpu().numpy(),
            enabled=True,
            radius=0.001,
        )
        ps.register_point_cloud(
            "centerline_idcs",
            verts[centerline_idcs].detach().cpu().numpy(),
            enabled=True,
            radius=0.001,
        )
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    """
    python -m iskra.apps.comparisons.terrain
    """
    torch.manual_seed(620)
    torch.cuda.manual_seed_all(620)

    input_dir = Path().home() / "Dropbox" / "Data" / "iskra-data" / "terrain"
    default_results_dir = Path().home() / "experiments" / "iskra" / "terrain"
    default_results_dir.mkdir(exist_ok=True, parents=True)

    parser = ArgumentParser(description="Demonstrates mesh inflation using inverse GP.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=default_results_dir,
        help="Where to save outputs.",
    )
    parser.add_argument("--sigma", type=float, default=0.25, help="Smoothing.")
    parser.add_argument(
        "--method",
        type=str,
        default="iskra",
        help="Which method to use in (iskra, cvxpylayers).",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Mesh data type.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    parser.add_argument("--subdiv", type=int, default=0, help="Number of subdivisions.")
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()
    if args.plot:
        plot_main(args.sigma)
        exit()
    result_dir = (
        args.results_dir
        / f"s{args.subdiv}_{args.sigma:.2f}_{args.method}_{args.device}_{args.dtype}"
    )
    result_dir.mkdir(exist_ok=True, parents=True)
    main(
        args.subdiv,
        input_dir,
        result_dir,
        args.method,
        args.sigma,
        args.device,
        args.dtype,
    )

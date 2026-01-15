# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
import polyscope as ps
import sparse_solver
import theseus as th
import torch
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
    lap_solver: SolverT | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alphak = 1.7
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

    tri_areas_sqrt_grad_u = tri_areas_sqrt[None, :] * grad_u
    tri_areas_sqrt_z_new = tri_areas_sqrt[None, :] * z_new
    r_norm = torch.linalg.norm(tri_areas_sqrt_grad_u - tri_areas_sqrt_z_new, ord="fro")
    s_norm = rho * torch.linalg.norm((div_z - div_z_new)[:, None], ord="fro")
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


def iskra_setup(verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor) -> SolverT:
    return None


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

    solver = CholmodSolver(lap_free)
    # solver.refactor_numeric(mat)
    # faired = linear_solve(mat, sp.matmul(mass, verts), solver_fn=solver)[1]

    geodesic_layer = make_solver_layer(
        partial(rdg_step, lap_solver=solver),
        [(0, 0), (1, 1), (8, 3)],
        (2, 3, 4, 5, 6, 7),
        fwd_method="fixed-point",
        fwd_max_iter=2_000,
        fwd_eps=1e-12,
        bwd_method="gmres",
        bwd_max_iter=600,
        bwd_eps=1e-12,
    )
    u_unknown = torch.zeros([free_idx.shape[0]], device=device, dtype=dtype)
    y = torch.zeros([3, nf], device=device, dtype=dtype)
    z = torch.zeros([3, nf], device=device, dtype=dtype)

    y, z, u_unknown, _ = geodesic_layer(
        y, z, tri_areas.sqrt(), vert_areas_free, g_free, div_free, lap_free, alpha, rho
    )

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
    verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor
) -> th.TheseusLayer:
    pass


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
    tri_areas = triangle_areas(face_index(verts, faces))
    vert_areas = reduce_on_subface(tri_areas / 3, faces, nv, "sum")

    alpha = 0.05 * torch.sqrt(torch.sum(vert_areas))
    gx, gy, gz = grad(verts, faces)  # type: ignore

    u = cp.Variable(nv)
    va_param = cp.Parameter(nv)
    alpha_param = cp.Parameter(nonneg=True)

    lap_sp_scipy = cp.psd_wrap(sp.torch_to_scipy(lap))
    obj = cp.Maximize(va_param @ u - alpha_param * cp.quad_form(u, lap_sp_scipy))

    gx_coeffs = extract_grad_coeffs(gx, faces)
    gy_coeffs = extract_grad_coeffs(gy, faces)
    gz_coeffs = extract_grad_coeffs(gz, faces)
    u_faces = u[faces.cpu().numpy()]
    grad_x_expr = cp.sum(cp.multiply(gx_coeffs.detach().numpy(), u_faces), axis=1)
    grad_y_expr = cp.sum(cp.multiply(gy_coeffs.detach().numpy(), u_faces), axis=1)
    grad_z_expr = cp.sum(cp.multiply(gz_coeffs.detach().numpy(), u_faces), axis=1)
    grad_vec = cp.bmat([[grad_x_expr], [grad_y_expr], [grad_z_expr]]).T

    constraints = [
        u[start_idcs] == 0,
        cp.pnorm(grad_vec, p=2, axis=1) <= 1,
    ]

    problem = cp.Problem(obj, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[va_param, alpha_param],
        variables=[u],
    )
    (u,) = layer(vert_areas, alpha)
    return u


FN_MAP = {
    "iskra": [iskra_setup, iskra_forward],
    "cvxpylayers": [cvxpylayers_setup, cvxpylayers_forward],
}


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
    h1_solver = default_solver(mass + 20.0 * lap)

    setup_fn, forward_fn = FN_MAP[method]
    with profile_block(method):
        with profile_block("setup"):
            data = setup_fn(verts, lap, mass)
        for _ in range(1):
            optim.zero_grad()
            with profile_block("forward"):
                verts_param = terrain()
                with profile_block("operators"):
                    cots = dec.hodge_1(verts_param, faces)
                    lap = sp.matmul(d_01.mT, sp.matmul(cots, d_01))
                    mass = dec.hodge_0(verts_param, faces)

                dist = forward_fn(terrain(), faces, start_idcs, lap, mass, data)
                dist_term = ((dist[centerline_idcs] - target_dist) ** 2).sum()
                lap = lap.to_sparse_csr()
                smooth_term = sp.matmul(terrain.z.mT, sp.matmul(lap, terrain.z))
                nonneg_term = torch.relu(-torch.log(terrain.z + 1)).sum()
                loss = dist_term + sigma * smooth_term + 0.1 * nonneg_term
            with profile_block("backward"):
                loss.backward()
            if terrain.z.grad is None:
                raise RuntimeError("terrain.z.grad is None!")
            with profile_block("h1"):
                with torch.no_grad():
                    terrain.z.grad = h1_solver(sp.matmul(mass, terrain.z.grad))
            # optim.step()
    return terrain, dist


def read_numbers(
    path: Path, device: str | torch.device, dtype: torch.dtype
) -> torch.Tensor:
    with path.open("r") as f:
        idcs = torch.tensor(
            [float(i) for i in f.readline().split(", ")], device=device, dtype=dtype
        )
    return idcs


def main(
    input_path: Path,
    results_dir: Path,
    method: str,
    sigma: float,
    device_name: str,
    dtype_name: str,
):
    torch.set_num_threads(16)
    torch.autograd.detect_anomaly(True)
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    mesh, _ = Mesh.from_path(input_path / "plane.obj", device=device, dtype=dtype)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices
    start_idcs = read_numbers(input_path / "start.txt", device, torch.int64)
    centerline_idcs = read_numbers(input_path / "centerline.txt", device, torch.int64)
    target_dist = read_numbers(input_path / "target_dist.txt", device, dtype)
    print(start_idcs)
    print(centerline_idcs)
    print(target_dist)

    with Path(results_dir, "profile.json").open("w") as f:
        f.write("{}")

    terrain, dist = optimize(
        verts, faces, start_idcs, centerline_idcs, target_dist, sigma, 500, method
    )

    with Path(results_dir, "profile.json").open("w") as f:
        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
    global_profiler.dump(path=Path(results_dir, "profile"))

    # COMPUTE OUTPUT

    import igl

    # igl.writeOBJ(
    #     results_dir / f"inflated_{mesh_path.stem}_{method}_{device}.obj",
    #     inflated.detach().cpu().numpy(),
    #     faces.detach().cpu().numpy(),
    # )
    # igl.writeOBJ(
    #     results_dir / f"init_{mesh_path.stem}_{method}_{device}.obj",
    #     verts.detach().cpu().numpy(),
    #     faces.detach().cpu().numpy(),
    # )

    try:
        ps.init()
        ps.register_surface_mesh(
            "init", verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
        )
        # ps_mesh = ps.register_surface_mesh(
        #     "terrain",
        #     terrain().detach().cpu().numpy(),
        #     faces.detach().cpu().numpy(),
        # )
        # ps_mesh.add_scalar_quantity(
        #     "dist",
        #     dist.detach().cpu().numpy(),
        #     isolines_enabled=True,
        #     defined_on="vertices",
        #     enabled=True,
        # )
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
    args = parser.parse_args()
    main(input_dir, args.results_dir, args.method, args.sigma, args.device, args.dtype)

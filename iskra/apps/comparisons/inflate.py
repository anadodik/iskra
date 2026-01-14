# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
import sparse_solver
import theseus as th
import torch

import iskra.sparse as sp
from iskra import dec
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block
from iskra.sparse_linalg import (
    CholmodSolver,
    CUDSSSolver,
    _linear_solver_fn,
    linear_solve,
)


def iskra_setup(
    verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor, t: float
) -> CholmodSolver | CUDSSSolver:
    if verts.is_cpu:
        return CholmodSolver(mass + t * lap, analyze_only=True)
    else:
        return CUDSSSolver(mass + t * lap, analyze_only=True)


def iskra_forward(
    verts: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    t: float,
    solver: CholmodSolver,
) -> torch.Tensor:
    mat = (mass + t * lap).coalesce()
    # TODO: figure this out:
    mass = mass.to_sparse_csr()
    with profile_block(name="solver"):
        solver.refactor_numeric(mat)
        faired = linear_solve(mat, sp.matmul(mass, verts), solver_fn=solver)[1]
    return faired


def alec_setup(
    verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor, t: float
) -> None:
    return


def alec_forward(
    verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor, t: float, data: Any
) -> torch.Tensor:
    mat = (mass + t * lap).to(torch.float64).coalesce()
    mass = mass.to_sparse_csr()
    faired = torch.empty_like(verts)
    rhs = sp.matmul(mass, verts).to(torch.float64)
    with profile_block("solver"):
        for i in range(verts.shape[-1]):
            solved_i = sparse_solver.SparseSolver.apply(mat, rhs[:, i].clone())
            faired[:, i] = solved_i.to(verts.dtype)
    return faired


def theseus_setup(
    verts: torch.Tensor, lap: torch.Tensor, mass: torch.Tensor, t: float
) -> th.TheseusLayer:
    def fairing_residual(optim_vars, aux_vars):
        (verts_var,) = optim_vars
        verts_target_var, lap_sqrt_var, mass_sqrt_var = aux_vars

        verts = verts_var.tensor.reshape(-1, 3)
        verts_target = verts_target_var.tensor.reshape(-1, 3)

        lap_sqrt = lap_sqrt_var.tensor
        mass_sqrt = mass_sqrt_var.tensor

        diff = verts - verts_target

        r_data = mass_sqrt @ diff
        r_smooth = (t**0.5) * (lap_sqrt @ verts)

        r = torch.cat([r_data, r_smooth], dim=0)
        return 0.5 * r.reshape(1, -1)

    mass_sqrt = sp.diag(torch.sqrt(sp.get_diag(mass).to_dense()))
    eigvals, eigvecs = torch.linalg.eigh(lap.to_dense())
    eigvals = torch.clamp(eigvals, min=0.0)
    lap_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    # lap_sqrt = torch.linalg.cholesky(lap.to_dense())
    lap_sqrt = lap_sqrt.to_dense().to(dtype=verts.dtype)
    mass_sqrt = mass_sqrt.to_dense().to(dtype=verts.dtype)

    n_verts, dim = verts.shape
    verts_var = th.Vector(dof=n_verts * dim, dtype=verts.dtype, name="verts_var")
    verts_target = th.Variable(tensor=verts.reshape(1, -1), name="verts_target")
    lap_sqrt_var = th.Variable(tensor=lap_sqrt.unsqueeze(0), name="lap_sqrt_var")
    mass_sqrt_var = th.Variable(tensor=mass_sqrt.unsqueeze(0), name="mass_sqrt_var")
    cost_weight = th.ScaleCostWeight(
        torch.tensor(1.0, dtype=verts.dtype, device=verts.device)
    )
    cost_fn = th.AutoDiffCostFunction(
        optim_vars=[verts_var],
        err_fn=fairing_residual,
        dim=2 * n_verts * dim,
        cost_weight=cost_weight,
        aux_vars=[verts_target, lap_sqrt_var, mass_sqrt_var],
        name="fairing_res",
    )

    objective = th.Objective(dtype=verts.dtype).to(device=verts.device)
    objective.add(cost_fn)
    optimizer = th.LevenbergMarquardt(objective, max_iterations=20, step_size=2)
    th_layer = th.TheseusLayer(optimizer)
    return th_layer


def theseus_forward(
    verts: torch.Tensor,
    lap: torch.Tensor,
    mass: torch.Tensor,
    t: float,
    th_layer: th.TheseusLayer,
) -> torch.Tensor:
    mass_sqrt = sp.diag(torch.sqrt(sp.get_diag(mass).to_dense()))

    eigvals, eigvecs = torch.linalg.eigh(lap.to_dense())
    eigvals = torch.clamp(eigvals, min=0.0)
    lap_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    lap_sqrt = lap_sqrt.to_dense()
    mass_sqrt = mass_sqrt.to_dense()
    inputs = {
        "verts_var": verts.reshape(1, -1),
        "verts_target": verts.reshape(1, -1),
        "lap_sqrt_var": lap_sqrt.unsqueeze(0),
        "mass_sqrt_var": mass_sqrt.unsqueeze(0),
    }
    solution, info = th_layer.forward(
        inputs,
        optimizer_kwargs={"track_best_solution": True, "verbose": True},
    )
    faired = solution["verts_var"].reshape(*verts.shape)
    return faired


FN_MAP = {
    "iskra": [iskra_setup, iskra_forward],
    "theseus": [theseus_setup, theseus_forward],
    "alec": [alec_setup, alec_forward],
}


def optimize(
    verts: torch.Tensor,
    faces: torch.Tensor,
    t: float,
    alpha: float,
    lr: float,
    method: str,
):
    verts_param = torch.nn.Parameter(verts.clone())
    optim = torch.optim.SGD([verts_param], lr=lr)
    lap, mass = dec.laplacian(verts, faces)
    d_01 = dec.d_01(faces, dtype=verts.dtype)
    h1_solver = _linear_solver_fn(mass + alpha * lap)

    setup_fn, forward_fn = FN_MAP[method]
    with profile_block(method):
        with profile_block("setup"):
            data = setup_fn(verts, lap, mass, t)
        for _ in range(100):
            optim.zero_grad()
            with profile_block("forward"):
                with profile_block("laplacian"):
                    cots = dec.hodge_1(verts_param, faces)
                    lap = sp.matmul(d_01.mT, sp.matmul(cots, d_01))
                    mass = dec.hodge_0(verts_param, faces)
                faired = forward_fn(verts_param, lap, mass, t, data)
                diff = faired - verts
                mass = mass.to_sparse_csr()
                loss = sp.matmul(diff.mT, sp.matmul(mass, diff))
                loss = loss.diagonal(dim1=-1, dim2=-2).sum()
                loss = loss
            with profile_block("backward"):
                loss.backward()
            if verts_param.grad is None:
                raise RuntimeError("verts_param.grad is None!")
            with profile_block("h1"):
                with torch.no_grad():
                    verts_param.grad = h1_solver(sp.matmul(mass, verts_param.grad))
            optim.step()
    return verts_param.data


def main(
    mesh_path: Path,
    results_dir: Path,
    method: str,
    t: float,
    device_name: str,
    dtype_name: str,
):
    torch.set_num_threads(16)
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    mesh, _ = Mesh.from_path(mesh_path, device=device, dtype=dtype)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    with Path(results_dir, "profile.json").open("w") as f:
        f.write("{}")

    if method == "theseus" and verts.shape[0] > 1_000:
        # theseus runs out of memory on large meshes on my machine,
        # larges successful was around 600 vertices.
        return

    inflated = optimize(verts, faces, t, 0.01, 2_000, method)

    with Path(results_dir, "profile.json").open("w") as f:
        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
    global_profiler.dump(path=Path(results_dir, "profile"))

    lap, mass = dec.laplacian(verts, faces)
    faired = linear_solve((mass + t * lap).coalesce(), mass @ verts)[1]
    import igl

    Path("results").mkdir(exist_ok=True)
    igl.writeOBJ(
        Path("results") / f"inflated_{mesh_path.stem}_{method}_{device}.obj",
        inflated.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
    )
    return
    try:
        ps.init()
        ps.register_surface_mesh(
            "Init", verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
        )
        ps.register_surface_mesh(
            "Deflated",
            faired.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )

        ps.register_surface_mesh(
            "IskraInflated",
            inflated.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )
        # ps.register_surface_mesh(
        #     "TheseusInflated",
        #     inflated_theseus.detach().cpu().numpy(),
        #     faces.detach().cpu().numpy(),
        # )
        ps.register_surface_mesh(
            "AlecInflated",
            inflated_alec.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    """
    python -m iskra.apps.comparisons.inflate ~/cgbcdata/open_cube.obj --t 0.04
    python -m iskra.apps.comparisons.inflate ~/cgbcdata/sphere.obj --t 0.04
    python -m iskra.apps.comparisons.inflate ~/cgbcdata/venus.obj --t 0.04
    """
    default_results_dir = Path().home() / "experiments" / "iskra" / "mcf"
    default_results_dir.mkdir(exist_ok=True, parents=True)

    parser = ArgumentParser(description="Demonstrates mesh inflation using inverse GP.")
    parser.add_argument("mesh_path", type=Path, help="The path of the mesh to load.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=default_results_dir,
        help="Where to save outputs.",
    )
    parser.add_argument("--t", type=float, default=0.001, help="Heat time parameter.")
    parser.add_argument(
        "--method",
        type=str,
        default="iskra",
        help="Which method to use in (iskra, alec, theseus).",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Mesh data type.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    args = parser.parse_args()
    main(args.mesh_path, args.results_dir, args.method, args.t, args.device, args.dtype)

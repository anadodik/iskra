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


def read_numbers(
    path: Path, device: str | torch.device, dtype: torch.dtype
) -> torch.Tensor:
    with path.open("r") as f:
        idcs = torch.tensor(
            [float(i) for i in f.readline().split(", ")], device=device, dtype=dtype
        )
    return idcs


def main(input_path: Path, result_path: Path):
    device = "cpu"
    dtype = torch.float32

    mesh, _ = Mesh.from_path(result_path, device=device, dtype=dtype)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices
    start_idcs = read_numbers(input_path / "start.txt", device, torch.int64)
    centerline_idcs = read_numbers(input_path / "centerline.txt", device, torch.int64)
    target_dist = read_numbers(input_path / "target_dist.txt", device, dtype)

    try:
        ps.init()
        ps.register_surface_mesh(
            "init", verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
        )
        ps_mesh = ps.register_surface_mesh(
            "terrain",
            verts.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )
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
    args = parser.parse_args()
    main(input_dir, args.results_dir)

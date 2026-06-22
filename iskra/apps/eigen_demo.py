# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
import torch
from rich import print
from rich.table import Table

import iskra.sparse as sp
from iskra.dec import laplacian
from iskra.geometry import normal_coordinate_system, triangle_areas, triangle_normals
from iskra.logging import getLogger
from iskra.mesh import Mesh
from iskra.sparse_linalg import eigsh
from iskra.topology import boundary, face_index, get_subfaces, ordered_boundary_edges

LOGGER = getLogger(__file__)


def complicated_loss(evals: torch.Tensor, evecs: torch.Tensor, k: int) -> torch.Tensor:
    w_evals = torch.linspace(1.0, 2.0, steps=k, dtype=evals.dtype, device=evals.device)
    # linear + quadratic, per-index wehts
    loss_evals = (w_evals * evals + 0.1 * (w_evals**2) * (evals**2)).sum()

    # Distinct weights for each entry of evecs (n x k)
    i = torch.arange(evecs.size(0), dtype=evecs.dtype, device=evecs.device).unsqueeze(1)
    j = torch.arange(evecs.size(1), dtype=evecs.dtype, device=evecs.device).unsqueeze(0)
    # Unique weights per (i,j)
    w = (i + 1) + 0.37 * (j + 1) + 0.01 * (i + 1) * (j + 1)
    # quadratic in evecs; sign-invariant
    loss_evecs = ((w * torch.abs(evecs)) ** 2).sum()
    # TODO: Fix and renable evals loss.
    return loss_evecs  # + loss_evals


def test_adjoint(
    lap: torch.Tensor, mass: torch.Tensor, k: int, sigma: float | None, adjoint: str
):
    LOGGER.info(f"Testing adjoint for method `{method}`.")
    lap = lap.clone()
    mass = mass.clone()
    if adjoint == "dense":
        lap = lap.to_dense()
        mass = mass.to_dense()

    lap = lap.requires_grad_(True)
    lap.grad = torch.zeros_like(lap)

    start = time.perf_counter()
    # TODO: CUDA sync on GPU for perf
    if adjoint == "dense":
        evals, evecs = torch.linalg.eigh(torch.linalg.solve(mass, lap))
        evals, evecs = evals.real, evecs.real
        if sigma is not None:
            shifted_evals = 1 / (evals - sigma) if sigma is not None else evals
            sort_idx = torch.argsort(torch.abs(shifted_evals))
            evals = evals[sort_idx][-k:]
            evecs = evecs[:, sort_idx][:, -k:]
        else:
            sort_idx = torch.argsort(torch.abs(evals), descending=True)
            evals = evals[sort_idx][:k]
            evecs = evecs[:, sort_idx][:, :k]
    else:
        evals, evecs = eigsh(lap, M=mass, k=k, sigma=sigma, bwd_method=adjoint)
    forward_time = time.perf_counter() - start

    # loss = ((evecs.abs() - 0.5) ** 2).sum()
    loss = complicated_loss(evals, evecs, k)

    LOGGER.info(f"Running backward pass for method `{method}`.")
    start = time.perf_counter()
    loss.backward()
    backward_time = time.perf_counter() - start
    grad_a = lap.grad
    stats = {"forward_time": forward_time, "backward_time": backward_time}
    return evals, evecs, grad_a, stats


def construct_graph_laplacian() -> tuple[torch.Tensor, torch.Tensor]:
    graph: nx.Graph = nx.Graph()  # pyright: ignore[reportMissingTypeArgument]
    n = 5
    graph.add_edge(0, 1, weight=4.0)
    graph.add_edge(1, 3, weight=2)
    graph.add_edge(0, 2, weight=3.0)
    graph.add_edge(2, 3, weight=4)
    graph.add_edge(2, 4, weight=4)
    graph.add_nodes_from(range(n))

    lap = nx.laplacian_matrix(graph)
    lap = sp.from_scipy(lap).to_sparse_coo().to(dtype=dtype)
    lap = lap + 1e-2 * sp.eye(n)

    n = graph.number_of_nodes()
    mass = sp.eye(n, dtype=dtype)
    return lap, mass


def construct_diag_laplacian(
    n: int = 4, repeated: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    diag = torch.arange(n, device=device, dtype=dtype)
    if repeated:
        diag[1] = diag[2]
    lap = sp.diag(diag)
    mass = sp.eye(n, dtype=dtype)
    return lap, mass


if __name__ == "__main__":
    dtype = torch.double
    device = "cpu"
    torch.set_printoptions(sci_mode=False, linewidth=80 * 2)
    lap, mass = construct_graph_laplacian()

    n = lap.shape[0]
    k = 3
    sigma = -1e-12

    methods = [
        "unroll",
        "dense",
        "dodik-fixedpoint",
        # "dodik-invert",  # TODO: try with new PyTorch upstream
        "truncate",
        "individual",
    ]

    # TODO: Fix fixed-point solver for eigenvalue loss.
    # TODO: Fix M gradients for all solvers.

    table = Table(title="Eigenvector Comparison")
    table.add_column("Method", justify="right", style="cyan", no_wrap=True)
    table.add_column("Sigma", style="magenta")
    table.add_column("U", style="magenta")
    table.add_column("grad A", style="green")
    table.add_column("Sparse?", style="red")
    table.add_column("Forward (s)", style="red")
    table.add_column("Backward (s)", style="red")
    for method in methods:
        test_adjoint(lap, mass, k, sigma, method)  # warmup
        evals, evecs, grad_a, stats = test_adjoint(lap, mass, k, sigma, method)
        table.add_row(
            method,
            ",".join(f"{e:.4f}" for e in evals.cpu().tolist()),
            str(evecs),
            str(grad_a.to_dense()),
            str(grad_a.is_sparse),
            f"{stats['forward_time']:.4f}",
            f"{stats['backward_time']:.4f}",
        )

    print(table)

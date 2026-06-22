# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import itertools
import time
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
import torch
from rich import print
from rich.table import Table

torch.set_num_threads(32)
torch.set_num_interop_threads(32)

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from palettable.cartocolors.diverging import TealRose_7 as diverging_cmap
from palettable.cartocolors.qualitative import Prism_9 as qualitative_cmap

import iskra.sparse as sp
from iskra import dec
from iskra.dec import laplacian
from iskra.geometry import (
    cotan_weights,
    normal_coordinate_system,
    triangle_areas,
    triangle_normals,
)
from iskra.logging import getLogger
from iskra.mesh import Mesh
from iskra.sparse_linalg import eigsh
from iskra.topology import boundary, face_index, get_subfaces, ordered_boundary_edges

_MATPLOTLIB_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Linux Libertine"],
    "font.size": 55,
    "figure.facecolor": "#FCFCFC",
    "axes.facecolor": "#FCFCFC",
    "axes.prop_cycle": cycler(color=qualitative_cmap.mpl_colors),
    "lines.antialiased": True,
    "text.latex.preamble": r"""
    \usepackage{libertine}
    \usepackage[libertine]{newtxmath}
    """,
    "mathtext.rm": "libertine",
    "mathtext.it": "libertine:italic",
    "mathtext.bf": "libertine:bold",
}

mpl.rcParams.update(_MATPLOTLIB_STYLE)

LOGGER = getLogger(__file__)


def complicated_loss(evals: torch.Tensor, evecs: torch.Tensor) -> torch.Tensor:
    k = evals.shape[0]
    w_evals = torch.linspace(1.0, 2.0, steps=k, dtype=evals.dtype, device=evals.device)
    # linear + quadratic, per-index wehts
    loss_evals = (w_evals * evals + 0.1 * (w_evals**2) * (evals**2)).mean()

    # Distinct weights for each entry of evecs (n x k)
    i = torch.arange(evecs.size(0), dtype=evecs.dtype, device=evecs.device).unsqueeze(1)
    j = torch.arange(evecs.size(1), dtype=evecs.dtype, device=evecs.device).unsqueeze(0)
    # Unique weights per (i,j)
    w = (i + 1) + 0.37 * (j + 1) + 0.01 * (i + 1) * (j + 1)
    # quadratic in evecs; sign-invariant
    loss_evecs = ((w * torch.abs(evecs)) ** 2).mean()
    # TODO: Fix and renable evals loss.
    return loss_evecs  # + loss_evals


def test_adjoint(
    lap: torch.Tensor, mass: torch.Tensor, k: int, sigma: float | None, adjoint: str
):
    LOGGER.info(f"Testing adjoint for method `{adjoint}`.")
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

    # print(evals.shape, evecs.shape)
    loss = complicated_loss(evals[:-2], evecs[:, :-2])
    # loss = ((evecs.abs() - 0.5) ** 2).mean()

    LOGGER.info(f"Running backward pass for method `{adjoint}`.")
    start = time.perf_counter()
    loss.backward()
    backward_time = time.perf_counter() - start
    grad_a = lap.grad
    stats = {"forward_time": forward_time, "backward_time": backward_time}
    return evals, evecs, grad_a, stats


def vertex_area_matrix(
    n_vertices: int, faces: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    bdr_edges = boundary(faces)
    bdr_edges_bwd = bdr_edges[:, (1, 0)]
    n_bdr_edges = bdr_edges.shape[0]
    idcs_i = torch.cat([bdr_edges, bdr_edges_bwd + n_vertices], -2).flatten(-2, -1)
    idcs_j = torch.cat([bdr_edges_bwd + n_vertices, bdr_edges], -2).flatten(-2, -1)
    values = torch.tensor([0.25, -0.25], device=faces.device, dtype=dtype)
    values = values[None, :].expand(2 * n_bdr_edges, -1).flatten(-2, -1)
    return sp.coo_tensor(
        torch.stack([idcs_i, idcs_j], -2), values, size=[2 * n_vertices, 2 * n_vertices]
    )


def build_conformal_laplacian(verts: torch.Tensor, faces: torch.Tensor):
    dtype = verts.dtype
    device = verts.device
    n_verts = verts.shape[0]
    va_mat = vertex_area_matrix(n_verts, faces, dtype=dtype)
    d_01 = dec.d_01(faces, dtype=dtype)
    weights = cotan_weights(verts, faces, 1e-8)
    lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(weights), d_01)).coalesce()
    lap = (sp.repdiag(lap, 2) - 2 * va_mat).coalesce()

    bdr_idx = boundary(faces).flatten().unique()
    bdr_ii = torch.stack([bdr_idx, bdr_idx], 0)
    bdr_val = torch.ones(bdr_ii.shape[1], dtype=dtype, device=device)
    boundary_mat_block = sp.coo_tensor(bdr_ii, bdr_val, size=[n_verts, n_verts])
    boundary_mat = sp.repdiag(boundary_mat_block, 2)
    return lap, boundary_mat


def construct_graph_laplacian(dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
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
    mass = sp.eye(n, dtype=dtype, device=device)
    return lap, mass


def main():
    LOGGER.setLevel("INFO")
    dtype = torch.double
    device = "cpu"
    torch.set_printoptions(sci_mode=False, linewidth=80 * 2)

    mesh_dir = Path().home() / "Dropbox" / "Data" / "iskra-data" / "mcf"
    results_dir = Path().home() / "experiments" / "iskra" / "eigenstudy"
    results_dir.mkdir(parents=True, exist_ok=True)

    mesh_paths = list(mesh_dir.glob("*.obj"))
    # mesh_paths = [mesh_dir / "venus.obj"]

    sigma = -1e-6  # None

    methods = [
        "unroll",
        # "dodik-fixedpoint",
        # "truncate",
        # "individual",
    ]

    # TODO: Fix fixed-point solver for eigenvalue loss.
    # TODO: Fix M gradients for all solvers.[]

    table = Table(title="Eigenvector Comparison")
    table.add_column("Mesh Name", justify="right", style="cyan", no_wrap=True)
    table.add_column("|V|", justify="right", style="cyan", no_wrap=True)
    table.add_column("k", justify="right", style="cyan", no_wrap=True)
    table.add_column("Method", justify="right", style="green", no_wrap=True)
    table.add_column("Grad. accuracy.", style="red")
    table.add_column("Sparse?", style="red")
    table.add_column("Forward (s)", style="red")
    table.add_column("Backward (s)", style="red")
    ks = [3]  # 3, 50, 150, 250]
    devices = ["cpu"]
    combinations = list(itertools.product(mesh_paths, ks, devices))

    results = {}
    for mesh_path, k, device in combinations:
        print(f"Testing {mesh_path.stem}, {k}")
        columns = ["mesh", "nv", "k", "method", "accuracy", "fwd_s", "bwd_s"]
        mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
        out_path = results_dir / f"results_{mesh_path.stem}_{k}_{device}_{dtype}.csv"
        if out_path.exists():
            print(f"Path exists {out_path}")
            continue
        if mesh.n_vertices > 170_000 or k >= mesh.n_vertices:
            print("Mesh or k too large.")
            continue
        mesh.geom.normalize()
        faces, verts = mesh.topo.faces, mesh.geom.vertices
        # lap, mass = build_conformal_laplacian(verts, faces)
        # lap, mass = construct_graph_laplacian(dtype, device)
        lap, mass = dec.laplacian(verts, faces)
        # lap = sp.eye(lap.shape[0], dtype=dtype, device=device)
        mass = sp.eye(lap.shape[0], dtype=dtype, device=device)
        _, _, gt_grad_a, _ = test_adjoint(lap, mass, k, sigma, "individual")
        # print(evals, evecs)
        # print(gt_grad_a)

        df = pd.DataFrame(columns=columns)
        print(f"Mesh: {mesh_path}")
        for method in methods:
            # if method == "unroll" and mesh.n_vertices > 500_000:
            #     continue
            #     df.loc[len(df)] = [
            #         mesh_path.stem,
            #         mesh.n_vertices,
            #         k,
            #         method,
            #         float("nan"),
            #         float("nan"),
            #         float("nan"),
            #     ]
            # TODO: MEASURE MEMORY
            test_adjoint(lap, mass, k, sigma, method)  # warmup
            eye = sp.eye(mass.shape[0], dtype=dtype, device=device)
            evals, evecs, grad_a, stats = test_adjoint(lap, mass, k, sigma, method)
            grad_diff = (grad_a - gt_grad_a).coalesce().values()
            # / (gt_grad_a.coalesce().values() + 1e-8)
            grad_a_accuracy = (grad_diff.abs()).mean().item()
            print(grad_diff)
            print(
                mesh_path.stem,
                mesh.n_vertices,
                k,
                method,
                str(grad_a_accuracy),
                str(grad_a.is_sparse),
                f"{stats['forward_time']:.4f}",
                f"{stats['backward_time']:.4f}",
            )
            table.add_row(
                mesh_path.stem,
                str(mesh.n_vertices),
                str(k),
                method,
                str(grad_a_accuracy),
                str(grad_a.is_sparse),
                f"{stats['forward_time']:.4f}",
                f"{stats['backward_time']:.4f}",
            )
            df.loc[len(df)] = [
                mesh_path.stem,
                mesh.n_vertices,
                k,
                method,
                grad_a_accuracy,
                stats["forward_time"],
                stats["backward_time"],
            ]
        print("saving to", out_path)
        df.to_csv(out_path)
    print(table)


def plot_df(df, n_verts_col, y, out_path):
    fig, ax = plt.subplots(figsize=(14, 14), layout="constrained", dpi=300)
    # fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)
    label_map = {
        "fwd_s": "Forward Time (s)",
        "bwd_s": "Backward Time (s)",
        "accuracy": "Mean Absolute Error",
    }
    ax.set_ylabel(label_map[y])
    ax.set_xlabel(r"$$|\mathbb V|$$")

    method_map = {
        "dodik-fixedpoint": "Fixed-point",
        "truncate": "Truncate",
        "unroll": "Unroll",
        "individual": "Individual",
    }
    # plot_df.loc[:, plot_df.columns != n_verts_col] *= 1e-3
    colors = (
        qualitative_cmap.mpl_colors[3],
        qualitative_cmap.mpl_colors[0],
        qualitative_cmap.mpl_colors[5],
        qualitative_cmap.mpl_colors[7],
    )
    styles = ["o--", "X:", "D-.", "P-"]
    gs = list(df.groupby("method"))
    gs.sort(key=lambda x: x[0] if x[0] != "individual" else "zzzzzzzzzzzzz")
    for i, (method, g) in enumerate(gs):
        ax.semilogx(
            g["nv"],
            g[y],
            styles[i],
            label=method_map[method],
            lw=4.5,
            ms=14,
            color=colors[i],
        )
    # plot_df.plot(
    #     x=n_verts_col,
    #     y=y,
    #     ax=ax,
    #     # style=["o-", "X-", "D-"],
    #     # color=colors,
    #     lw=3.5,
    #     ms=14,
    # )

    ax.grid(True, which="major", axis="y", zorder=0, linewidth=1)
    ax.grid(True, which="minor", axis="y", zorder=0, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc="upper left")
    plt.tight_layout(pad=0.05)
    fig.savefig(out_path)
    plt.show()


def plot(k):
    results_dir = Path().home() / "experiments" / "iskra" / "eigenstudy"
    out_dir = Path().home() / "Dropbox" / "Results" / "iskra" / "eigenstudy"
    out_dir.mkdir(exist_ok=True)
    dtype = torch.double
    device = "cpu"

    in_path = results_dir / f"results_{device}_{dtype}.csv"
    all_paths = [
        (int(r.stem[8:-18].split("_")[-1]), r) for r in results_dir.glob("*.csv")
    ]
    k_paths = [path for path_k, path in all_paths if path_k == k]
    dfs = []
    for in_path in k_paths:
        df = pd.read_csv(
            in_path,
            dtype={
                "mesh": str,
                "nv": int,
                "k": int,
                "method": str,
                "accuracy": float,
                "fwd_s": float,
                "bwd_s": float,
            },
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # print(df)
    n_verts_col = "nv"
    df = df.sort_values(n_verts_col)
    ks = df.loc[:, "k"].unique()
    for k in ks:
        df_k = df[df["k"] == k]
        for measurement in ["accuracy", "fwd_s", "bwd_s"]:
            df_k = df_k[df_k["mesh"] != "sphere"]
            df_k = df_k[df_k["mesh"] != "torus2"]
            print(df)
            df_m = df_k[["nv", "method", measurement]]
            # print(df_m)
            out_path = out_dir / f"{measurement}_{k}_{device}_{dtype}.png"
            if measurement == "accuracy":
                df_m = df_m[df_m["nv"] >= 50]
                df_m[df_m["accuracy"] > 2000] = float("nan")
                df_m = df_m[df_m["method"] != "individual"]
            plot_df(df_m, n_verts_col, measurement, out_path)


if __name__ == "__main__":
    # main()
    plot(3)
    plot(50)

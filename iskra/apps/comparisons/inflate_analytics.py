import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from palettable.cartocolors.diverging import TealRose_7 as diverging_cmap
from palettable.cartocolors.qualitative import Prism_9 as qualitative_cmap

from iskra.mesh import Mesh

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


def get_value(data, *path):
    if not path:
        return None
    name = path[0]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("name") == name:
                if len(path) == 1:
                    return item
                elif len(path) == 2:
                    return item.get(path[1])
                else:
                    return get_value(item.get("children", []), *path[1:])
    return None


def main():
    mesh_dir = Path().home() / "Dropbox" / "Data" / "iskra-data" / "mcf"
    results_root_dir = Path().home() / "experiments" / "iskra" / "mcf"

    mesh_paths = list(mesh_dir.glob("*.obj"))
    devices = ["cpu", "cuda"]
    dtypes = ["float64"]
    methods = ["iskra", "alec", "theseus"]

    combinations = list(itertools.product(mesh_paths, devices, dtypes, methods))

    n_verts_col = r"$$|\mathbb V|$$"
    columns = [
        "Mesh Name",
        n_verts_col,
        "SparseSolve",
        "Theseus",
        "Ours",
    ]
    columns_gpu = [
        "Mesh Name",
        n_verts_col,
        "SparseSolve",
        "Theseus",
        "Ours",
    ]

    plot_types = ["Total", "Setup", "Solver", "Forward", "Backward"]
    for plot_type in plot_types:
        df = pd.DataFrame(columns=columns)
        df_gpu = pd.DataFrame(columns=columns_gpu)
        for mesh_path, device, dtype, method in combinations:
            mesh_name = Path(mesh_path).stem
            out_dir = results_root_dir / mesh_name / f"{method}_{device}_{dtype}"
            profile_path = out_dir / "profile.json"

            if not profile_path.exists():
                # if method == "iskra":
                #     print(profile_path)
                continue
            if df.index[df["Mesh Name"] == mesh_name].empty:
                n_verts = count_vertices(mesh_path)
                df.loc[len(df)] = [mesh_name, n_verts] + [float("nan")] * (
                    len(columns) - 2
                )
                df_gpu.loc[len(df_gpu)] = [mesh_name, n_verts] + [float("nan")] * (
                    len(columns) - 2
                )

            with profile_path.open("r") as f:
                profile_data = json.load(f)

            if profile_data == {}:
                continue

            if method == "alec":
                getter_map = {
                    "Total": ("alec", "mean_time_ns"),
                    "Setup": ("alec", "setup", "mean_time_ns"),
                    "Forward": ("alec", "forward", "mean_time_ns"),
                    "Solver": ("alec", "forward", "solver", "mean_time_ns"),
                    "Backward": ("alec", "backward", "mean_time_ns"),
                }
                time = get_value(profile_data, *getter_map[plot_type])
                if device == "cpu":
                    df.loc[df["Mesh Name"] == mesh_name, "SparseSolve"] = time
                else:
                    df_gpu.loc[df["Mesh Name"] == mesh_name, "SparseSolve"] = time
            elif method == "iskra":
                getter_map = {
                    "Total": ("iskra", "mean_time_ns"),
                    "Setup": ("iskra", "setup", "mean_time_ns"),
                    "Forward": ("iskra", "forward", "mean_time_ns"),
                    "Solver": ("iskra", "forward", "solver", "mean_time_ns"),
                    "Backward": ("iskra", "backward", "mean_time_ns"),
                }
                time = get_value(profile_data, *getter_map[plot_type])
                if device == "cpu":
                    df.loc[df["Mesh Name"] == mesh_name, r"Ours"] = time
                else:
                    df_gpu.loc[df["Mesh Name"] == mesh_name, r"Ours"] = time
            elif method == "theseus":
                getter_map = {
                    "Total": ("theseus", "mean_time_ns"),
                    "Setup": ("theseus", "setup", "mean_time_ns"),
                    "Forward": ("theseus", "forward", "mean_time_ns"),
                    "Solver": ("theseus", "forward", "solver", "mean_time_ns"),
                    "Backward": ("theseus", "backward", "mean_time_ns"),
                }
                time = get_value(profile_data, *getter_map[plot_type])
                if device == "cpu":
                    df.loc[df["Mesh Name"] == mesh_name, r"Theseus"] = time
                else:
                    df_gpu.loc[df["Mesh Name"] == mesh_name, r"Theseus"] = time

        def plot_df(df, device, out_path):
            df = df.sort_values(n_verts_col)
            fig, ax = plt.subplots(figsize=(14, 14), layout="constrained", dpi=300)
            # fig.patch.set_alpha(0.0)
            # ax.patch.set_alpha(0.0)
            ax.set_ylabel(f"Runtime {device} (s)")
            plot_df = df.loc[:, df.columns != "Mesh Name"]
            plot_df.loc[:, plot_df.columns != n_verts_col] *= 1e-3
            colors = (
                qualitative_cmap.mpl_colors[3],
                qualitative_cmap.mpl_colors[0],
                qualitative_cmap.mpl_colors[7],
            )
            plot_df.plot(
                x=n_verts_col,
                ax=ax,
                style=["o-", "X-", "D-"],
                color=colors,
                lw=3.5,
                ms=14,
            )

            ax.grid(True, which="major", axis="y", zorder=0, linewidth=1)
            ax.grid(True, which="minor", axis="y", zorder=0, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.legend(loc="upper left")
            plt.tight_layout(pad=0.05)
            print(df)
            fig.savefig(out_path)
            # plt.show()

        print(f"\n\nSaving plot type {plot_type}.")
        plot_dir = Path().home() / "Dropbox" / "Results" / "iskra" / "inflate_2"
        plot_df(
            df_gpu,
            "GPU",
            plot_dir / f"mcf_{plot_type.lower()}_gpu.png",
        )
        plot_df(df, "CPU", plot_dir / f"mcf_{plot_type.lower()}_cpu.png")

    # print(f"{mesh_path.name} ({num_vertices} vertices) - {method}_{device}_{dtype}")
    # print(f"  Profile: {profile_data}\n")


def count_vertices(mesh_path):
    from iskra.io.io import LOGGER

    LOGGER.setLevel("WARN")
    mesh, _ = Mesh.from_path(mesh_path)
    return mesh.n_vertices


if __name__ == "__main__":
    main()

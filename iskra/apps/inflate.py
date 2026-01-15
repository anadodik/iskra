# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
from argparse import ArgumentParser

import torch
from gemvis.geometry import dual_quat

import iskra.sparse as sp
from iskra.dec import laplacian
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block
from iskra.sparse_linalg import default_solver, linear_solve

if __name__ == "__main__":
    """
    python -m iskra.apps.inflate ~/cgbcdata/bust_of_sappho/sapphos_head.obj --t 0.001
    python -m iskra.apps.inflate ~/cgbcdata/bust_of_sappho/sapphos_head.obj --t 0.005
    python -m iskra.apps.inflate ~/cgbcdata/bust_of_sappho/sapphos_head.obj --t 0.01
    python -m iskra.apps.inflate ~/cgbcdata/bust_of_sappho/sapphos_head.obj --t 0.02
    python -m iskra.apps.inflate ~/cgbcdata/bust_of_sappho/sapphos_head.obj --t 0.04
    """
    parser = ArgumentParser(description="Demonstrates mesh inflation using inverse GP.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    parser.add_argument("--t", type=float, default=0.001, help="Heat time parameter.")
    args = parser.parse_args()

    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    t = args.t
    alpha = 0.01

    verts_var = torch.nn.Parameter(verts.clone())
    optim = torch.optim.SGD([verts_var], lr=2_000)
    lap, mass = laplacian(verts, faces)
    # mcf_solver = _linear_solver_fn(mass + t * lap)
    h1_solver = default_solver(mass + alpha * lap)
    for i in range(10):
        optim.zero_grad()
        with profile_block("forward"):
            lap, mass = laplacian(verts_var, faces)
            faired = linear_solve(mass + t * lap, sp.matmul(mass, verts_var))[1]
            diff = faired - verts
            loss = (sp.matmul(diff.mT, sp.matmul(mass, diff))).diagonal().sum()
        with profile_block("backward"):
            loss.backward()
        if verts_var.grad is None:
            raise RuntimeError("verts_var.grad is None!")
        with profile_block("h1"):
            verts_var.grad = h1_solver(mass @ verts_var.grad)
        optim.step()

    lap, mass = laplacian(verts, faces)
    faired = linear_solve(mass + t * lap, mass @ verts_var)[1]

    try:
        from pathlib import Path

        import igl
        from gemvis import Plot

        out_path = Path.home() / r"MIT Dropbox/Ana Dodik/Results/iskra"
        results_path = out_path / "inflate_2" / Path(args.mesh_path).stem / f"t={t:.4f}"

        global_profiler.dump()
        with Path(out_path, "profile.json").open("w") as f:
            f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
        global_profiler.dump(path=Path(out_path, "profile.txt"))

        plt = Plot(ground=False)
        plt.camera.rigid = (
            dual_quat.from_translation([0.15, 0.0, 8.5]) @ plt.camera_views[1]
        )
        init_plot = plt.plot_mesh(verts, faces)
        init_plot.set_colors(init_plot.qualitative_cmap[-3])
        plt.render()
        plt.save_screenshot("init.png", results_path)
        igl.writeOBJ(
            results_path / "init.obj",
            verts.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )

        plt = Plot(ground=False)
        plt.camera.rigid = (
            dual_quat.from_translation([0.15, 0.0, 8.5]) @ plt.camera_views[1]
        )
        deflated_plot = plt.plot_mesh(faired, faces)
        deflated_plot.set_colors(deflated_plot.qualitative_cmap[-5])
        plt.render()
        plt.save_screenshot("deflated.png", results_path)
        igl.writeOBJ(
            results_path / "deflated.obj",
            faired.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )

        plt = Plot(ground=False)
        plt.camera.rigid = (
            dual_quat.from_translation([0.15, 0.0, 8.5]) @ plt.camera_views[1]
        )
        inflated_plot = plt.plot_mesh(verts_var, faces)
        inflated_plot.set_colors(inflated_plot.qualitative_cmap[-4])
        plt.render()
        plt.save_screenshot("inflated.png", results_path)
        igl.writeOBJ(
            results_path / "inflated.obj",
            verts_var.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )

    except ImportError:
        print(
            "Could not import gemvis to visualize the results."
            "Install it by running: pip install gemvis"
        )

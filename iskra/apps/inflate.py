# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch
from gemvis.geometry import dual_quat

from iskra.dec import laplacian
from iskra.mesh import Mesh
from iskra.sparse import CholeskySolver, min_quadratic_energy

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

    lap, mass = laplacian(verts, faces)
    verts_var = torch.nn.Parameter(verts.clone())
    optim = torch.optim.SGD([verts_var], lr=2_000)
    mcf_solver = CholeskySolver(mass + t * lap)
    h1_solver = CholeskySolver(mass + alpha * lap)
    for i in range(500):
        optim.zero_grad()
        diff = mcf_solver(mass @ verts_var) - verts
        loss = (diff.mT @ mass @ diff).diagonal().sum()
        loss.backward()
        if verts_var.grad is None:
            raise RuntimeError("verts_var.grad is None!")
        verts_var.grad = h1_solver(mass @ verts_var.grad)
        optim.step()

    try:
        bdr = torch.tensor([], dtype=torch.long, device=faces.device)
        verts_smooth = min_quadratic_energy(
            mass + 0.001 * lap, mass @ verts, bdr, verts[bdr]
        )[1]

        from pathlib import Path

        import igl
        from gemvis import Plot

        results_dir = Path.home() / r"MIT Dropbox/Ana Dodik/Results/iskra"
        results_path = (
            results_dir / "inflate" / Path(args.mesh_path).stem / f"t={t:.4f}"
        )

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
        deflated_plot = plt.plot_mesh(verts_smooth, faces)
        deflated_plot.set_colors(deflated_plot.qualitative_cmap[-5])
        plt.render()
        plt.save_screenshot("deflated.png", results_path)
        igl.writeOBJ(
            results_path / "deflated.obj",
            verts_smooth.detach().cpu().numpy(),
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

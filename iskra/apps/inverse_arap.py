# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
from argparse import ArgumentParser
from pathlib import Path

import igl
import numpy as np
import torch

torch.set_num_threads(32)
torch.set_num_interop_threads(32)

from iskra.deformation import arap_precompute, arap_solve
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block
from iskra.topology import boundary


def main(
    mesh_path: Path,
    target_mesh_path: Path,
    handles_path: Path,
    device_name: str,
    dtype_name: str,
    method: str,
    lr: float,
    arap_steps: int,
    max_steps: int,
):
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    results_dir = Path.home() / "Dropbox" / "Results" / "iskra" / "arap_2"
    results_dir = results_dir / mesh_path.stem / f"{method}_{device_name}_{dtype_name}"

    # Load meshes
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    target_mesh, _ = Mesh.from_path(target_mesh_path, dtype=dtype, device=device)
    _, target_verts = target_mesh.topo.faces, target_mesh.geom.vertices

    # Load handles
    bdr_idx = boundary(faces)[:, 0]
    with Path(handles_path).open("r") as f:
        control_idx = torch.tensor(
            [int(i) for i in f.readline().split(", ")],
            device=device,
            dtype=torch.int64,
        )
    handle_idx = torch.cat([bdr_idx, control_idx])
    print("Handles loaded.")

    vert_vert, vert_vert_weights, lap, lap_factors = arap_precompute(
        verts, faces, handle_idx, 1e-5
    )

    handles = verts[handle_idx]
    handles = handles.requires_grad_(True)
    optimizer = torch.optim.SGD([handles], lr=lr)
    optimizer.zero_grad()
    print("Solving with iskra.")
    deformed, energy = arap_solve(
        verts, handle_idx, handles, vert_vert, vert_vert_weights, lap, lap_factors
    )
    loss = ((deformed - target_verts) ** 2).mean()
    loss.backward()

    print("Solving with libigl.")
    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    with profile_block("igl_arap_precomp"):
        igl.arap_precomputation(
            verts.cpu().numpy(),
            faces.cpu().numpy(),
            3,
            handle_idx.cpu().numpy(),
            arap_data_igl,
        )
    with profile_block(f"igl_arap_solve_{arap_data_igl.max_iter}_steps"):
        arap_deformed_igl = igl.arap_solve(
            handles.detach().cpu().numpy(), arap_data_igl, verts.detach().cpu().numpy()
        )
    print("Solved with libigl.")

    global_profiler.dump()
    global_profiler.summary_to_json()
    # with Path(results_dir, "profile.json").open("w") as f:
    #     f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
    # global_profiler.dump(path=Path(results_dir, "profile"))

    try:
        import polyscope as ps

        ps.set_allow_headless_backends(True)
        ps.init()

        ps.set_ground_plane_mode("shadow_only")
        ps.register_surface_mesh(
            "mesh", verts.cpu().numpy(), faces.cpu().numpy(), enabled=False
        )
        ps_mesh = ps.register_surface_mesh(
            "deformed", deformed.detach().cpu().numpy(), faces.cpu().numpy()
        )
        _ = ps.register_surface_mesh(
            "target", target_verts.detach().cpu().numpy(), faces.cpu().numpy()
        )
        ps_mesh_arap = ps.register_surface_mesh(
            "arap_deformed", arap_deformed_igl, faces.cpu().numpy(), enabled=False
        )
        ps_edges = ps.register_curve_network(
            "edges",
            deformed.detach().cpu().numpy(),
            vert_vert.cpu().numpy(),
            enabled=False,
            radius=0.01,
        )
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.cpu().numpy(), defined_on="faces"
        )
        ps_cloud = ps.register_point_cloud(
            "bc", handles.detach().cpu().numpy(), enabled=True
        )
        ps_mesh.add_scalar_quantity(
            "energy", energy.detach().cpu().numpy(), defined_on="vertices", enabled=True
        )
        assert handles.grad is not None
        ps_cloud.add_vector_quantity(
            "-grad bc", -handles.grad.cpu().numpy(), enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity(
            "cotan", vert_vert_weights.cpu().numpy(), defined_on="edges"
        )

        optimizing = False
        optim_step = 0
        out_path = Path(mesh_path.parent, "animation")
        out_path.mkdir(exist_ok=True, parents=True)

        igl.writeOBJ(
            str(out_path / f"step_handles_{optim_step}.obj"),
            handles.detach().cpu().numpy(),
            np.empty([0, 3]),
        )

        def callback():
            nonlocal optimizing, deformed, optim_step, out_path

            if ps.imgui.Button(
                "Start Optimization" if not optimizing else "Stop Optimizing"
            ):
                optimizing = not optimizing
            if optimizing:
                with profile_block("optim_step"):
                    optimizer.zero_grad()
                    deformed, energy = arap_solve(
                        verts,
                        handle_idx,
                        handles,
                        vert_vert,
                        vert_vert_weights,
                        lap,
                        lap_factors,
                        fwd_max_iter=arap_steps,
                        verbose=True,
                    )
                    print(f"ARAP energy: {energy.mean().detach().cpu().item()}")
                    loss = ((deformed - target_verts) ** 2).mean()
                    print(f"Loss = {loss.detach().cpu().item()}.")
                    loss.backward()
                    optimizer.step()

                optim_step += 1
                if optim_step % max_steps == 0:
                    global_profiler.dump()
                    with Path(out_path, f"profile_{optim_step}.json").open("w") as f:
                        f.write(json.dumps(global_profiler.summary_to_json(), indent=2))
                    global_profiler.dump(path=Path(out_path, f"profile_{optim_step}"))
                    optimizing = False

                with torch.no_grad():
                    print(f"Step {optim_step}.")
                    ps_cloud.update_point_positions(handles.detach().cpu().numpy())
                    arap_deformed_igl = igl.arap_solve(
                        handles.detach().cpu().numpy(),
                        arap_data_igl,
                        verts.detach().cpu().numpy(),
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_{optim_step}.obj"),
                        arap_deformed_igl,
                        faces.cpu().numpy(),
                    )
                    igl.writeOBJ(
                        str(out_path / f"step_handles_{optim_step}.obj"),
                        handles.detach().cpu().numpy(),
                        np.empty([0, 3]),
                    )
                    ps_mesh.update_vertex_positions(deformed.detach().cpu().numpy())
                    ps_mesh_arap.update_vertex_positions(arap_deformed_igl)
                    ps_mesh.add_scalar_quantity(
                        "energy",
                        energy.detach().cpu().numpy(),
                        defined_on="vertices",
                        enabled=True,
                    )

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )


if __name__ == "__main__":
    """
    python -m iskra.apps.inverse_arap data/hand/hand.obj data/hand/hand_deformed_sculpt.obj --handles data/hand/hand_handles.txt --lr 10 --arap_steps 150
    python -m iskra.apps.inverse_arap data/penguin/penguin.obj data/penguin/penguin_deformed.obj --handles data/penguin/penguin_handles.txt --lr 5 --arap_steps 100 --steps 150
    python -m iskra.apps.inverse_arap data/armadillo/armadillo.obj data/armadillo/armadillo_deformed.obj --handles data/armadillo/armadillo_handles.txt --lr 10 --arap_steps 200
    python -m iskra.apps.inverse_arap data/springer_rm/springer.obj data/springer_rm/springer_deformed.obj --handles data/springer_rm/springer_handles.txt --lr 5 --arap_steps 100
    """
    print(f"Default num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(32)
    torch.set_printoptions(linewidth=200, sci_mode=False)

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=Path, help="Source mesh path.")
    parser.add_argument("target_mesh_path", type=Path, help="Target mesh path.")
    parser.add_argument("--handles", type=Path, help="The path of the handles to load.")
    parser.add_argument("--lr", default=5.0, type=float, help="Learning rate.")
    parser.add_argument(
        "--arap_steps", default=100, type=int, help="Num. steps for ARAP."
    )
    parser.add_argument(
        "--steps", default=250, type=int, help="Num. steps for outer loop."
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Mesh data type.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    args = parser.parse_args()
    main(
        args.mesh_path,
        args.target_mesh_path,
        args.handles,
        args.device,
        args.dtype,
        "iskra",
        args.lr,
        args.arap_steps,
        args.steps,
    )

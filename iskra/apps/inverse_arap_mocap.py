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
    handles_path: Path,
    markers_path: Path,
    target_markers_path: Path,
    device_name: str,
    dtype_name: str,
    method: str,
    lr: float,
    arap_steps: int,
    max_steps: int,
):
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)

    results_dir = Path.home() / "Dropbox" / "Results" / "iskra" / "arap_mocap"
    results_dir = results_dir / mesh_path.stem / f"{method}_{device_name}_{dtype_name}"

    # Load meshes
    mesh, _ = Mesh.from_path(mesh_path, dtype=dtype, device=device)
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    markers, _ = Mesh.from_path(markers_path, dtype=dtype, device=device)
    markers_verts = markers.geom.vertices

    target_markers, _ = Mesh.from_path(target_markers_path, dtype=dtype, device=device)
    target_markers_verts = target_markers.geom.vertices

    # Load handles
    marker_idx = torch.cdist(markers_verts, verts).argmin(dim=1)
    print("Handles loaded.")

    with Path(handles_path).open("r") as f:
        handle_idx = torch.tensor(
            [int(i) for i in f.readline().split(", ")],
            device=device,
            dtype=torch.int64,
        )

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
    loss = ((deformed[marker_idx] - target_markers_verts) ** 2).mean()
    loss.backward()

    print("Solving with libigl.")
    arap_data_igl = igl.ARAPData()
    arap_data_igl.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
    arap_data_igl.max_iter = 100
    igl.arap_precomputation(
        verts.cpu().numpy(),
        faces.cpu().numpy(),
        3,
        handle_idx.cpu().numpy(),
        arap_data_igl,
    )
    print("Solved with libigl.")

    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.register_surface_mesh(
            "mesh", verts.cpu().numpy(), faces.cpu().numpy(), enabled=False
        )
        ps_mesh = ps.register_surface_mesh(
            "deformed", deformed.detach().cpu().numpy(), faces.cpu().numpy()
        )
        ps_target_markers = ps.register_point_cloud(
            "target", target_markers_verts.detach().cpu().numpy(), enabled=True
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
            "-grad bc", -handles.grad.cpu(), enabled=True, length=0.15
        )
        ps_edges.add_scalar_quantity(
            "cotan", vert_vert_weights.cpu(), defined_on="edges"
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
                        bwd_max_iter=500,
                        verbose=False,
                    )
                    loss = ((deformed[marker_idx] - target_markers_verts) ** 2).mean()
                    print(f"Step {optim_step}.")
                    print(f"ARAP energy: {energy.mean().detach().cpu().item()}")
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
    python -m iskra.apps.inverse_arap_mocap \
        ~/Dropbox\ \(MIT\)/Data/iskra-data/arap/amass/amass.obj \
        ~/Dropbox\ \(MIT\)/Data/iskra-data/arap/amass/amass_handles.txt  \
        ~/Dropbox\ \(MIT\)/Data/iskra-data/arap/amass/amass_markers.obj \
        ~/Dropbox\ \(MIT\)/Data/iskra-data/arap/amass/amas_32_markers.obj \
        --steps 10
    """
    print(f"torch.num_threads: {torch.get_num_threads()}")

    parser = ArgumentParser(description="Demonstrates ARAP.")
    parser.add_argument("mesh_path", type=Path, help="Source mesh path.")
    parser.add_argument("handles_path", type=Path, help="Handle indices path.")
    parser.add_argument("markers_path", type=Path, help="Mocap markers path.")
    parser.add_argument("target_markers_path", type=Path, help="Target markers path.")
    parser.add_argument("--lr", default=3, type=float, help="Learning rate.")
    parser.add_argument(
        "--arap_steps", default=250, type=int, help="Num. steps for ARAP."
    )
    parser.add_argument(
        "--steps", default=250, type=int, help="Num. steps for outer loop."
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Mesh data type.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    args = parser.parse_args()
    main(
        args.mesh_path,
        args.handles_path,
        args.markers_path,
        args.target_markers_path,
        args.device,
        args.dtype,
        "iskra",
        args.lr,
        args.arap_steps,
        args.steps,
    )

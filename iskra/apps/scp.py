# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch

from iskra.mesh import Mesh
from iskra.parameterization import boundary_matrix, conformal_laplacian, scp_solve
from iskra.topology import boundary, get_subfaces, ordered_boundary_edges

if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates a SLIM parameterization.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, dtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    # Assome one boundary loop, and take first vertex of each edge:
    bdr = ordered_boundary_edges(boundary(faces))[0][:, 0]

    edges, face_edges, face_edge_sign = get_subfaces(faces)

    conformal_lap = conformal_laplacian(verts, faces, clamp_min=0.0)
    boundary_mat = boundary_matrix(mesh.n_vertices, faces, dtype=dtype)

    uv_opt = scp_solve(conformal_lap, boundary_mat)

    try:
        import polyscope as ps

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", verts.numpy(), faces.numpy())
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        ps_mesh.add_parameterization_quantity("rand param", uv_opt, enabled=True)

        ps_param_mesh = ps.register_surface_mesh(
            "param_mesh", uv_opt, faces.numpy(), edge_width=1
        )

        optimizing = False

        # def callback():
        #     global optimizing

        #     if ps.imgui.Button(
        #         "Start Optimization" if not optimizing else "Stop Optimizing"
        #     ):
        #         optimizing = not optimizing
        #     if optimizing:
        #         for _ in range(10):
        #             optimizer.zero_grad()
        #             energy = step_fn()
        #             optimizer.step(lambda: step_fn().mean())

        #         ps_param_mesh.update_vertex_positions(uv_opt.detach().numpy())
        #         ps_param_mesh.add_scalar_quantity(
        #             "energy",
        #             energy.detach().numpy(),
        #             defined_on="faces",
        #             enabled=True,
        #         )
        #         ps_mesh.add_parameterization_quantity(
        #             "rand param", uv_opt.detach().numpy(), enabled=True
        #         )

        # ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )

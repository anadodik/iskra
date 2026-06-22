# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch

import iskra.sparse as sp  # import repdiag, to_scipy
from iskra.dec import laplacian
from iskra.mesh import Mesh
from iskra.sparse_linalg import eigsh
from iskra.topology import boundary, get_subfaces, ordered_boundary_edges


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

    lap, mass = laplacian(verts, faces, clamp_min=0.0)
    area = vertex_area_matrix(mesh.n_vertices, mesh.faces, dtype=dtype)
    bdr_vertices = boundary(faces).flatten().unique()

    lhs = sp.repdiag(lap, 2) - 2 * area
    bdr_ii = torch.stack([bdr_vertices, bdr_vertices], 0)
    rhs_block = sp.coo_tensor(
        bdr_ii,
        torch.ones(bdr_ii.shape[1], dtype=dtype, device=device),
        size=[mesh.n_vertices, mesh.n_vertices],
    )
    rhs = sp.repdiag(rhs_block, 2)

    lhs_sp = lhs.scipy()
    rhs_sp = rhs.scipy()

    evals, evecs = eigsh(lhs, M=rhs, k=3, sigma=-1e-12, adjoint="individual")

    print(evecs[:, 2:3])
    uv_opt = evecs[:, 2:3].reshape(2, -1).mT

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

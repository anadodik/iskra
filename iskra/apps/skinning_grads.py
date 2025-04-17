# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path

import igl
import numpy as np
import torch

from iskra.directional import (
    face_connection_laplacian,
    face_tangent_bundle,
    smooth_frame_field,
    to_extrinsic_frame_field,
    to_extrinsic_n_rosy,
    to_intrinsic_frame_field,
    to_intrinsic_n_rosy,
)
from iskra.fem import grad
from iskra.geometry import dual_quat
from iskra.geometry.normals import triangle_normals
from iskra.geometry.volume import edge_lengths, triangle_areas
from iskra.mesh import Mesh
from iskra.random import sample_circle
from iskra.sparse import diag, min_quadratic_energy, scipy_to_torch
from iskra.topology import edge_flaps, face_index, get_subfaces

if __name__ == "__main__":
    device = "cpu"
    mesh_name = "hand"
    mesh_path = (
        Path().home()
        / "neural-biharmonic-weights-data"
        / "dataset-3d"
        / f"{mesh_name}.obj"
    )
    handles_path = (
        Path().home()
        / "neural-biharmonic-weights-data"
        / "dataset-3d"
        / f"{mesh_name}_handles_bones.obj"
    )
    mesh = Mesh.from_path(mesh_path, device=device)
    handles = Mesh.from_path(handles_path, device=device)
    verts = mesh.geom.vertices
    faces = mesh.topo.faces

    weights_path = Path().home() / "experiments" / "bc_off" / mesh_name / "weights.npz"
    weights_file = np.load(weights_path)
    weights = torch.tensor(weights_file["weights"], dtype=torch.float32, device=device)

    grad_x, grad_y, grad_z = grad(mesh.geom.vertices, mesh.topo.faces)
    grad_weights = torch.stack(
        [grad_x @ weights, grad_y @ weights, grad_z @ weights], -1
    )

    k = 1
    grad_mags = torch.linalg.vector_norm(grad_weights, dim=-1, ord=2)
    top_grad_mags, top_k_idcs = torch.topk(grad_mags, k, dim=-1)
    top_grads = torch.gather(grad_weights, -2, top_k_idcs[:, :, None].expand(-1, k, 3))
    top_grads = 0.01 * torch.nn.functional.normalize(top_grads, p=2, dim=-1)
    sources = torch.nonzero(top_grad_mags[:, 0] > 8)[:, 0]
    print(sources.shape)

    flaps = edge_flaps(mesh.topo.faces)
    tangents, binormals, connection, _ = face_tangent_bundle(
        mesh.geom.vertices, mesh.topo.faces, flaps
    )
    t_source, b_source = tangents[sources], binormals[sources]
    intrinsic_u = 1j**2 * to_intrinsic_n_rosy(
        top_grads[sources, 0, :], t_source, b_source, 2
    )
    extrinsic_u = to_extrinsic_n_rosy(intrinsic_u, t_source, b_source, 2)

    transported_uv = smooth_frame_field(
        verts, faces, flaps, connection, partial_idcs=sources, partial_vals=intrinsic_u
    )
    extrinsic_frame = to_extrinsic_frame_field(transported_uv, tangents, binormals)
    # extrinsic_frame = torch.nn.functional.normalize(extrinsic_frame, p=2, dim=-1)

    import polyscope as ps

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        "mesh",
        mesh.geom.vertices.cpu().numpy(),
        mesh.topo.faces.cpu().numpy(),
        transparency=0.7,
    )
    ps_handles = ps.register_curve_network(
        "handles",
        handles.geom.vertices.cpu().numpy(),
        handles.topo.faces.cpu().numpy(),
    )
    for w_i in range(weights.shape[1]):
        ps_mesh.add_scalar_quantity(f"weight_{w_i}", weights[:, w_i])

    source_vs = torch.zeros([extrinsic_frame.shape[0], 2, 3])
    source_vs[sources, ...] = extrinsic_u
    for i in range(2):
        ps_mesh.add_vector_quantity(
            f"source_{i}",
            source_vs[:, i, :],
            radius=0.0015,
            length=0.015,
            color=(0.1, 0.9, 0.1),
            defined_on="faces",
            enabled=True,
        )

    for i in range(4):
        ps_mesh.add_vector_quantity(
            f"extrinsic_{i}",
            extrinsic_frame[:, i, :],
            radius=0.002,
            length=0.005,
            color=(0.9, 0.1, 0.1),
            defined_on="faces",
            enabled=True,
        )
    # for i in range(k):
    #     ps_mesh.add_vector_quantity(
    #         f"grad_weight{i}",
    #         top_grads[:, i, :].cpu().numpy(),
    #         radius=0.002,
    #         length=0.01,
    #         color=(0.9, 0.1, 0.1),
    #         defined_on="faces",
    #         enabled=True,
    #     )
    # for i in range(weights.shape[-1]):
    #     ps_mesh.add_scalar_quantity(
    #         f"weight{i}",
    #         weights[:, i].cpu().numpy(),
    #         defined_on="vertices",
    #         enabled=i == 10,
    #     )
    #     ps_mesh.add_vector_quantity(
    #         f"grad_weight{i}",
    #         grad_weights[:, i, :].cpu().numpy(),
    #         radius=0.002,
    #         length=0.01,
    #         color=(0.9, 0.1, 0.1),
    #         defined_on="faces",
    #         enabled=i == 10,
    #     )
    ps.show()

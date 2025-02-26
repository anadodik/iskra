# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from typing import Literal

import torch

from iskra.geometry.volume import volume_form, volume_form_intrinsic
from iskra.sparse import diag
from iskra.topology import face_index, reduce_on_subface


def mass(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return diag(dual_volume)


def mass_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return diag(1 / dual_volume)


def mass_intrinsic(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return diag(vertex_areas)


def mass_intrinsic_inv(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return diag(1 / vertex_areas)


def grad(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.shape[-1] != 3:
        raise ValueError("Grad implemented only for triangle meshes.")
    # TODO(anadodik): make it into a batched [3, n_faces, n_vertices] tensor.
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    device = vertices.device

    triangles = face_index(vertices, faces)
    edge_21 = triangles[:, 2, :] - triangles[:, 0, :]
    edge_13 = triangles[:, 1, :] - triangles[:, 0, :]

    face_normals = torch.linalg.cross(edge_21, -edge_13)
    double_face_areas = torch.linalg.vector_norm(face_normals, dim=-1, keepdim=True)
    face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=-1)

    rot_edge_21 = torch.linalg.cross(face_normals, edge_21) / double_face_areas
    rot_edge_13 = torch.linalg.cross(face_normals, edge_13) / double_face_areas

    idx_i = torch.cat([torch.arange(0, n_faces, device=device)] * 4)
    idx_j = torch.cat([faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 0]])
    idx = torch.stack([idx_i, idx_j])
    values = torch.cat([rot_edge_13, -rot_edge_13, rot_edge_21, -rot_edge_21])

    grad_x = torch.sparse_coo_tensor(idx, values[:, 0], size=[n_faces, n_vertices])
    grad_y = torch.sparse_coo_tensor(idx, values[:, 1], size=[n_faces, n_vertices])
    grad_z = torch.sparse_coo_tensor(idx, values[:, 2], size=[n_faces, n_vertices])
    return grad_x, grad_y, grad_z


def laplacian(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weight_type: Literal["cotan", "uniform"] = "cotan",
) -> torch.Tensor:
    pass


def laplacian_intrinsic(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
    faces: torch.Tensor,
    weight_type: Literal["cotan", "uniform"] = "cotan",
) -> torch.Tensor:
    pass

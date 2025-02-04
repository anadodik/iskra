# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry import cotan_weights, cotan_weights_intrinsic, volume_form
from iskra.geometry.volume import volume_form_intrinsic
from iskra.sparse import diag
from iskra.topology import face_index, incidence_matrix, reduce_on_subface


def d_01(edges: torch.Tensor) -> torch.Tensor:
    return incidence_matrix(edges, signed=True)


def d_10(edges: torch.Tensor) -> torch.Tensor:
    return d_01(edges).mT


def d_12(faces: torch.Tensor) -> torch.Tensor:
    return incidence_matrix(faces, signed=True)


def d_21(faces: torch.Tensor) -> torch.Tensor:
    return d_21(faces).mT


def hodge_0(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    area = volume_form(embedded_faces)
    vertex_areas = reduce_on_subface(area, faces, vertices.shape[0], "sum") / 3
    return diag(vertex_areas)


def hodge_0_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    area = volume_form(embedded_faces)
    vertex_areas = reduce_on_subface(area, faces, vertices.shape[0], "sum") / 3
    return diag(1 / vertex_areas)


def hodge_1(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights(vertices, faces)
    if clamp_min is not None:
        cot = cot.clamp(clamp_min)
    return diag(cot)


def hodge_1_inv(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights(vertices, faces)
    if clamp_min is not None:
        cot = cot.clamp(clamp_min)
    return diag(1 / cot)


def hodge_2(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volumes = volume_form(embedded_faces)
    return diag(volumes)


def hodge_2_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    inv_volumes = 1 / volume_form(embedded_faces)
    return diag(inv_volumes)


def laplacian(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lap = d_10(vertices, faces) @ hodge_1(vertices, faces) @ d_01(vertices, faces)
    mass = hodge_0(vertices, faces)
    return lap, mass


def hodge_0_intrinsic(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
    faces: torch.Tensor,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    vertex_areas = reduce_on_subface(area, faces, n_vertices, "sum") / 3
    return diag(vertex_areas)


def hodge_0_intrinsic_inv(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
    faces: torch.Tensor,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    vertex_areas = reduce_on_subface(area, faces, n_vertices, "sum") / 3
    return diag(1 / vertex_areas)


def hodge_1_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return diag(cot)


def hodge_1_intrinsic_inv(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return diag(1 / cot)


def hodge_2_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    volumes = volume_form_intrinsic(edge_lengths, face_to_edge)
    return diag(volumes)


def hodge_2_intrinsic_inv(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    volumes = volume_form_intrinsic(edge_lengths, face_to_edge)
    return diag(1 / volumes)

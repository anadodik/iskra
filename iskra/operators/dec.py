# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry import cotan_weights, cotan_weights_intrinsic, volume_form
from iskra.sparse import diag
from iskra.topology import face_index, incidence_matrix


def differential_01(edges: torch.Tensor) -> torch.Tensor:
    return incidence_matrix(edges, signed=True)


def differential_10(edges: torch.Tensor) -> torch.Tensor:
    return differential_01(edges).mT


def differential_12(faces: torch.Tensor) -> torch.Tensor:
    return incidence_matrix(faces, signed=True)


def differential_21(faces: torch.Tensor) -> torch.Tensor:
    return differential_21(faces).mT


def hodge_0(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    cot = cotan_weights(vertices, faces)
    return diag(cot)


def hodge_1(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    cot = cotan_weights(vertices, faces)
    return diag(cot)


def hodge_1_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    cot = cotan_weights(vertices, faces)
    return diag(1 / cot)


def hodge_1_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return diag(cot)


def hodge_1_intrinsic_inv(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return diag(1 / cot)


def hodge_2(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces, squeeze=False)
    volumes = volume_form(embedded_faces)
    return diag(volumes)


def hodge_2_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces, squeeze=False)
    inv_volumes = 1 / volume_form(embedded_faces)
    return diag(inv_volumes)

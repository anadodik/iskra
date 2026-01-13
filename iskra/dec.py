# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch

import iskra.sparse as sp
from iskra.geometry import cotan_weights, cotan_weights_intrinsic, volume_form
from iskra.geometry.volume import edge_lengths, volume_form_intrinsic
from iskra.profiling import profile_fn
from iskra.topology import face_index, get_subfaces, incidence_matrix, reduce_on_subface


@profile_fn(name="d_01")
def d_01(faces: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    edges, _, _ = get_subfaces(faces, 1)
    return incidence_matrix(edges, signed=True).to(dtype=dtype)


@profile_fn(name="d_10")
def d_10(faces: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    edges, _, _ = get_subfaces(faces, 1)
    derivative = d_01(edges, dtype=dtype).mT.coalesce()
    return derivative


def d_12(faces: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    triangles, _, _ = get_subfaces(faces, 2)
    return incidence_matrix(triangles, signed=True).to(dtype=dtype)


def d_21(faces: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    triangles, _, _ = get_subfaces(faces, 2)
    return d_12(triangles).mT.to(dtype=dtype)


def hodge_0(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    volume = volume_form(face_index(vertices, faces))
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return sp.diag(dual_volume)


def hodge_0_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    volume = volume_form(face_index(vertices, faces))
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return sp.diag(1 / dual_volume)


@profile_fn(name="hodge_1")
def hodge_1(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> torch.Tensor:
    if faces.shape[-1] == 2:
        return sp.diag(1 / edge_lengths(face_index(vertices, faces)))
    elif faces.shape[-1] == 3:
        weights = cotan_weights(vertices, faces, clamp_min)
        return sp.diag(weights)
    else:
        raise ValueError(f"hodge_1 not implemented for faces.shape={faces.shape}.")


def hodge_1_inv(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> torch.Tensor:
    if faces.shape[-1] == 2:
        return sp.diag(edge_lengths(face_index(vertices, faces)))
    elif faces.shape[-1] == 3:
        weights = cotan_weights(vertices, faces)
        if clamp_min is not None:
            weights = weights.clamp(clamp_min)
        return sp.diag(-1 / weights)
    else:
        raise ValueError(f"hodge_1_inv not implemented for faces.shape={faces.shape}.")


def hodge_2(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volumes = volume_form(embedded_faces)
    return sp.diag(volumes)


def hodge_2_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    inv_volumes = 1 / volume_form(embedded_faces)
    return sp.diag(inv_volumes)


def laplacian(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    derivative = d_01(faces, dtype=vertices.dtype)
    hodge = hodge_1(vertices, faces, clamp_min)
    lap = sp.matmul(sp.matmul(derivative.mT, hodge), derivative)
    mass = hodge_0(vertices, faces)
    return lap, mass


def laplacian_from_weights(weights: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    dtype = weights.dtype
    derivative = d_01(faces, dtype=dtype)
    hodge = sp.diag(weights)
    lap = sp.matmul(sp.matmul(derivative.mT, hodge), derivative)
    return lap


def hodge_0_intrinsic(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return sp.diag(vertex_areas)


def hodge_0_intrinsic_inv(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return sp.diag(1 / vertex_areas)


def hodge_1_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return sp.diag(cot)


def hodge_1_intrinsic_inv(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    # TODO(anadodik): should also work for polyline meshes
    cot = cotan_weights_intrinsic(edge_lengths, face_to_edge)
    return sp.diag(1 / cot)


def hodge_2_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    volumes = volume_form_intrinsic(edge_lengths, face_to_edge)
    return sp.diag(volumes)


def hodge_2_intrinsic_inv(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    volumes = volume_form_intrinsic(edge_lengths, face_to_edge)
    return sp.diag(1 / volumes)

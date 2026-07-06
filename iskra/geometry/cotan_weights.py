# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.volume import edge_lengths, triangle_areas_intrinsic
from iskra.topology import face_index, get_subfaces, reduce_on_subface


def cotan_weights(
    vertices: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = None
) -> torch.Tensor:
    """Computes cotangent stiffness edge weights.

    This function calls `iskra.cotan_weights_intrinsic()` under the hood.

    Args:
        vertices (Tensor[Float, [V, 3]]): Vertex positions.
        faces (Tensor[Int, [F, 3]]): Triangle face indices.
        clamp_min (float | None): Optionally clamp the cotan weights to a minimum value.
            Often used to prevent negative weights. Defaults to None.

    Returns:
        Tensor[Float, [E]]: Edge weights. Edge ordering corresponds to one obtained
            via `iskra.topology.get_subfaces(faces)`.
    """
    edges, face_to_edge, _ = get_subfaces(faces)
    lines = face_index(vertices, edges)
    lengths = edge_lengths(lines)
    return cotan_weights_intrinsic(lengths, face_to_edge, clamp_min)


def cotan_weights_intrinsic(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    clamp_min: float | None = None,
) -> torch.Tensor:
    """Computes cotangent stiffness edge weights from edge lengths.

    Args:
        edge_lengths (Tensor[Float, [E, 1]]): Triangle edge lengths.
        face_to_edge (Tensor[Int, [F, 3]]): Triangle to edge indices,
            e.g., obtained via `iskra.topology.get_subfaces(faces)`.
        clamp_min (float | None): Optionally clamp the cotan weights to a minimum value.
            Often used to prevent negative weights. Defaults to None.

    Returns:
        Tensor[Float, [E]]: Edge weights. Edge ordering corresponds to one obtained
            via `iskra.topology.get_subfaces(faces)`.
    """
    face_edge_lengths = face_index(edge_lengths, face_to_edge)
    area = triangle_areas_intrinsic(edge_lengths, face_to_edge)
    face_edge_lengths_sq = face_edge_lengths**2

    # For each edge in a triangle, its value is the sum of the values
    # on the other two edges, minus its own value.
    cot_ij = (
        -face_edge_lengths_sq
        + face_edge_lengths_sq[:, (1, 0, 0)]
        + face_edge_lengths_sq[:, (2, 2, 1)]
    )
    cot_ij = cot_ij / area[:, None] / 8

    edge_cot_weights = reduce_on_subface(
        cot_ij, face_to_edge, edge_lengths.shape[0], "sum", data_ndim=0
    )

    if clamp_min is not None:
        edge_cot_weights = edge_cot_weights.clamp(clamp_min)
    return edge_cot_weights

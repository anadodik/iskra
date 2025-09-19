# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.volume import edge_lengths
from iskra.topology import face_index, get_subfaces


def cotan_weights(vertices: torch.Tensor, faces: torch.Tensor):
    edges, face_to_edge, _ = get_subfaces(faces)
    lines = face_index(vertices, edges)
    lengths = edge_lengths(lines)
    return cotan_weights_intrinsic(lengths, face_to_edge)


def cotan_weights_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    face_edge_lengths = edge_lengths[face_to_edge.flatten()].reshape(
        *face_to_edge.shape, *edge_lengths.shape[1:]
    )
    semiperimeters = 0.5 * face_edge_lengths.sum(-1, keepdim=True)
    double_area = 2 * torch.sqrt(
        semiperimeters[:, 0] * torch.prod(semiperimeters - face_edge_lengths, dim=-1)
    )
    face_edge_lengths_sq = face_edge_lengths**2

    edge_idx = [0, 1, 2]
    nbh_edge_idcs = [(1, 2), (0, 2), (0, 1)]
    edge_cot_weights = torch.zeros(
        [edge_lengths.shape[0]], dtype=edge_lengths.dtype, device=edge_lengths.device
    )
    for edge_i, nbh_edge_i in zip(edge_idx, nbh_edge_idcs):
        cot_ij = (
            -face_edge_lengths_sq[:, edge_i]
            + face_edge_lengths_sq[:, nbh_edge_i[0]]
            + face_edge_lengths_sq[:, nbh_edge_i[1]]
        )
        cot_ij = cot_ij / double_area / 4
        edge_cot_weights = edge_cot_weights.scatter_add(
            0, face_to_edge[:, edge_i], cot_ij
        )
    return edge_cot_weights

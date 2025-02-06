# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.topology import reduce_on_subface


def mass_matrix(vertex_areas: torch.Tensor) -> torch.Tensor:
    if vertex_areas.ndim > 1 and vertex_areas.shape[-1] == 1:
        vertex_areas = vertex_areas.squeeze(-1)
    n_vertices = vertex_areas.shape[0]
    ij = torch.arange(n_vertices, device=vertex_areas.device)[None, :].expand(2, -1)
    result = torch.sparse_coo_tensor(ij, vertex_areas, size=[n_vertices, n_vertices])
    return result.coalesce()


def mass_matrix_inv(vertex_areas: torch.Tensor) -> torch.Tensor:
    if vertex_areas.ndim > 1 and vertex_areas.shape[-1] == 1:
        vertex_areas = vertex_areas.squeeze(-1)
    n_vertices = vertex_areas.shape[0]
    ij = torch.arange(n_vertices, device=vertex_areas.device)[None, :].expand(2, -1)
    result = torch.sparse_coo_tensor(ij, vertex_areas, size=[n_vertices, n_vertices])
    return result.coalesce()


def mass_matrix_intrinsic(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    faces: torch.Tensor,
    n_vertices: int,
    inverse: bool = False,
) -> torch.Tensor:
    # TODO(anadodik): refactor
    face_edge_lengths = edge_lengths[face_to_edge.flatten()].reshape(
        *face_to_edge.shape, *edge_lengths.shape[1:]
    )
    semiperimeters = 0.5 * face_edge_lengths.sum(-1, keepdim=True)
    area = torch.sqrt(
        semiperimeters[:, 0] * torch.prod(semiperimeters - face_edge_lengths, dim=-1)
    )
    vertex_areas = reduce_on_subface(area, faces, n_vertices, "sum") / 3
    return mass_matrix(vertex_areas, inverse=inverse)

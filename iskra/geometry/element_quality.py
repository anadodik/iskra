# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.topology import scatter_vertex_values

# TODO: look at quality measures in this article
# https://people.sc.fsu.edu/~jburkardt/presentations/cg_lab_tetrahedrons.pdf


def triangle_altitudes(points: torch.Tensor) -> torch.Tensor:
    edge_list = [points[:, (i + 1) % 3, :] - points[:, i, :] for i in range(3)]
    edges = torch.stack(edge_list, 1)
    edge_lengths = torch.sqrt(
        torch.clamp(torch.sum(edges**2, dim=-1, keepdim=True), min=0)
    )
    semiperimeter = torch.sum(edge_lengths, dim=1, keepdim=True) / 2
    numerator = semiperimeter * torch.prod(
        semiperimeter - edge_lengths, dim=1, keepdim=True
    )
    numerator = 2 * torch.sqrt(torch.clamp(numerator, min=0))
    altitudes = numerator / edge_lengths
    altitudes[~altitudes.isfinite()] = 0.0
    return altitudes


def abs_tetrahedron_heights(
    vertices: torch.Tensor, tet_idcs: torch.Tensor
) -> torch.Tensor:
    dim = vertices.shape[-1]
    assert dim == 3

    n_tet_faces = 4
    n_tri_verts = 3

    face_idx_combinations = [
        (1, 2, 3),  # opposite 0
        (0, 2, 3),  # opposite 1
        (0, 1, 3),  # opposite 2
        (0, 1, 2),  # opposite 3
    ]
    tet_face_idcs = torch.stack(
        [tet_idcs[:, c_idx] for c_idx in face_idx_combinations], -2
    )
    tet_faces = scatter_vertex_values(vertices, tet_face_idcs.reshape(-1, dim))
    tet_faces = tet_faces.reshape(-1, n_tet_faces, n_tri_verts, dim)  # [B, 4, 3, D]
    tet_vertices = scatter_vertex_values(vertices, tet_idcs)  # [B, 4, D]
    heights_list = []
    for v_i in range(4):
        opposite_triangle = tet_faces[:, v_i, :, :]  # [B, 3, D]
        vertex = tet_vertices[:, v_i : (v_i + 1), :]  # [B, 1, D]
        relative_triangle = opposite_triangle - opposite_triangle[..., 0:1, :]
        relative_vertex = vertex - opposite_triangle[..., 0:1, :]

        normal = torch.linalg.cross(
            relative_triangle[..., 1:2, :], relative_triangle[..., 2:, :], dim=-1
        )
        triangle_area = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        normal /= triangle_area
        distance_to_plane = torch.sum(normal * relative_vertex, -1)
        heights_list.append(torch.abs(distance_to_plane))
    return torch.cat(heights_list, -1)

# Copyright (c) 2024 - present, Ana Dodik. All rights reserved.

import torch

import iskra.sparse as sp
from iskra.logging import getLogger

LOGGER = getLogger(__name__)


def upsample(
    faces: torch.Tensor,
    vertex_values: torch.Tensor,
    edge_values: torch.Tensor,
    face_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = faces.device
    n_vertices = vertex_values.shape[0]
    per_face_half_edges = torch.stack(
        [faces[:, (1, 2)], faces[:, (2, 0)], faces[:, (0, 1)]], -2
    )
    half_edges = torch.flatten(per_face_half_edges, -3, -2)
    edges, half_edge_sort = torch.sort(half_edges, -1)
    edges, edge_to_half_edge_idx = torch.unique(edges, dim=-2, return_inverse=True)

    face_edge = torch.arange(edges.shape[0], device=device)[
        edge_to_half_edge_idx
    ].reshape(-1, 3)

    half_edge_values = edge_values[edge_to_half_edge_idx]
    half_edge_values[half_edge_sort[:, 0] == 1] *= -1
    half_edge_values = half_edge_values.reshape(-1, 3)

    vertex_values_sub = torch.cat(
        [
            vertex_values,
            0.5 * (vertex_values[edges[:, 0], :] + vertex_values[edges[:, 1], :]),
        ]
    )
    faces_e21_sub = torch.stack(
        [faces[:, 0], n_vertices + face_edge[:, 2], n_vertices + face_edge[:, 1]], -1
    )
    faces_e20_sub = torch.stack(
        [
            n_vertices + face_edge[:, 2],
            faces[:, 1],
            n_vertices + face_edge[:, 0],
        ],
        -1,
    )
    faces_e10_sub = torch.stack(
        [n_vertices + face_edge[:, 1], n_vertices + face_edge[:, 0], faces[:, 2]], -1
    )
    faces_center = torch.stack(
        [
            n_vertices + face_edge[:, 1],
            n_vertices + face_edge[:, 2],
            n_vertices + face_edge[:, 0],
        ],
        -1,
    )

    face_list = [faces_e21_sub, faces_e20_sub, faces_e10_sub, faces_center]
    faces_sub = torch.cat(face_list)

    ii = torch.arange(faces_sub.shape[0], device=device)
    jj = torch.cat(len(face_list) * [torch.arange(faces.shape[0], device=device)])
    face_to_face_sub = sp.coo_tensor(
        torch.stack([ii, jj]),
        torch.ones(faces_sub.shape[0], device=device),
        size=[faces_sub.shape[0], faces.shape[0]],
    )

    LOGGER.debug("face to face sub", face_to_face_sub.shape)
    LOGGER.debug("face values", face_values.shape)
    face_values_sub = face_to_face_sub @ face_values

    per_face_half_edges_sub = torch.stack(
        [faces_sub[:, (1, 2)], faces_sub[:, (2, 0)], faces_sub[:, (0, 1)]], -2
    )
    half_edges_sub = torch.flatten(per_face_half_edges_sub, -3, -2)
    edges_sub, half_edge_sort_sub = torch.sort(half_edges_sub, -1)
    edges_sub, edge_to_half_edge_sub, edge_counts = torch.unique(
        edges_sub, dim=-2, return_inverse=True, return_counts=True
    )
    inverse_edge_counts = edge_counts[edge_to_half_edge_sub]

    # The subdivided triangles are congruent to the original triangle scaled by a half.
    half_edge_values_sub = torch.cat(
        [
            0.5 * half_edge_values,
            0.5 * half_edge_values,
            0.5 * half_edge_values,
            torch.zeros_like(half_edge_values),
        ]
    )

    n_faces = faces.shape[0]
    half_edge_values_sub[3 * n_faces : 4 * n_faces, 0] = -0.5 * half_edge_values[:, 1]
    half_edge_values_sub[3 * n_faces : 4 * n_faces, 1] = -0.5 * half_edge_values[:, 2]
    half_edge_values_sub[3 * n_faces : 4 * n_faces, 2] = -0.5 * half_edge_values[:, 0]

    half_edge_values_sub = half_edge_values_sub.flatten()
    half_edge_values_sub /= inverse_edge_counts
    half_edge_values_sub[half_edge_sort_sub[:, 0] == 1] *= -1
    edge_values_sub = torch.full([edges_sub.shape[0]], 0.0, device=device)
    edge_values_sub.scatter_add_(
        0, edge_to_half_edge_sub, half_edge_values_sub.reshape(-1)
    )

    return faces_sub, vertex_values_sub, edge_values_sub, face_values_sub

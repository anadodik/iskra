# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch


def edge_length_normals(edges: torch.Tensor) -> torch.Tensor:
    edge_vector = edges[..., 1, :] - edges[..., 0, :]
    orth_vector = torch.tensor([1.0, -1.0], device=edges.device, dtype=torch.float32)
    orth_vector = torch.broadcast_to(orth_vector, edge_vector.shape)
    normal = edge_vector[..., (1, 0)] * orth_vector
    assert normal.shape[:-1] == edges.shape[:-2]
    assert normal.shape[-1] == 2
    return normal


def edge_normals(edges: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(edge_length_normals(edges), dim=-1)


def triangle_area_normals(triangles: torch.Tensor) -> torch.Tensor:
    if triangles.shape[-1] == 2:
        triangles = torch.nn.functional.pad(triangles, pad=(0, 1))
    assert len(triangles.shape) >= 2
    assert triangles.shape[-2] == 3
    assert triangles.shape[-1] == 3

    relative_triangles = triangles - triangles[..., 0:1, :]
    double_area_normals: torch.Tensor = torch.linalg.cross(
        relative_triangles[..., 1, :], relative_triangles[..., 2, :], dim=-1
    )
    return 0.5 * double_area_normals


def triangle_normals(triangles: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(triangle_area_normals(triangles), dim=-1)

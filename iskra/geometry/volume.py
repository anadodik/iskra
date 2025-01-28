# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.normals import triangle_area_normals


def edge_lengths(edges: torch.Tensor) -> torch.Tensor:
    edge_dir = edges[..., 1, :] - edges[..., 0, :]
    length: torch.Tensor = torch.linalg.vector_norm(edge_dir, dim=-1)
    return length


def triangle_areas(triangles: torch.Tensor) -> torch.Tensor:
    double_area_normals = triangle_area_normals(triangles)
    areas: torch.Tensor = torch.linalg.vector_norm(
        double_area_normals, dim=-1, keepdim=True
    )
    return areas


def _scalar_triple_product(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    return torch.sum(a * torch.linalg.cross(b, c), dim=-1, keepdim=True)


def tetrahedron_volumes(tets: torch.Tensor) -> torch.Tensor:
    batch_shape = tets.shape[:-2]
    assert tets.shape[-2] == 4
    assert tets.shape[-1] == 3
    tets = tets.reshape(-1, 4, 3)
    edge_dirs = tets[:, 1:, :] - tets[:, 0:1, :]
    volume = _scalar_triple_product(
        edge_dirs[:, 0, :],
        edge_dirs[:, 1, :],
        edge_dirs[:, 2, :],
    )
    return volume.reshape(*batch_shape, 1)

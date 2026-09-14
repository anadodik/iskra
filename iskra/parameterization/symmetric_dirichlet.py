# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry import triangle_coordinate_system
from iskra.topology import face_index


def triangle_to_local(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = face_index(verts, faces)
    _, t, b = triangle_coordinate_system(triangles)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    world_to_local = torch.stack([t, b], -2)
    local = world_to_local @ edge_vecs.mT
    return local


def uv_local(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # Do not project on a local coordinate frame because that will
    # leave us not knowing if there is a flip or not!
    triangles = face_index(uv, faces)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    return edge_vecs.mT


def symmetric_dirichlet_energy(
    rest_local: torch.Tensor, param_local: torch.Tensor, rest_areas: torch.Tensor
) -> torch.Tensor:
    jac = param_local @ torch.linalg.inv(rest_local)
    energy_fwd = (jac**2).sum((-2, -1))
    energy_bwd = (torch.linalg.inv(jac) ** 2).sum((-2, -1))
    energy = rest_areas * (energy_fwd + energy_bwd)

    is_flipped = torch.linalg.det(param_local.mT) <= 0
    energy = torch.where(is_flipped, float("inf"), energy)
    return energy

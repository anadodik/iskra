# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch


def extrude_boundary_polygon(
    vertices: torch.Tensor, faces: torch.Tensor, depth: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices_pos = torch.nn.functional.pad(vertices, (0, 1), value=depth)
    vertices_neg = torch.nn.functional.pad(vertices, (0, 1), value=-depth)
    vertices = torch.cat([vertices_pos, vertices_neg], 0)
    offset = vertices_pos.shape[0]
    faces_1 = torch.stack([faces[:, 0], faces[:, 0] + offset, faces[:, 1]], -1)
    faces_2 = torch.stack([faces[:, 0] + offset, faces[:, 1] + offset, faces[:, 1]], -1)
    faces = torch.cat([faces_1, faces_2], 0)
    return vertices, faces

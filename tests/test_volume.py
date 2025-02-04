# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

import numpy as np
import pytest
import torch

from iskra.geometry import (
    edge_lengths,
    tetrahedron_volumes,
    tetrahedron_volumes_intrinsic,
    triangle_areas,
    triangle_areas_intrinsic,
)
from iskra.topology import face_index, get_subfaces


@pytest.fixture
def tetrahedron() -> torch.Tensor:
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2, 3]],
    )
    return verts, faces


@pytest.fixture
def triangle() -> torch.Tensor:
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2]],
    )
    return verts, faces


def test_tetrahedron(tetrahedron: tuple[torch.Tensor, torch.Tensor]) -> None:
    verts, faces = tetrahedron
    expected_volumes = torch.tensor([1 / 6], device=verts.device)

    volumes = tetrahedron_volumes(face_index(verts, faces))
    torch.testing.assert_close(volumes, expected_volumes)

    edges, face_to_edge, _ = get_subfaces(faces, 1)
    lines = face_index(verts, edges)
    lengths = edge_lengths(lines)
    volumes = tetrahedron_volumes_intrinsic(lengths, face_to_edge)

    torch.testing.assert_close(volumes, expected_volumes)


def test_triangle(triangle: tuple[torch.Tensor, torch.Tensor]) -> None:
    verts, faces = triangle
    expected_volumes = torch.tensor([1 / 2], device=verts.device)

    volumes = triangle_areas(face_index(verts, faces))
    torch.testing.assert_close(volumes, expected_volumes)

    edges, face_to_edge, _ = get_subfaces(faces, 1)
    lines = face_index(verts, edges)
    lengths = edge_lengths(lines)
    volumes = triangle_areas_intrinsic(lengths, face_to_edge)

    torch.testing.assert_close(volumes, expected_volumes)

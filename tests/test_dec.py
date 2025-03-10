# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

import pytest
import torch

from iskra.dec import hodge_0, hodge_0_inv, hodge_1, hodge_1_inv, hodge_2, hodge_2_inv


@pytest.fixture
def triangles() -> torch.Tensor:
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2], [1, 3, 2]],
    )
    return verts, faces


@pytest.fixture
def tetrahedra() -> torch.Tensor:
    # TODO (anadodik): implement tet DEC
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2, 3], [4, 1, 3, 2]],
    )
    return verts, faces


@pytest.fixture
def edges() -> torch.Tensor:
    # TODO (anadodik): implement edge DEC
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1], [1, 2], [2, 0]],
        dtype=torch.int64,
    )
    return verts, faces


def test_triangles(triangles: tuple[torch.Tensor, torch.Tensor]) -> None:
    verts, faces = triangles

    expected_hodge_0 = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], device=verts.device)
    torch.testing.assert_close(hodge_0(verts, faces).values(), expected_hodge_0)
    torch.testing.assert_close(hodge_0_inv(verts, faces).values(), 1 / expected_hodge_0)

    expected_hodge_1 = torch.tensor(
        [1 / 2, 1 / 2, 0.0, 1 / 2, 1 / 2], device=verts.device
    )
    torch.testing.assert_close(hodge_1(verts, faces).values(), expected_hodge_1)
    expected_hodge_1_inv = torch.tensor([2, 2, 1e7, 2, 2], device=verts.device)
    torch.testing.assert_close(
        hodge_1_inv(verts, faces, clamp_min=1e-7).values(), expected_hodge_1_inv
    )

    expected_hodge_2 = torch.tensor([1 / 2, 1 / 2], device=verts.device)
    torch.testing.assert_close(hodge_2(verts, faces).values(), expected_hodge_2)
    torch.testing.assert_close(hodge_2_inv(verts, faces).values(), 1 / expected_hodge_2)

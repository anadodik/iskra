# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

import numpy as np
import pytest
import torch

from iskra.geometry.distances import (
    closest_edge,
    edge_project,
    point_dist,
    point_dist_matrix,
    point_edge_dist,
    point_edge_dist_matrix,
    point_triangle_dist_matrix,
    tetrahedron_project,
    triangle_project,
)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def y() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def edges() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.5, 0.5], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def tri_x() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def triangles() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def tet_x() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def tetrahedra() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )


def test_point_dist(x: torch.Tensor, y: torch.Tensor) -> None:
    expected = torch.tensor([np.sqrt(2), 1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(point_dist(x, y), expected)

    with pytest.raises(ValueError):
        point_dist(x[None, ...], y)

    with pytest.raises(RuntimeError):
        point_dist(torch.cat([x, x]), y)

    expected = torch.tensor(
        [
            [np.sqrt(2), np.sqrt(2), 0.5 * np.sqrt(2)],
            [1.0, 1.0, 0.5 * np.sqrt(2)],
            [0.5 * np.sqrt(2), 0.5 * np.sqrt(2), 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(point_dist_matrix(x, y), expected)

    expected = torch.tensor([[1.0]], dtype=torch.float32)
    torch.testing.assert_close(point_dist_matrix(x[0, :], x[1, :]), expected)


def test_point_edge(x: torch.Tensor, edges: torch.Tensor) -> None:
    expected = torch.tensor([[0.0, 1.0], [0.5, 0.5], [0.5, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(edge_project(x, edges), expected)

    with pytest.raises(ValueError):
        edge_project(x[..., None, :], edges)

    expected = torch.tensor([1.0, 0.5 * np.sqrt(2), 0.5], dtype=torch.float32)
    torch.testing.assert_close(point_edge_dist(x, edges), expected)

    expected = torch.tensor([2, 2, 1], dtype=torch.long)
    closest_idx = closest_edge(x, edges)[-1]
    torch.testing.assert_close(closest_idx, expected)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5 * np.sqrt(2), 0.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(point_edge_dist_matrix(x, edges), expected)


def test_point_triangle(tri_x: torch.Tensor, triangles: torch.Tensor) -> None:
    expected = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]], dtype=torch.float32
    )
    torch.testing.assert_close(triangle_project(tri_x, triangles), expected)

    x_2d = torch.tensor([-0.5, -0.5], dtype=torch.float32)
    tri_2d = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, -2.0]], dtype=torch.float32)
    expected = torch.tensor([-0.5, -0.5], dtype=torch.float32)
    torch.testing.assert_close(triangle_project(x_2d, tri_2d), expected)

    expected = torch.tensor(
        [
            [0.5 * np.sqrt(2), 0.5 * np.sqrt(2), 0.0],
            [0.5 * np.sqrt(2), 0.5 * np.sqrt(2), 0.0],
            [0.0, 0.0, np.sqrt(3 * 1 / 3**2)],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(point_triangle_dist_matrix(tri_x, triangles), expected)


def test_point_tetrahedron(tet_x: torch.Tensor, tetrahedra: torch.Tensor) -> None:
    expected = torch.tensor(
        [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [0.1, 0.1, 0.1]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tetrahedron_project(tet_x, tetrahedra), expected)


if __name__ == "__main__":
    pytest.main([__file__])

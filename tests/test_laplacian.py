import pytest
import torch

from iskra.fem import laplacian


def test_cotan_laplacian_2d_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.long)

    lap = laplacian(vertices, faces, "cotan")

    gt = torch.tensor(
        [
            [-2.0, 1.0, 0.5, 0.5],
            [1.0, -1.0, 0.0, 0.0],
            [0.5, 0.0, -0.5, 0.0],
            [0.5, 0.0, 0.0, -0.5],
        ],
        dtype=vertices.dtype,
    )

    torch.testing.assert_close(lap.to_dense(), gt, rtol=0.0, atol=1e-6)


def test_uniform_laplacian_2d_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.long)

    lap = laplacian(vertices, faces, "uniform")

    gt = torch.tensor(
        [
            [-3.0, 1.0, 1.0, 1.0],
            [1.0, -3.0, 1.0, 1.0],
            [1.0, 1.0, -2.0, 0.0],
            [1.0, 1.0, 0.0, -2.0],
        ],
        dtype=vertices.dtype,
    )

    torch.testing.assert_close(lap.to_dense(), gt, rtol=0.0, atol=1e-6)


def test_cotan_laplacian_3d_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.long)

    lap = laplacian(vertices, faces, "cotan")

    gt = torch.tensor(
        [
            [-2.0, 1.0, 0.5, 0.5],
            [1.0, -1.0, 0.0, 0.0],
            [0.5, 0.0, -0.5, 0.0],
            [0.5, 0.0, 0.0, -0.5],
        ],
        dtype=vertices.dtype,
    )

    torch.testing.assert_close(lap.to_dense(), gt, rtol=0.0, atol=1e-6)


def test_uniform_laplacian_3d_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.long)

    lap = laplacian(vertices, faces, "uniform")

    gt = torch.tensor(
        [
            [-3.0, 1.0, 1.0, 1.0],
            [1.0, -3.0, 1.0, 1.0],
            [1.0, 1.0, -2.0, 0.0],
            [1.0, 1.0, 0.0, -2.0],
        ],
        dtype=vertices.dtype,
    )

    torch.testing.assert_close(lap.to_dense(), gt, rtol=0.0, atol=1e-6)

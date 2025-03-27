import numpy as np
import pytest
import torch

from iskra.fem import grad


def test_grad_polyline() -> None:
    vertices = torch.tensor([[0], [0.2], [0.5], [0.98], [1.0]])

    edges = torch.stack(
        (
            torch.arange(vertices.shape[0] - 1, dtype=torch.long),
            torch.arange(1, vertices.shape[0], dtype=torch.long),
        ),
        dim=1,
    )

    fun_zero_grad = torch.zeros_like(vertices) + 5

    fun_const_grad = 2 * vertices
    fun_other_grad = vertices**2

    g = grad(vertices, edges)

    const_res = g @ fun_zero_grad
    torch.testing.assert_close(
        const_res,
        torch.zeros_like(const_res),
        rtol=0,
        atol=1e-5,
    )

    fun_const_res = g @ fun_const_grad
    torch.testing.assert_close(
        fun_const_res,
        torch.full_like(fun_const_res, 2.0),
        rtol=0,
        atol=1e-5,
    )

    edge_centers = 0.5 * (vertices[:-1, :] + vertices[1:, :])
    fun_grad_res = g @ fun_other_grad
    torch.testing.assert_close(fun_grad_res, 2.0 * edge_centers, rtol=0, atol=1e-5)


def test_grad_2d_triangle() -> None:
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    gx, gy = grad(vertices, faces)

    g = torch.cat([gx.to_dense(), gy.to_dense()], dim=0)

    gt = torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=vertices.dtype)

    torch.testing.assert_close(g, gt, rtol=0, atol=1e-6)


def test_grad_3d_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    gx, gy, gz = grad(vertices, faces)

    g = torch.cat([gx.to_dense(), gy.to_dense(), gz.to_dense()], dim=0)

    gt = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=vertices.dtype
    )

    torch.testing.assert_close(g, gt, rtol=0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

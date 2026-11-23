import numpy as np
import pytest
import torch

from iskra.fem import grad, grad_edge, grad_tetrahedron, grad_triangle


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

    g = grad_edge(vertices, edges)

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


def test_grad_triangle_2d() -> None:
    verts = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

    g = grad_triangle(verts, faces).to_dense()
    gt = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
        ],
        dtype=verts.dtype,
    )

    torch.testing.assert_close(g, gt, rtol=0, atol=1e-6)


def test_grad_triangle_3d() -> None:
    h = np.sqrt(3) / 2
    verts = torch.tensor(
        [[-h, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, +h]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

    g = grad_triangle(verts, faces).to_dense()
    gt = torch.tensor(
        [
            [-1.0 / h, 0.5 / h, 0.5 / h, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, -0.5 / h, -0.5 / h, 1.0 / h],
        ],
        dtype=verts.dtype,
    )
    torch.testing.assert_close(g, gt, rtol=0, atol=1e-6)


def test_grad_tetrahedron() -> None:
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    g = grad_tetrahedron(verts, faces).to_dense()
    gt = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.5, -0.5, -0.5, 0.5],
            [0.0, -0.5, 0.5, -0.5, 0.5],
            [0.0, -0.5, -0.5, 0.5, 0.5],
        ],
        dtype=verts.dtype,
    )
    torch.testing.assert_close(g, gt, rtol=0, atol=1e-6)

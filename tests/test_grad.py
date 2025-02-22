import pytest
import torch

from iskra.grad import grad

# Taken from https://github.com/sgsellan/gpytoolbox/blob/main/test/test_grad.py


def test_grad_polyline() -> None:
    V = torch.tensor([[0], [0.2], [0.5], [0.98], [1.0]])
    E = torch.stack(
        (
            torch.arange(V.shape[0] - 1, dtype=torch.long),
            torch.arange(1, V.shape[0], dtype=torch.long),
        ),
        dim=1,
    )

    fun_zero_grad = torch.zeros_like(V) + 5
    fun_const_grad = 2 * V
    fun_other_grad = V**2
    G = grad(V, E)

    G_dense = torch.tensor(G.todense(order="C"), dtype=V.dtype)

    torch.testing.assert_close(
        G_dense @ fun_zero_grad,
        torch.zeros_like(G_dense @ fun_zero_grad),
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        G_dense @ fun_const_grad,
        torch.full_like(G_dense @ fun_const_grad, 2.0),
        rtol=0,
        atol=1e-6,
    )

    edge_centers = (V[:-1, :] + V[1:, :]) / 2.0

    torch.testing.assert_close(
        G_dense @ fun_other_grad, 2.0 * edge_centers, rtol=0, atol=1e-6
    )


def test_grad_2d_triangle() -> None:
    V = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    F = torch.tensor([[0, 1, 2]], dtype=torch.long)

    G = grad(V, F)

    G_dense = torch.tensor(G.todense(), dtype=V.dtype)

    G_gt = torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=V.dtype)

    torch.testing.assert_close(G_dense, G_gt, rtol=0, atol=1e-6)


def test_grad_3d_triangle() -> None:
    V = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    F = torch.tensor([[0, 1, 2]], dtype=torch.long)

    G = grad(V, F)

    G_gt = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=V.dtype
    )

    G_dense = torch.tensor(G.todense(), dtype=V.dtype)

    torch.testing.assert_close(G_dense, G_gt, rtol=0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

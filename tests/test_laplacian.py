import numpy as np
import pytest
import torch

from iskra.cotan_laplacian import triangle_cot_laplacian, squared_edge_lengths
from iskra.geometry import cotan_weights


def test_laplacian_2d_triangle() -> None:
    V = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    F = torch.tensor([[0, 1, 2]], dtype=torch.long)

    L = triangle_cot_laplacian(V, F)

    gt = np.array([[-1.0, 0.5, 0.5], [0.5, -0.5, 0.0], [0.5, 0.0, -0.5]])

    torch.testing.assert_close(L.toarray(), gt, rtol=0.0, atol=1e-6)


def test_laplacian_3d_triangle() -> None:
    V = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    F = torch.tensor([[0, 1, 2]], dtype=torch.long)

    L = triangle_cot_laplacian(V, F)

    gt = np.array([[-1.0, 0.5, 0.5], [0.5, -0.5, 0.0], [0.5, 0.0, -0.5]])

    torch.testing.assert_close(L.toarray(), gt, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

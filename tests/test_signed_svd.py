# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from typing import Callable

import pytest
import torch

from iskra.signed_svd import signed_svd


def run_svd(
    svd_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mat = mat.clone()
    mat.requires_grad_(True)
    mat.grad = torch.zeros_like(mat)
    u, s, vh = svd_fn(mat)
    loss = loss_fn(u, s, vh)
    loss.backward()
    return u, s, vh, mat.grad


def loss_s_quadratic(u, s, vh):
    return (s**2).sum()


def loss_s_quadratic_shift(u, s, vh):
    return ((s - 0.2) ** 2).sum()


def loss_quadratic_shift(u, s, vh):
    return ((u - 0.1) ** 2).sum() + (s**2).sum() + ((vh - 0.2) ** 2).sum()


def loss_u_quadratic_shift(u, s, vh):
    return ((u - 0.1) ** 2).sum()


def loss_vh_quadratic_shift(u, s, vh):
    return ((vh - 0.1) ** 2).sum()


@pytest.fixture(
    params=[
        [[1.0, 0.5, 0.8], [0.1, 0.3, 0.5], [0.9, 0.2, 0.4]],
        [[-1.0, 0.5, 0.8], [0.3, 0.3, 0.5], [0.9, 0.2, 0.8]],
    ]
)
def mat(request):
    return torch.tensor(request.param)[None, ...]


@pytest.mark.parametrize(
    "loss_fn",
    [
        loss_s_quadratic,
        loss_s_quadratic_shift,
        loss_u_quadratic_shift,
        loss_vh_quadratic_shift,
        loss_quadratic_shift,
    ],
)
def test_signed_svd_gradients(mat, loss_fn):
    # Compares manual signed SVD implementation and its gradients with torch.svd.
    u_torch, s_torch, vh_torch, grad_torch = run_svd(torch.svd, loss_fn, mat)
    u_signed, s_signed, vh_signed, grad_signed = run_svd(signed_svd, loss_fn, mat)
    assert (torch.det(u_signed @ vh_signed) > 0).all()

    is_reflection = torch.det(u_torch @ vh_torch) < 0
    assert (s_signed[is_reflection][..., -1] < 0).all()

    torch.testing.assert_close(
        grad_torch[~is_reflection], grad_signed[~is_reflection], rtol=1e-4, atol=1e-5
    )

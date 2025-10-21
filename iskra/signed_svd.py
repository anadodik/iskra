# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch


class SignedSVDVals(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        x = inputs
        u, s, vh = output

        ctx.mark_non_differentiable(u, vh)

        ctx.save_for_backward(u, s, vh)
        ctx.save_for_forward(u, s, vh)

    @staticmethod
    def forward(x):
        u, s, vh = torch.linalg.svd(x)
        sign = torch.ones_like(s)
        sign[..., -1] = torch.sign(torch.linalg.det(u @ vh))
        # print(sign.shape, s.shape)

        signed_s = sign * s
        det_u = torch.linalg.det(u)
        det_v = torch.linalg.det(vh.mH)

        sign_u = torch.where(
            (det_u[..., None] < 0) & (det_v[..., None] > 0), sign, torch.ones_like(s)
        )
        u = u @ torch.diag_embed(sign_u)

        sign_v = torch.where(
            (det_u[..., None] > 0) & (det_v[..., None] < 0), sign, torch.ones_like(s)
        )
        vh = torch.diag_embed(sign_v) @ vh
        return u, signed_s, vh

    @staticmethod
    def jvp(ctx, grad_x):
        u, s, vh = ctx.saved_tensors

        grad_s = torch.einsum("...ij,...jk,...kl->...i", u.conj(), grad_x, vh.conj())
        return None, grad_s, None

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        u, s, vh = ctx.saved_tensors

        grad_mat = u @ torch.diag_embed(grad_s) @ vh
        return grad_mat


def signed_svdvals(
    mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return SignedSVDVals.apply(mat)  # noqa: F821

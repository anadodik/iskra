# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from typing import Callable

import torch

from iskra.profiling import profile_block, profile_fn


class SignedSVD(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        x = inputs
        u, s, vh = output

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
    def backward(
        ctx,
        grad_u: torch.Tensor | None,
        grad_s: torch.Tensor | None,
        grad_vh: torch.Tensor | None,
    ) -> torch.Tensor | None:
        u, s, vh = ctx.saved_tensors
        device = u.device
        dtype = u.dtype

        if grad_u is None and grad_s is None and grad_vh is None:
            return None

        if grad_u is None and grad_vh is None and grad_s is not None:
            return u @ torch.diag_embed(grad_s) @ vh

        if grad_u is None:
            grad_u = torch.zeros_like(u)
        if grad_s is None:
            grad_s = torch.zeros_like(s)
        if grad_vh is None:
            grad_vh = torch.zeros_like(vh)

        # # Line 3535: s_squared
        s_sq = s**2
        #
        # # Line 3536: s_squared_diff
        s_diff = s_sq.unsqueeze(-1) - s_sq.unsqueeze(-2)

        # # Line 3537-3538: Add eye to avoid division by zero
        # s_squared_diff = s_squared_diff + torch.eye(
        #     s_squared_diff.size(-1),
        #     device=s_squared_diff.device,
        #     dtype=s_squared_diff.dtype,
        # )

        # # Line 3540-3541: U^H @ grad_U
        # u_conj_t_gu = u.mH @ grad_u

        # # Line 3542-3543: grad_V^H @ V
        # ## Note: vh is already V^H, so we need grad_vh @ vh.mH
        # gvh_v_conj_t = grad_vh @ vh.mH

        # # Line 3545-3547: vT computation
        # vT = u_conj_t_gu * s.unsqueeze(-2) + s.unsqueeze(-1) * gvh_v_conj_t

        # # Line 3549: Divide by s_squared_diff
        # vT = vT / s_squared_diff

        # # Line 3551-3552: Set diagonal to grad_s
        # vT.diagonal(0, -2, -1).copy_(grad_s)

        # # Line 3554: Final gradient U @ vT @ V^H
        # grad_a = u @ vT @ vh

        # return grad_a
        diag_ones = torch.ones(s_diff.shape[:-1], device=device, dtype=dtype)
        s_diff = torch.diagonal_scatter(s_diff, diag_ones, dim1=-2, dim2=-1)
        eps = torch.finfo(s_diff.dtype).eps
        one_over_diff = torch.where(torch.abs(s_diff) > eps, 1.0 / s_diff, 0.0)

        uh_gu = u.mH @ grad_u
        vh_gv = grad_vh @ vh.mH
        # print(vh_gv)
        # ga = (uh_gu * s[..., None, :] + s[..., :, None] * vh_gv) * one_over_diff
        # ga = u @ (ga + torch.diag_embed(grad_s)) @ vh
        # return ga
        uh_gu = one_over_diff * uh_gu
        grad_u_mat = u @ (uh_gu @ torch.diag_embed(s)) @ vh
        # print("grad_u_mat", grad_u_mat)

        grad_s_mat = u @ torch.diag_embed(grad_s) @ vh

        vh_gv = one_over_diff.mT * vh_gv
        grad_vh_mat = u @ torch.diag_embed(s) @ vh_gv @ vh
        # print("grad_vh_mat", grad_vh_mat)

        grad_mat = grad_u_mat + grad_s_mat + grad_vh_mat
        return grad_mat


class PrintGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # No need to save anything since we're not modifying the gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print("Gradient:\n", grad_output)
        # Identity backward: pass gradients through unchanged
        return grad_output


def print_grad(x):
    """Identity function that prints gradient during backward."""
    return PrintGrad.apply(x)


@profile_fn(name="signed_svd")
def signed_svd(
    mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with profile_block(name="signed_svd.svd"):
        u, s, vh = torch.linalg.svd(mat)
    repeated_count = (s[..., 1:] == s[..., :-1]).any(-1).count_nonzero()
    if repeated_count > 0:
        print(
            f"Warning: detected {repeated_count} matrices with repeated singular values."
        )
    sign = torch.ones_like(s)
    sign[..., -1] = torch.sign(torch.linalg.det(u @ vh))

    signed_s = sign * s
    # I think that flipping both u and/or vh to ensure both are rotations
    # introduces additional discontinuities in derivatives?
    # Therefore we always flip u.
    u = u @ torch.diag_embed(sign)
    return u, signed_s, vh


def sk(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([O, -z, y], dim=-1),
            torch.stack([z, O, -x], dim=-1),
            torch.stack([-y, x, O], dim=-1),
        ],
        dim=-2,
    )


def sk_inv(B):
    # half of the antisymmetric differences
    return (
        torch.stack(
            [
                B[..., 2, 1] - B[..., 1, 2],
                B[..., 0, 2] - B[..., 2, 0],
                B[..., 1, 0] - B[..., 0, 1],
            ],
            dim=-1,
        )
        * 0.5
    )


class Polar3x3(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        r, p = output

        ctx.save_for_backward(x, r, p)
        ctx.save_for_forward(x, r, p)

    @staticmethod
    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u, s, vh = signed_svd(x)
        # TODO: use r.mT in ARAP
        r = u @ vh
        p = vh.mT @ torch.diag_embed(s) @ vh
        torch.testing.assert_close(r @ p, x, rtol=1e-4, atol=1e-5)
        return r, p

    @staticmethod
    def jvp(ctx, grad_x):
        x, r, s = ctx.saved_tensors  # s = symmetric stretch tensor (your "p")

        # --------- derivative steps ----------
        # C = R^T Ȧ - Ȧ^T R
        C = r.transpose(-1, -2) @ grad_x - grad_x.transpose(-1, -2) @ r

        # vector form of skew(C)
        c_vec = sk_inv(C)

        # (tr(S) I - S) inverse
        I = torch.eye(3, device=x.device, dtype=x.dtype)
        tr_s = s.diagonal(dim1=-2, dim2=-1).sum(-1)
        T = tr_s[:, None, None] * I - s
        T_inv = torch.linalg.inv(T)

        # m_vec = (tr(S)I - S)^(-1) * sk_inv(C)
        m_vec = (T_inv @ c_vec.unsqueeze(-1)).squeeze(-1)

        M = sk(m_vec)

        # Ṙ = R M
        r_dot = r @ M

        # Ṡ = R^T (Ȧ - Ṙ S)
        s_dot = r.transpose(-1, -2) @ (grad_x - r_dot @ s)

        return r_dot, s_dot

    @staticmethod
    def backward(
        ctx, grad_r: torch.Tensor | None, grad_p: torch.Tensor | None
    ) -> torch.Tensor | None:
        x, r, s = ctx.saved_tensors
        device = x.device
        dtype = x.dtype

        if grad_r is None and grad_p is None:
            return None

        if grad_r is None:
            grad_r = torch.zeros_like(s)
        if grad_p is None:
            grad_p = torch.zeros_like(r)

        # ==== adjoint of ṗ = Rᵀ(Ȧ − Ṙ S) ====
        # contributions:
        #   grad_A += R grad_p
        #   grad_(Ṙ) -= R grad_p Sᵀ  =  R grad_p S  (S is symmetric)

        grad_A = r @ grad_p  # term hitting Ȧ
        grad_Rdot_from_p = -(r @ grad_p @ s)  # term hitting Ṙ

        # ==== adjoint of ṙ = R M ====
        #   grad_M = Rᵀ grad_r
        #   grad_R += grad_r Mᵀ (but grad_R isn't needed; R has no grad input)

        # total grad_r includes contribution from ṗ through Ṙ
        grad_r_total = grad_r + grad_Rdot_from_p

        grad_M = r.transpose(-1, -2) @ grad_r_total

        # M is skew: grad_M contributes only through its skew part
        grad_M_vec = sk_inv(grad_M)

        # ==== adjoint of M_vec = T_inv sk⁻¹(C), where C = Rᵀ Ȧ − Ȧᵀ R ====
        # backprop through linear map M_vec = T_inv c_vec
        I = torch.eye(3, device=device, dtype=dtype)
        tr_s = s.diagonal(dim1=-2, dim2=-1).sum(-1)
        T = tr_s[:, None, None] * I - s
        T_inv = torch.linalg.inv(T)

        grad_c_vec = (T_inv.transpose(-1, -2) @ grad_M_vec.unsqueeze(-1)).squeeze(-1)

        # ==== adjoint of c_vec = sk⁻¹(C) ====
        grad_C = sk(grad_c_vec)  # skew-symmetric contribution only

        # ==== adjoint of C = Rᵀ Ȧ − Ȧᵀ R ====
        # dC/dȦ applied to G gives:
        #   grad_A += R G
        #   grad_A += - (Gᵀ Rᵀ)ᵀ = - R Gᵀ

        grad_A += r @ grad_C
        grad_A += -(r @ grad_C.transpose(-1, -2))

        return grad_A


polar_3x3: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = Polar3x3.apply


def closest_rot_3x3(x: torch.Tensor) -> torch.Tensor:
    return polar_3x3(x)[0]

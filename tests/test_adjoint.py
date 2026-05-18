# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import pytest
import torch

from iskra.adjoint import (
    compute_jacobians,
    compute_numerical_jacobian,
    make_adjoint_layer,
    make_adjoint_vjps,
    make_fixed_point_layer,
)
from iskra.dec import laplacian
from iskra.deformation.arap import arap_solve, arap_step
from iskra.geometry import cotan_weights
from iskra.sparse_linalg import gmres_solve
from iskra.topology import get_subfaces


@pytest.fixture
def tet() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    bc_idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    bc_vals = verts[bc_idx] - 1

    grad_deformed = torch.zeros_like(verts)
    grad_deformed[3] += -0.1
    return verts, faces, bc_idx, bc_vals, grad_deformed


def test_arap_adjoint(tet):
    verts, faces, bc_idx, bc_vals, grad_deformed = tet

    weights = cotan_weights(verts, faces)
    lap, _ = laplacian(verts, faces)

    edges, _, _ = get_subfaces(faces)
    _, edge_verts, _ = get_subfaces(edges)
    halfedges = torch.cat([edge_verts, edge_verts.flip(-1)], 0)
    halfedge_weights = torch.cat([weights, weights], 0)

    bc_vals = bc_vals.requires_grad_(True)
    # , max_iter=35
    deformed, _ = arap_solve(verts, bc_idx, bc_vals, halfedges, halfedge_weights, lap)
    deformed.backward(grad_deformed)

    jac_verts, jac_bc = compute_jacobians(
        arap_step,
        0,
        0,
        6,
        deformed,
        verts,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )
    jac_full = -torch.linalg.solve(jac_verts, jac_bc)

    num_jac = compute_numerical_jacobian(
        arap_solve,
        0,
        -1,
        1e-8,
        verts,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )
    num_grad = (grad_deformed.flatten() @ num_jac).reshape(*bc_vals.shape)
    torch.testing.assert_close(num_jac, jac_full, rtol=1e-4, atol=1e-5)

    vjp_deformed, vjp_bc_verts = make_adjoint_vjps(
        arap_step,
        (-1,),
        0,
        0,
        deformed,
        verts,
        halfedge_weights,
        halfedges,
        lap,
        bc_idx,
        bc_vals,
    )
    init = torch.randn_like(deformed)
    dl_df = -gmres_solve(
        lambda z: vjp_deformed(z)[0], grad_deformed, init, maxiter=200, tol=1e-12
    )
    manual_grad = vjp_bc_verts(dl_df)[0]
    torch.testing.assert_close(manual_grad, num_grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(bc_vals.grad, num_grad, rtol=1e-4, atol=1e-5)

    # print("NUM JACOBIAN:\n", num_jac)
    # print("JACOBIAN VERTS:\n", jac_verts)
    # print("JACOBIAN BC:\n", jac_bc)
    # print("JACOBIAN FULL:\n", -torch.linalg.solve(jac_verts, jac_bc))


def test_make_adjoint_layer_scalar() -> None:
    @torch.no_grad()
    def solver(x: torch.Tensor) -> torch.Tensor:
        return x**2

    def implicit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x**2

    layer = make_adjoint_layer(
        solver,
        implicit,
        param_args=0,
        sol_args=0,
        zero_args=0,
        bwd_max_iter=50,
        bwd_abs_tol=1e-14,
        bwd_rel_tol=1e-14,
    )

    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    grad_out = torch.tensor(1.0, dtype=torch.float64)

    # Test forward pass:
    y = layer(x)
    torch.testing.assert_close(y, torch.tensor(9.0, dtype=torch.float64))

    # Test backward pass:
    y.backward(grad_out)
    torch.testing.assert_close(x.grad, torch.tensor(6.0, dtype=torch.float64))

    # Test manual backward pass with numerical Jacobian:
    x_ift = x.detach().clone().requires_grad_(True)
    y_ift = x_ift**2
    jac_y, jac_x = compute_jacobians(implicit, 1, 0, 0, x_ift, y_ift)
    analytic = -torch.linalg.solve(jac_y, jac_x)
    torch.testing.assert_close(
        analytic.reshape(()),
        torch.tensor(6.0, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(x.grad, analytic.reshape(()), rtol=1e-6, atol=1e-8)

    grad_vec = grad_out.flatten()
    vjp_y, vjp_x = make_adjoint_vjps(implicit, 0, 1, 0, x_ift, y_ift)
    init = grad_vec + vjp_y(grad_vec)
    dl_df = -gmres_solve(
        vjp_y,
        grad_vec,
        init,
        max_iter=50,
        abs_tol=1e-14,
        rel_tol=1e-14,
    )
    manual_grad = vjp_x(dl_df)[0]
    torch.testing.assert_close(
        manual_grad.reshape(()),
        torch.tensor(6.0, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(x.grad, manual_grad.reshape(()), rtol=1e-6, atol=1e-8)


def test_make_fixed_point_layer() -> None:
    def fp_step(y: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 0.5 * y + a + b

    layer = make_fixed_point_layer(
        fp_step,
        iterates=(0, 0),
        param_args=1,
        fwd_max_iter=40,
        fwd_error_metric="delta",
        fwd_abs_tol=1e-14,
        fwd_rel_tol=1e-14,
        bwd_max_iter=50,
        bwd_abs_tol=1e-14,
        bwd_rel_tol=1e-14,
    )

    y0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    y = layer(y0, a, b)
    torch.testing.assert_close(
        y, torch.tensor(10.0, dtype=torch.float64), rtol=1e-9, atol=1e-9
    )

    y.backward(torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        a.grad, torch.tensor(2.0, dtype=torch.float64), rtol=1e-8, atol=1e-8
    )
    assert y0.grad is None
    assert b.grad is None

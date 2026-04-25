# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import pytest
import torch

import iskra.sparse as sp
from iskra.adjoint import (
    compute_jacobians,
    compute_numerical_jacobian,
    make_solver_layer,
    make_vjp,
)
from iskra.apps.arap import arap_solve, arap_step, make_arap_vjp
from iskra.dec import d_01, d_10, laplacian
from iskra.geometry import cotan_weights
from iskra.mesh import Mesh
from iskra.signed_svd import signed_svd
from iskra.sparse_linalg import gmres_solve
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface


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

    vjp_deformed, vjp_bc_verts = make_vjp(
        arap_step,
        0,
        0,
        (-1,),
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

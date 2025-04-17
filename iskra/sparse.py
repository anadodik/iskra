# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from functools import reduce

import numpy as np
import scipy.sparse
import torch
from cholespy import CholeskySolverF, MatrixType


def eye(n: int, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
    idx = torch.arange(n, device=device)
    ij = torch.stack(2 * [idx])
    values = torch.ones([n], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(ij, values, size=[n, n]).coalesce()


def diag(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    assert values.ndim == 1
    n = values.shape[0]
    idx = torch.arange(n, device=values.device)
    ii = torch.stack(2 * [idx])
    return torch.sparse_coo_tensor(ii, values, size=[n, n]).coalesce()


def scipy_to_torch(
    x: scipy.sparse.sparray, device: torch.device | str = "cpu"
) -> torch.Tensor:
    x_coo = x.tocoo()
    row = torch.tensor(x_coo.row, device=device)
    col = torch.tensor(x_coo.col, device=device)
    data = torch.tensor(x_coo.data, device=device)
    x_torch = torch.sparse_coo_tensor(torch.stack([row, col]), data, size=x.shape)
    x_torch = x_torch.to_sparse_csr()
    return x_torch


def torch_to_scipy(x: torch.Tensor) -> scipy.sparse.coo_array:
    x = x.to_sparse_coo().coalesce()
    data = x.values().cpu().numpy()
    idcs = x.indices().cpu().numpy().astype(np.int64)
    x_scipy = scipy.sparse.coo_array((data, (idcs[0], idcs[1])), shape=x.shape)
    return x_scipy


_INDEX_TYPE = slice | int | torch.Tensor | tuple[int, ...]


def _build_index_selection_mask(x: torch.Tensor, *indices: _INDEX_TYPE) -> torch.Tensor:
    assert x.layout == torch.sparse_coo
    new_shape = list(x.shape)
    mask = torch.ones(x.indices().shape[-1], device=x.device, dtype=torch.bool)
    for dim, idx in enumerate(indices):
        match idx:
            case None:
                continue
            case int():
                if idx < 0:
                    idx = x.shape[dim] + idx
                mask &= x.indices()[dim] == idx
                new_shape[dim] = 1
            case tuple() | torch.Tensor(dtype=torch.long):
                # TODO(anadodik): check tensor dimensions
                idx = tuple(i if i >= 0 else x.shape[dim] + i for i in idx)
                mask &= reduce(torch.logical_or, (x.indices()[dim] == i for i in idx))
                new_shape[dim] = len(idx)
            case torch.Tensor(dtype=torch.bool):
                idx = torch.nonzero(idx)
                mask &= reduce(torch.logical_or, (x.indices()[dim] == i for i in idx))
                new_shape[dim] = len(idx)
            case slice():
                start = idx.start if idx.start is not None else 0
                end = idx.stop if idx.stop is not None else x.shape[dim]
                mask &= (x._indices()[dim] >= start) & (x._indices()[dim] < end)
                new_shape[dim] = end - start
            case _:
                raise ValueError(
                    f"Unrecognized type of index at dim {dim}: {type(idx)}."
                )
    return mask, new_shape


def get_slice(x: torch.Tensor, *indices: _INDEX_TYPE) -> torch.Tensor:
    assert x.layout == torch.sparse_coo
    mask, new_shape = _build_index_selection_mask(x, *indices)
    selected_idx = x._indices()[:, mask]
    selected_val = x._values()[mask]

    for dim, idx in enumerate(indices):
        match idx:
            case None:
                continue
            case int():
                selected_idx[dim, :] = 0
            case tuple() | torch.Tensor() | slice():
                idx_map = torch.empty(x.shape[dim], dtype=torch.long, device=x.device)
                idx_map[idx] = torch.arange(new_shape[dim], device=x.device)
                selected_idx[dim, :] = idx_map[selected_idx[dim, :]]

    return torch.sparse_coo_tensor(
        selected_idx,
        selected_val,
        size=new_shape,
        check_invariants=False,
        is_coalesced=True,
    )


def fill_slice(
    x: torch.Tensor, fill_value: float | int, *indices: slice | int | tuple[int, ...]
) -> torch.Tensor:
    assert x.layout == torch.sparse_coo
    mask, _ = _build_index_selection_mask(x, *indices)
    selected_idx = x._indices()
    selected_val = x._values()
    selected_val[mask] = fill_value
    return torch.sparse_coo_tensor(
        selected_idx,
        selected_val,
        size=x.shape,
        check_invariants=False,
        is_coalesced=True,
    )


def zero_slice(
    x: torch.Tensor, *indices: slice | int | tuple[int, ...]
) -> torch.Tensor:
    return fill_slice(x, 0, *indices)


def append(
    x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    assert x.layout == torch.sparse_coo
    return torch.sparse_coo_tensor(
        torch.cat([x._indices(), indices.to(x.device)], -1),
        torch.cat([x._values(), values.to(x.device)], -1),
        size=x.shape,
    ).coalesce()


class CholespySolve(torch.autograd.Function):
    """Differentiable function to solve the linear system.

    This simply calls the solve methods implemented by the Solver classes.
    """

    @staticmethod
    def forward(
        ctx, solver: CholeskySolverF, b: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        ctx.solver = solver
        b = b.contiguous()
        if x is None:
            x = torch.zeros_like(b)
        solver.solve(b.detach(), x)
        return x

    @staticmethod
    def backward(ctx, forward_grad: torch.Tensor) -> tuple[None, torch.Tensor, None]:
        forward_grad = forward_grad.contiguous()
        b_grad = None
        if ctx.needs_input_grad[1]:
            x = torch.zeros_like(forward_grad)
            ctx.solver.solve(forward_grad, forward_grad, x)
        return None, b_grad, None


cholespy_solve = CholespySolve.apply


class CholeskySolver:
    """Cholesky solver.

    Precomputes the Cholesky decomposition of the system matrix and solves the
    system by back-substitution.
    """

    def __init__(self, mat: torch.Tensor):
        mat = mat.coalesce()
        self.solver = CholeskySolverF(
            mat.shape[0],
            mat.indices()[0],
            mat.indices()[1],
            mat.values(),
            MatrixType.COO,
        )

    def solve(self, b: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        return cholespy_solve(self.solver, b, x)


def min_quadratic_energy(
    system: torch.Tensor,
    rhs: torch.Tensor,
    known_idx: torch.Tensor,
    known_values: torch.Tensor,
) -> torch.Tensor:
    if rhs.ndim != known_values.ndim or (
        rhs.ndim > 1 and rhs.shape[-1] != known_values.shape[-1]
    ):
        raise ValueError(
            "rhs must have the same number of dim and same last dim as known_values. "
            f"rhs.shape = {rhs.shape}, known_values.shape = {known_values.shape}."
        )
    if not system.is_coalesced():
        system = system.coalesce()
    n_rows = system.shape[0]
    unknown_mask = torch.ones([n_rows], dtype=torch.bool, device=known_idx.device)
    unknown_mask[known_idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    system_uu = get_slice(system, unknown_idx, unknown_idx)
    if known_idx.nelement() > 0:
        system_uk = get_slice(system, unknown_idx, known_idx)
        known_part = system_uk @ known_values
    else:
        known_part = 0
    rhs_u = rhs[unknown_idx, ...]
    new_rhs = rhs_u - known_part

    if system_uu.dtype == torch.cfloat:
        # TODO: make complex numbers work on the GPU and with gradients
        system_uu_sp = torch_to_scipy(system_uu.detach().cpu())
        solver = scipy.sparse.linalg.splu(system_uu_sp)
        unknown = torch.tensor(
            solver.solve(new_rhs.detach().cpu().numpy()), device=rhs.device
        )
    else:
        solver = CholeskySolver(system_uu)
        unknown = solver.solve(new_rhs)

    result = torch.zeros_like(rhs)
    result[unknown_idx] = unknown
    result[known_idx] = known_values
    return result

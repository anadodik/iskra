# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from functools import reduce
from typing import Sequence, cast

import numpy as np
import scipy.sparse
import torch


def index_complement(n: int, idx: torch.Tensor) -> torch.Tensor:
    unknown_mask = torch.ones([n], dtype=torch.bool, device=idx.device)
    unknown_mask[idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    return unknown_idx


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


def get_diag(mat: torch.Tensor) -> torch.Tensor:
    assert mat.ndim >= 2
    assert mat.shape[-1] == mat.shape[-2]
    n = mat.shape[0]
    idcs = mat.indices()
    values = mat.values()
    diag_mask = idcs[-1] == idcs[-2]
    return torch.sparse_coo_tensor(
        idcs[:-1, diag_mask], values[diag_mask], size=mat.shape[:-1]
    ).coalesce()


def inv_diag(mat: torch.Tensor) -> torch.Tensor:
    return diag(1.0 / get_diag(mat))


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


_INDEX_TYPE = None | slice | int | torch.Tensor | tuple[int, ...]


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
    selected_idx = x.indices()[:, mask]
    selected_val = x.values()[mask]

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
    selected_idx = x.indices()
    selected_val = x.values()
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


def ravel_indices(
    indices: torch.Tensor, shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    linear: torch.Tensor = cast(torch.Tensor, 0)
    stride = 1
    for i, s in zip(reversed(indices), reversed(shape)):
        linear += i * stride
        stride *= s
    return linear


def unravel_index(
    linear: torch.Tensor, shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    linear = linear.clone()
    idx = []
    for s in reversed(shape):
        idx.append(linear % s)
        linear = linear // s
    return torch.stack(tuple(reversed(idx)))


def reshape(x: torch.Tensor, *shape: int) -> torch.Tensor:
    assert len(x.shape) == 2, x.shape[0] == x.shape[1]
    x = x.coalesce()
    indices = x.indices()
    values = x.values()
    new_indices = unravel_index(ravel_indices(indices, x.shape), shape)

    return torch.sparse_coo_tensor(new_indices, values, size=[*shape]).coalesce()


def repdiag(x: torch.Tensor, n_reps: int) -> torch.Tensor:
    assert len(x.shape) == 2, x.shape[0] == x.shape[1]
    x = x.coalesce()
    indices = x.indices()
    values = x.values()
    size = x.shape[0]

    return torch.sparse_coo_tensor(
        torch.cat([indices + i * size for i in range(n_reps)], -1),
        torch.cat(n_reps * [values], -1),
        size=[n_reps * size, n_reps * size],
    ).coalesce()


def cat(xs: Sequence[torch.Tensor], dim=0) -> torch.Tensor:
    assert isinstance(xs, Sequence)
    xs = [x.coalesce() for x in xs]
    shapes = [x.shape for x in xs]
    shape_0 = xs[0].shape
    new_shape = [*shape_0]
    for x in xs[1:]:
        if not (
            x.shape[:dim] == shape_0[:dim] and x.shape[dim + 1 :] == shape_0[dim + 1 :]
        ):
            raise ValueError(
                f"Shapes must match except in dimension {dim}. Got shapes: {shapes}."
            )
        new_shape[dim] += x.shape[dim]

    new_indices = torch.cat([x.indices() for x in xs], 1)
    new_values = torch.cat([x.values() for x in xs], 0)

    idcs_step = 0
    shape_step = 0
    for x in xs:
        indices = x.indices()
        new_indices[dim, idcs_step : idcs_step + indices.shape[1]] += shape_step
        idcs_step += indices.shape[1]
        shape_step += x.shape[dim]

    return torch.sparse_coo_tensor(new_indices, new_values, size=new_shape).coalesce()


def append(
    x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    assert x.layout == torch.sparse_coo
    return torch.sparse_coo_tensor(
        torch.cat([x.indices(), indices.to(x.device)], -1),
        torch.cat([x.values(), values.to(x.device)], -1),
        size=x.shape,
    ).coalesce()


def isect_indices(
    a_idx: torch.Tensor, b_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([a_idx, b_idx], dim=1)
    _, inverse, counts = torch.unique(
        combined, dim=1, return_inverse=True, return_counts=True
    )
    dupl_mask = torch.gather(counts == 2, 0, inverse)
    a_isect_mask = dupl_mask[: a_idx.shape[1]]
    b_isect_mask = dupl_mask[a_idx.shape[1] :]
    return a_isect_mask, b_isect_mask


def mul_sparse_sparse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(
            "Sparse-sparse multiplication currently only supports same-shaped tensors"
        )
    if not a.is_coalesced():
        a = a.coalesce()
    a_idx, a_val = a.indices(), a.values()

    if not b.is_coalesced():
        b = b.coalesce()
    b_idx, b_val = b.indices(), b.values()

    a_mask, b_mask = isect_indices(a_idx, b_idx)

    return torch.sparse_coo_tensor(
        a_idx[:, a_mask], a_val[a_mask] * b_val[b_mask], size=a.shape
    ).coalesce()


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_sparse and not b.is_sparse:
        return a * b
    elif a.is_sparse and b.is_sparse:
        return mul_sparse_sparse(a, b)
    elif b.is_sparse:
        a, b = b, a

    idx = a.indices()
    val = a.values()
    out_shape = torch.broadcast_shapes(a.shape, b.shape)

    b_idx = []
    for dim in range(a.dim()):
        b_idx.append(0 if b.shape[dim] == 1 else idx[dim])

    new_vals = val * b[tuple(b_idx)]
    return torch.sparse_coo_tensor(idx, new_vals, size=out_shape).coalesce()


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    swapped = False
    if not a.is_sparse and not b.is_sparse:
        return a @ b
    elif a.is_sparse and b.is_sparse:
        return torch.sparse.mm(a, b)
    elif b.is_sparse:
        swapped = True
        a, b = b.mT, a.mT

    if b.ndim == 1:
        result = torch.sparse.mm(a, b[:, None])[:, 0]
    elif a.ndim == 2 and b.ndim == 2:
        result = torch.sparse.mm(a, b)
    if swapped and result.ndim == 2:
        result = result.mT
    return result

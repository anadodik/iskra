# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from functools import reduce
from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    cast,
)

import numpy as np
import scipy.sparse
import torch

from iskra.profiling import profile_fn


def index_complement(n: int, idx: torch.Tensor) -> torch.Tensor:
    unknown_mask = torch.ones([n], dtype=torch.bool, device=idx.device)
    unknown_mask[idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    return unknown_idx


def isect_indices(
    a_idx: torch.Tensor, b_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checks which indices in two tensors apepar in both.

    Assumes unique sets of indices in both function inputs.

    Args:
        a_idx (torch.Tensor): First set of indices to compare.
        b_idx (torch.Tensor): Second set of indices to compare.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Masks (one per input),
            that are True if the index is repeated in the other tensor.
    """
    combined = torch.cat([a_idx, b_idx], dim=-1)
    _, inverse, counts = torch.unique(
        combined, dim=-1, return_inverse=True, return_counts=True
    )
    dupl_mask = torch.gather(counts == 2, 0, inverse)
    a_isect_mask = dupl_mask[: a_idx.shape[-1]]
    b_isect_mask = dupl_mask[a_idx.shape[-1] :]
    return a_isect_mask, b_isect_mask


def ravel_indices(
    indices: torch.Tensor, shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Converts a COO indices into a linear index corresponding to a strided layout.

    !!! example
        The sparse tensor
        ```
        a = [[1.0, 0.0],
             [0.0, 2.0]]
        ```
        has COO indices `a.indices() = [[0, 1], [0, 1]]`.
        The linear index otutput is `ravel_index(a.indices()) = [0, 3]`.

    Args:
        indices (Tensor[dim, nnz]): Indices of a sparse tensor in COO format.
        shape (torch.Size | tuple[int, ...]): Shape of the tensor.

    Returns:
        Tensor[nnz]: Linear indices into a strided layout tensor of same shape.
    """
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


class SparseTensor(torch.Tensor):
    """Sane sparse PyTorch Tensor.

    Builds upon and extends existing PyTorch sparse tensors,
    and offers a sane interface and sane defaults.
    """

    def __new__(
        cls,
        tensor: torch.Tensor,
        *,
        layout: Literal["coo", "csr"] = "coo",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool = False,
    ) -> "SparseTensor":
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "SparseTensor requires a torch.Tensor. "
                f"Got {type(tensor).__name__}. "
                "To construct from raw indices, use SparseTensor.from_coo() "
                "or SparseTensor.from_csr()."
            )
        if not tensor.is_sparse:
            raise TypeError(
                "SparseTensor does not accept dense tensors. "
                "Use SparseTensor.from_coo() or SparseTensor.from_csr() to "
                "construct from raw data, or convert first with .to_sparse()."
            )
        if isinstance(tensor, SparseTensor):
            tensor = tensor._tensor
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)

        if layout == "csr" and tensor.layout == torch.sparse_coo:
            tensor = tensor.to_sparse_csr()
        elif layout == "coo" and tensor.layout == torch.sparse_csr:
            tensor = tensor.to_sparse_coo()

        wrapper = torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)
        wrapper._tensor = tensor
        return wrapper

    @classmethod
    def from_coo(
        cls,
        indices: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        values: torch.Tensor,
        size: torch.Size | list[int] | tuple[int, ...] | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool = False,
        check_invariants: bool = False,
        is_coalesced: bool = True,
    ) -> "SparseTensor":
        """Construct a COO SparseTensor from data."""
        if isinstance(indices, Sequence) and not isinstance(indices, torch.Tensor):
            indices = torch.stack(indices, 0)
        t = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            check_invariants=check_invariants,
            is_coalesced=is_coalesced,
        )
        return cls(t)

    @classmethod
    def from_csr(
        cls,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        size: torch.Size | list[int] | tuple[int, ...] | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool = False,
    ) -> "SparseTensor":
        """Construct a CSR SparseTensor from data."""
        t = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return cls(t)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        matmul_funcs = {
            matmul,
            torch.matmul,
            torch.sparse.mm,
            torch.Tensor.__matmul__,
            torch.Tensor.__rmatmul__,
            SparseTensor.__matmul__,
            SparseTensor.__rmatmul__,
        }
        mul_funcs = {
            mul,
            torch.mul,
            torch.Tensor.__mul__,
            torch.Tensor.__rmul__,
            SparseTensor.__mul__,
            SparseTensor.__rmul__,
        }
        dense_funcs = {
            torch.Tensor.to_dense,
            torch.Tensor.indices,
            torch.Tensor.values,
            torch.Tensor._indices,
            torch.Tensor._values,
            torch.Tensor.crow_indices,
            torch.Tensor.col_indices,
        }
        non_tensor_funcs = {
            torch.Tensor.is_coalesced,
            torch.Tensor.dense_dim,
            torch.Tensor.dim,
            torch.Tensor.ndim,
            torch.Tensor.nelement,
            torch.Tensor.numel,
            torch.Tensor.shape,
            torch.Tensor.size,
        }

        if func in matmul_funcs:
            a, b = args
            with torch._C.DisableTorchFunctionSubclass():
                ret = matmul(a, b)
                if a.is_sparse and b.is_sparse:
                    return torch.Tensor._make_subclass(cls, ret, ret.requires_grad)
                else:
                    return ret
        elif func in mul_funcs:
            a, b = args
            with torch._C.DisableTorchFunctionSubclass():
                ret = mul(a, b)
        else:
            with torch._C.DisableTorchFunctionSubclass():
                ret = func(*args, **kwargs)

        if (
            isinstance(ret, torch.Tensor)
            and not isinstance(ret, cls)
            and func not in dense_funcs
        ):
            ret_sparse = torch.Tensor._make_subclass(cls, ret, ret.requires_grad)
            ret_sparse._tensor = ret
            ret = ret_sparse
        return ret

    def __matmul__(
        self, other: "SparseTensor | torch.Tensor"
    ) -> "SparseTensor | torch.Tensor":
        return matmul(self, other)

    def __rmatmul__(
        self, other: "SparseTensor | torch.Tensor"
    ) -> "SparseTensor | torch.Tensor":
        return matmul(self, other)

    def __mul__(
        self, other: "SparseTensor | torch.Tensor"
    ) -> "SparseTensor | torch.Tensor":
        return mul(self, other)

    def __rmul__(
        self, other: "SparseTensor | torch.Tensor"
    ) -> "SparseTensor | torch.Tensor":
        return mul(self, other)

    def reshape(self, *shape: int) -> "SparseTensor":
        return reshape(self, *shape)

    def scipy(self):
        return to_scipy(self)

    def torch_tensor(self) -> torch.Tensor:
        return self._tensor

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return get_slice(self, *index)
        else:
            return get_slice(self, index)

    def __setitem__(self, index, value):
        # Enforce lowercase keys and protect data types
        clean_key = str(key).lower()
        self._data[clean_key] = value


def coo_tensor(
    indices: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    values: torch.Tensor,
    size: torch.Size | list[int] | tuple[int, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    check_invariants=True,
    is_coalesced=False,
) -> SparseTensor:
    return SparseTensor.from_coo(
        indices,
        values,
        size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        check_invariants=check_invariants,
        is_coalesced=is_coalesced,
    )


def csr_tensor(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    size: torch.Size | list[int] | tuple[int, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> SparseTensor:
    return SparseTensor.from_csr(
        crow_indices,
        col_indices,
        values,
        size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def eye(
    n: int, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"
) -> SparseTensor:
    idx = torch.arange(n, device=device)
    values = torch.ones([n], dtype=dtype, device=device)
    return coo_tensor((idx, idx), values, size=[n, n], is_coalesced=True)


def diag(values: torch.Tensor) -> SparseTensor:
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    assert values.ndim == 1
    if not values.is_sparse:
        n = values.shape[0]
        idx = torch.arange(n, device=values.device)
        return coo_tensor((idx, idx), values, size=[n, n], is_coalesced=True)
    else:
        idx = values.indices()
        vals = values.values()
        ii = torch.cat([idx, idx[-1:]])
        shape = [*values.shape, values.shape[-1]]
        return coo_tensor(ii, vals, size=shape).coalesce()


def get_diag(mat: SparseTensor | torch.Tensor) -> SparseTensor:
    # TODO: document why it returns sparse.
    # maybe add a flag that determines whether dense or sparse is returned?
    assert mat.ndim >= 2
    assert mat.shape[-1] == mat.shape[-2]
    idcs = mat.indices()
    values = mat.values()
    diag_mask = idcs[-1] == idcs[-2]
    return coo_tensor(
        idcs[:-1, diag_mask], values[diag_mask], size=mat.shape[:-1]
    ).coalesce()


def inv_diag(mat: torch.Tensor) -> torch.Tensor:
    return diag(1.0 / get_diag(mat))


def from_scipy(
    x: scipy.sparse.sparray, device: torch.device | str = "cpu"
) -> SparseTensor:
    x_coo = x.tocoo()
    row = torch.tensor(x_coo.row, device=device)
    col = torch.tensor(x_coo.col, device=device)
    data = torch.tensor(x_coo.data, device=device)
    x_torch = coo_tensor((row, col), data, size=x.shape)
    x_torch = x_torch.to_sparse_csr()
    return x_torch


def to_scipy(x: SparseTensor | torch.Tensor) -> scipy.sparse.coo_array:
    x = x.detach().to_sparse_coo().coalesce()
    if x.values().is_cuda:
        data = x.values().cpu().numpy()
        idcs = x.indices().cpu().numpy().astype(np.int64)
    else:
        # if on CPU, .numpy creates view
        data = x.values().numpy()
        idcs = x.indices().numpy().astype(np.int64)
    x_scipy = scipy.sparse.coo_array((data, (idcs[0], idcs[1])), shape=x.shape)
    return x_scipy


_INDEX_TYPE = None | slice | int | torch.Tensor | tuple[int, ...]


@profile_fn(name="_build_index_selection_mask")
def _build_index_selection_mask(
    x: SparseTensor, *indices: _INDEX_TYPE
) -> tuple[torch.Tensor, list[int]]:
    if x.layout != torch.sparse_coo:
        raise ValueError("Attempted indexing into a non-COO tensor.")
    new_shape = list(x.shape)
    mask = torch.ones(x.indices().shape[-1], device=x.device, dtype=torch.bool)
    noneless_indices = [i for i in indices if i is not None]
    for dim, idx in enumerate(noneless_indices):
        if isinstance(idx, tuple):
            idx = torch.tensor(idx, device=x.device)
        match idx:
            case None:
                continue
            case int():
                if idx < 0:
                    idx = x.shape[dim] + idx
                mask &= x.indices()[dim] == idx
                new_shape[dim] = 1
            case torch.Tensor(dtype=torch.int64):
                # TODO(anadodik): check tensor dimensions
                idx = torch.where(idx >= 0, idx, x.shape[dim] + idx)
                x_idx_unq, inverse = torch.unique(x.indices()[dim], return_inverse=True)
                isect_unq, _ = isect_indices(x_idx_unq, idx)
                isect_mask = torch.gather(isect_unq, 0, inverse)
                mask &= isect_mask
                new_shape[dim] = len(idx)
            case torch.Tensor(dtype=torch.bool):
                idx = torch.nonzero(idx)
                mask &= reduce(torch.logical_or, (x.indices()[dim] == i for i in idx))
                new_shape[dim] = len(idx)
            case slice():
                if idx.step is not None and idx.step != 1:
                    raise ValueError(
                        f"Slicing only supports steps of 1. Got slice={idx}."
                    )
                start = idx.start if idx.start is not None else 0
                end = idx.stop if idx.stop is not None else x.shape[dim]
                mask &= (x._indices()[dim] >= start) & (x._indices()[dim] < end)
                new_shape[dim] = end - start
            case _:
                raise ValueError(
                    f"Unrecognized type of index at dim {dim}: {type(idx)}."
                )
    return mask, new_shape


@profile_fn(name="get_slice")
def get_slice(x: SparseTensor, *indices: _INDEX_TYPE) -> SparseTensor:
    """Slices a sparse tensor.

    !!! warning
        The behavior of this function is *not* the same PyTorch's [] operator.
        This function only slices a sparse Tensor via masking.

        Behavior with indices of types int, slice, and None is the same as
        dense tensor indexing. Indexing with tensors and tuples works differently.
        Specifically, if there are two or more indices which are tensors or tuples,
        it does _not_ use them to pick out the individual elements, rather it
        picks out the entire row or column selected by those indices.
        Therefore, it cannot be used with arbitrary integer indices to reorder and
        repeat certain elements.
        For example, if given:
        ```
        a =
           [[2., 0., 0., 0.],
            [6., 3., 0., 0.],
            [0., 0., 4., 0.],
            [0., 0., 0., 5.]]
        ```
        indexing `SparseTensor` gives:
        ```
        a[(0, 1), :] =
        a[0:2, :] =
        a[(True, True, False, False), :] =
            [[2., 0., 0., 0.],
             [6., 3., 0., 0.]]

        and

        a[(0, 1), (1, 2)] =
        a[0:2, 1:3] =
        a[(True, True, False, False), (False, True, True, False)] =
            [[0., 0.],
             [3., 0.]]
        ```
        whereas indexing `Tensor` gives:
        ```
        a[(0, 1), :] =
        a[0:2, :] =
        a[(True, True, False, False), :] =
            [[2., 0., 0., 0.],
             [6., 3., 0., 0.]]

        and

        a[[(0, 1), (1, 2)]] =
        a[[(True, False, True, False), (True, False, True, False)]] =
            [0., 0.]

        but

        a[0:2, 1:3] =
            [[0., 0.],
             [3., 0.]]
        ```

        Moreover, unlike dense tensors, where certain indexing operations can be
        achieved by modifying only the view of a tensor, all of sparse indexing
        operations produce copies.

    Args:
        x (SparseTensor): Sparse tensor to be sliced.
        *indices (None | slice | int | Tensor[bool | int, ...] | tuple[int, ...]):
            Slicing masks, one per dimension of input tensor.

    Returns:
        SparseTensor: Sliced sparse tensor.
    """
    # TODO: this whole slicing business needs a good refactor.
    # There should probably be one function that normalizes the input indices:
    # * Nones can be excluded.
    # * We can append missing indices as slice()
    # *
    # and computes the output shape from the input
    assert x.layout == torch.sparse_coo
    mask, new_shape = _build_index_selection_mask(x, *indices)
    selected_idx = x.indices()[:, mask]
    selected_val = x.values()[mask]

    # Pad missing indices:
    n_non_none_dims = 0
    for idx in indices:
        if idx is not None:
            n_non_none_dims += 1
    indices_norm = list(indices)
    if n_non_none_dims < x.ndim:
        for _ in range(n_non_none_dims):
            indices_norm.append(slice(None))

    # Compute shape and offset indices:
    reshaped_shape = []
    reshaped_idcs = []
    in_idx_dim = 0
    for dim, idx in enumerate(indices_norm):
        if isinstance(idx, tuple):
            idx = torch.tensor(idx, device=x.device)
        match idx:
            case None:
                reshaped_shape.append(1)
                reshaped_idcs.append(torch.zeros_like(selected_idx[0, :]))
                in_idx_dim -= 1
            case int():
                selected_idx[in_idx_dim, :] = 0
            case tuple() | torch.Tensor() | slice():
                reshaped_shape.append(new_shape[in_idx_dim])
                idx_map = torch.empty(
                    x.shape[in_idx_dim], dtype=torch.int64, device=x.device
                )
                idx_map[idx] = torch.arange(new_shape[in_idx_dim], device=x.device)
                selected_idx[in_idx_dim, :] = idx_map[selected_idx[in_idx_dim, :]]
                reshaped_idcs.append(selected_idx[in_idx_dim, :])
        in_idx_dim += 1
    if len(reshaped_idcs) == 0:
        reshaped_idx = torch.zeros([0, 1])
    else:
        reshaped_idx = torch.stack(reshaped_idcs)
    return coo_tensor(
        reshaped_idx,
        selected_val,
        size=reshaped_shape,
        check_invariants=False,
        is_coalesced=True,
    )


def fill_slice(
    x: SparseTensor, fill_value: float | int, *indices: slice | int | tuple[int, ...]
) -> SparseTensor:
    assert x.layout == torch.sparse_coo
    mask, _ = _build_index_selection_mask(x, *indices)
    selected_idx = x.indices()
    selected_val = x.values()
    selected_val[mask] = fill_value
    return coo_tensor(
        selected_idx,
        selected_val,
        size=x.shape,
        check_invariants=False,
        is_coalesced=True,
    )


def zero_slice(
    x: SparseTensor, *indices: slice | int | tuple[int, ...]
) -> SparseTensor:
    assert x.layout == torch.sparse_coo
    return fill_slice(x, 0, *indices)


def reshape(x: SparseTensor, *shape: int) -> SparseTensor:
    assert x.layout == torch.sparse_coo
    assert len(x.shape) == 2, x.shape[0] == x.shape[1]
    x = x.coalesce()
    indices = x.indices()
    values = x.values()
    new_indices = unravel_index(ravel_indices(indices, x.shape), shape)

    return coo_tensor(new_indices, values, size=[*shape], is_coalesced=True)


def repdiag(x: SparseTensor, n_reps: int) -> SparseTensor:
    assert len(x.shape) == 2, x.shape[0] == x.shape[1]
    x = x.coalesce()
    indices = x.indices()
    values = x.values()
    size = x.shape[0]

    return coo_tensor(
        torch.cat([indices + i * size for i in range(n_reps)], -1),
        torch.cat(n_reps * [values], -1),
        size=[n_reps * size, n_reps * size],
        is_coalesced=True,
    )


def cat(xs: Sequence[SparseTensor], dim=0) -> SparseTensor:
    assert isinstance(xs, Sequence)
    for x in xs:
        assert isinstance(x, torch.Tensor)
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

    return coo_tensor(new_indices, new_values, size=new_shape, is_coalesced=True)


def append(
    x: SparseTensor, indices: torch.Tensor, values: torch.Tensor
) -> SparseTensor:
    assert x.layout == torch.sparse_coo
    return coo_tensor(
        torch.cat([x.indices(), indices.to(x.device)], -1),
        torch.cat([x.values(), values.to(x.device)], -1),
        size=x.shape,
    ).coalesce()


def is_sparse_any(a: torch.Tensor):
    return a.is_sparse or a.is_sparse_csr


def mul_sparse_sparse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(
            "Sparse-sparse elementwise multiplication only supports same shape tensors."
        )
    if not a.is_coalesced():
        a = a.coalesce()
    a_idx, a_val = a.indices(), a.values()

    if not b.is_coalesced():
        b = b.coalesce()
    b_idx, b_val = b.indices(), b.values()

    a_mask, b_mask = isect_indices(a_idx, b_idx)

    return coo_tensor(
        a_idx[:, a_mask], a_val[a_mask] * b_val[b_mask], size=a.shape, is_coalesced=True
    )


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: csr matrices
    if not is_sparse_any(a) and not is_sparse_any(b):
        return a * b
    elif is_sparse_any(a) and is_sparse_any(b):
        return mul_sparse_sparse(a, b)
    elif b.is_sparse:
        a, b = b, a

    if not a.is_coalesced():
        a = a.coalesce()
    idx = a.indices()
    val = a.values()
    out_shape = torch.broadcast_shapes(a.shape, b.shape)

    b_idx = []
    for dim in range(a.dim()):
        b_idx.append(0 if b.shape[dim] == 1 else idx[dim])

    new_vals = val * b[tuple(b_idx)]
    return coo_tensor(idx, new_vals, size=out_shape).coalesce()


def matmul(
    a: SparseTensor | torch.Tensor, b: SparseTensor | torch.Tensor
) -> SparseTensor | torch.Tensor:
    swapped = False
    if not is_sparse_any(a) and not is_sparse_any(b):
        return a @ b
    elif is_sparse_any(a) and is_sparse_any(b):
        return torch.sparse.mm(a, b)
    elif is_sparse_any(b):
        swapped = True
        a, b = b.mT, a.mT
    if a.is_sparse and not a.is_sparse_csr:
        a = a.to_sparse_csr()

    if b.ndim == 1:
        result = torch.sparse.mm(a, b[..., None])[:, 0]
    elif a.ndim >= 2 and b.ndim >= 2:
        result = torch.sparse.mm(a, b)
    if swapped and result.ndim == 2:
        result = result.mT
    return result

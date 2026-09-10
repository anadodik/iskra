# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from collections.abc import Callable, Sequence
from functools import reduce
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np
import scipy.sparse
import torch

from iskra.profiling import profile_fn


def index_complement(n: int, idx: torch.Tensor) -> torch.Tensor:
    """Computes the complement of a set of linear indices.

    Given a vector of linear indices `idx` into some other vector with
    `n` elements, the function returns a vector of indices which selects
    all elements of the vector not selected by `idx`.

    Args:
        n (int): Total number of elements in a vector.
        idx (Tensor): Indices.

    Returns:
        Tensor: Vector of indices which select all elements not selected by `idx`.
    """
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
        a_idx (Tensor): First set of indices to compare.
        b_idx (Tensor): Second set of indices to compare.

    Returns:
        tuple[Tensor, Tensor]: Masks (one per input),
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

    Example:
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
    """Converts a linear index into COO indices.

    See Also:
        This function does the opposite of `ravel_indices()`. See for more information.

    Args:
        linear (Tensor[nnz]): Linear indices into a strided layout tensor.
        shape (torch.Size | tuple[int, ...]): Shape of the tensor.

    Returns:
        Tensor[dim, nnz]: Indices of a sparse tensor in COO format of same shape.
    """
    linear = linear.clone()
    idx = []
    for s in reversed(shape):
        idx.append(linear % s)
        linear = linear // s
    return torch.stack(tuple(reversed(idx)))


def is_sparse_any(x: torch.Tensor) -> bool:
    """Checks if tensor is either in sparse COO or CSR format.

    Args:
        x (torch.Tensor): Tensor to check.

    Returns:
        bool: Whether tensor is sparse or not.
    """
    return x.is_sparse or x.is_sparse_csr


def alias[T: torch.Tensor](x: T) -> T:
    """Creates an alias (view) of a tensor. Works on sparse tensors, unlike PyTorch's.

    The C++ function `alias()` is not implemented for sparse tensors, but we can simply
    implement a version of it from Python.
    This is useful for creating subclasses of `torch.Tensor` that have
    sparse storage; see `SparseTensor` implementation details.

    Warning:
        This function coalesces sparse COO tensors.

    Args:
        x (torch.Tensor): Tensor to alias.

    Returns:
        torch.Tensor: Alias to x.
    """
    if x.is_sparse:
        x = x.coalesce()
        x_alias = torch.sparse_coo_tensor(
            x.indices(),
            x.values(),
            x.shape,
            dtype=x.dtype,
            device=x.device,
            is_coalesced=x.is_coalesced(),
        )
        if isinstance(x, SparseTensor):
            x_alias.__class__ = SparseTensor
        return x_alias
    elif x.is_sparse_csr:
        x_alias = torch.sparse_csr_tensor(
            x.crow_indices(),
            x.col_indices(),
            x.values(),
            x.shape,
            dtype=x.dtype,
            device=x.device,
        )
        if isinstance(x, SparseTensor):
            x_alias.__class__ = SparseTensor
        return x_alias
    else:
        return x.view_as(x)


def _make_sparse_subclass(x: torch.Tensor) -> "SparseTensor":
    x = alias(x)
    x.__class__ = SparseTensor
    return x


class SparseTensor(torch.Tensor):
    """Sane PyTorch sparse tensor.

    A subclass of `torch.Tensor` with a sparse layout, either `torch.sparse_coo` or
    `torch.sparse_csr`.
    Unlike the default PyTorch sparse tensors, it offers many quality-of-life utilities
    which make working with sparse tensors a bareable experience in PyTorch, such as
    scalar/vector/matrix multiplications which just work and ensure your gradients
    remain sparse (which is somehow not the default in PyTorch), indexing, slicing,
    as well as small helpers here and there to patch missing functionality.

    Primarily, this class supports COO tensors; CSR tensor experience might be mixed.

    Warning:
        Constructing `SparseTensor` with COO values will automaticall call `coalesce()`
        for you, as this is required for `alias()` to work.

    Note:
        `SparseTensor` is not necessary to use the free-standing functions in this
        module. You should be able to use most free-standing functions with a normal
        tensor created via `torch.sparse_coo_tensor`. You should be careful with things
        like multiplications in that case (`@` will create dense gradients even if the
        operands are both sparse matrices).

    Important:
        You almost always want to use `sp.coo_tensor()` or `sp.csr_tensor()` to
        construct a `SparseTensor` object.

    Builds upon and extends existing PyTorch sparse tensors, and offers a (somewhat)
    sane interface and defaults.

    .. admonition:: Implementation Details

        Alright, I've spent a bunch of time digging through PyTorch's subclassing mess,
        so I am sharing what I learned here, hoping it is useful in the future.
        So, our goal is to create a subclass of `torch.Tensor` which has sparse storage
        under the hood (i.e., `torch.layout == torch.sparse_coo` or
        `torch.layout == torch.sparse_csr`).

        PyTorch's `_make_subclass()` is not what we need because it terminates autograd
        history, meaning, e.g., it won't propagate gradients to the values of a COO
        tensor if the values are autograd leafs. In essence, `_make_subclass()` makes
        a new leaf tensor with the same data as the input. This is not what we want!

        PyTorch's `as_subclass()` should be the fix, but fails on sparse tensors.
        Under the hood, it does a couple of things, see https://github.com/pytorch/pytorch/blob/ece45c392cb3ed8a958fbaea8eaf0fe5d812da6d/torch/csrc/autograd/python_variable.cpp#L610-L633
        First, it creates a view into the tensor, thus sharing memory but having a
        new tensor object. Second, it sets `__class__` to equal the subclass class.
        Lastly, it enables `__torch_dispatch__` on the subclass via
        `set_python_dispatch()`, but only if the subclass has a custom
        `__torch_dispatch__` defined.

        Sadly, the PyTorch does not implement aliasing for sparse tensors, but we can
        simply implement a Python alternative; see `alias()`. Likewiese, we set
        `tensor.__class__ = SparseTensor` manually from Python.
        The only thing we cannot do from Python is `__torch_dispatch__`. However,
        `__torch_dispatch__` is only needed if you wish to override low-level tensor
        behavior, such as behavior during autograd or CUDA kernel calls.
        In **our specific scenario**, we are not overriding `__torch_dispatch__`, so we
        do not really need to worry about `set_python_dispatch()`.
        However, if we were, we would probably need to do somehting akin to this:
        https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/negative_tensor.py
        This implementation seems to be more complex, so we will only migrate when
        this is absolutely necessary.
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
        if not is_sparse_any(tensor):
            raise TypeError(
                "SparseTensor does not accept dense tensors. "
                "Use SparseTensor.from_coo() or SparseTensor.from_csr() to "
                "construct from raw data, or convert first with .to_sparse()."
            )
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)

        if layout == "csr" and tensor.layout == torch.sparse_coo:
            tensor = tensor.to_sparse_csr()
        elif layout == "coo" and tensor.layout == torch.sparse_csr:
            tensor = tensor.to_sparse_coo()

        tensor = _make_sparse_subclass(tensor)
        if requires_grad:
            tensor.requires_grad_(requires_grad)
        return tensor

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
        """Constructs a COO `SparseTensor` from indices and values.

        This method wraps `torch.sparse_coo_tensor()` and does plumbing to convert
        it to a `SparseTensor`. The argument and return descriptions reuse
        the original torch descriptions. See that function for more details.

        See Also:
            `torch.sparse_coo_tensor()`, `coo_tensor()`.

        Args:
            indices (Tensor[Int64, [Dim, nnz]] | tuple[Tensor[Int64, [nnz]], ...] | list[Tensor[Int64, [nnz]]]):
                Initial data for the tensor. Will be cast to a
                `Tensor[Int64, [Dim, nzz]]` internally. If `indices` is a sequence
                of 1D tensors, we stack them into a index tensor. The indices are the
                coordinates of the non-zero values in the matrix, and thus
                should be two-dimensional where the first dimension is the
                number of tensor dimensions and the second dimension is the
                number of non-zero values. A sequence of per-dimension index
                tensors is also accepted and stacked along dimension 0.
            values (Tensor[DType, [nnz]]): Initial values for the tensor.
            size (list[int] | tuple[int, ...] | None): Size of the sparse
                tensor. If not provided the size will be inferred as the
                minimum size big enough to hold all non-zero elements.
            dtype (torch.dtype): the desired data type of returned
                tensor. Default: if None, infers data type from `values`.
            device (torch.device): the desired device of returned
                tensor. Default: if None, uses the device of the input tensors.
            requires_grad (bool): If ``True``, the returned `SparseTensor` is
                an autograd leaf. Default: ``False``.
            check_invariants (bool): If sparse tensor invariants are checked.
                Default: ``False``.
            is_coalesced (bool): When ``True``, the caller is responsible for
                providing tensor indices that correspond to a coalesced tensor.
                If the `check_invariants` flag is False, no error will be
                raised if the prerequisites are not met and this will lead to
                silently incorrect results. To force coalescion please use
                :meth:`~torch.Tensor.coalesce` on the resulting Tensor.
                Default: ``True``.

        Returns:
            SparseTensor[Float, [*Bs, N, M, *Ds]]: Sparse tensor in COO layout.
        """
        if isinstance(indices, Sequence) and not isinstance(indices, torch.Tensor):
            indices = torch.stack(indices, 0)
        # We pass requires_grad to the SparseTensor __new__ so that it is the leaf.
        t = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            dtype=dtype,
            device=device,
            requires_grad=False,
            check_invariants=check_invariants,
            is_coalesced=is_coalesced,
        )
        t_cls = cls(t, layout="coo", requires_grad=requires_grad)
        return t_cls

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
        """Constructs a CSR `SparseTensor` from compressed-row data.

        This method wraps `torch.sparse_csr_tensor()` and does plumbing to convert
        it to a `SparseTensor`. The argument and return descriptions reuse
        the original torch descriptions. See that function for more details.

        See Also:
            `torch.sparse_csr_tensor()`, `csr_tensor()`.

        Args:
            crow_indices (Tensor[Int64, [*Bs, N + 1]]): (B+1)-dimensional array
                of size ``(*batchsize, nrows + 1)``. The last element of each
                batch is the number of non-zeros. This tensor encodes the index
                in values and col_indices depending on where the given row
                starts. Each successive number in the tensor subtracted by the
                number before it denotes the number of elements in a given row.
            col_indices (Tensor[Int64, [*Bs, nnz]]): Column co-ordinates of each
                element in values. (B+1)-dimensional tensor with the same length
                as values.
            values (Tensor[Float, [*Bs, nnz, *Ds]]): Initial values for the
                tensor. Represents a (1+K)-dimensional tensor where ``K`` is the
                number of dense dimensions.
            size (list[int] | tuple[int, ...] | None): Size of the sparse
                tensor: ``(*batchsize, nrows, ncols, *densesize)``. If not
                provided, the size will be inferred as the minimum size big
                enough to hold all non-zero elements.
            dtype (torch.dtype): the desired data type of returned
                tensor. Default: if None, infers data type from `values`.
            device (torch.device): the desired device of returned
                tensor. Default: if None, uses the device of the input tensors.
            requires_grad (bool): If ``True``, the returned `SparseTensor` is
                an autograd leaf. Default: ``False``.

        Returns:
            SparseTensor[Float, [*Bs, N, M, *Ds]]: Sparse tensor in CSR layout.
        """
        # We pass requires_grad to the SparseTensor __new__ so that it is the leaf.
        t = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        t_cls = cls(t, layout="csr", requires_grad=requires_grad)
        return t_cls

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
        square_funcs = {
            torch.square,
            torch.Tensor.square,
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
                if is_sparse_any(a) and is_sparse_any(b):
                    return _make_sparse_subclass(ret)
                else:
                    return ret
        elif func in mul_funcs:
            a, b = args
            with torch._C.DisableTorchFunctionSubclass():
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    ret = mul(a, b)
                else:
                    ret = func(*args, **kwargs)
        elif func in square_funcs:
            ret = square(*args)
        else:
            with torch._C.DisableTorchFunctionSubclass():
                ret = func(*args, **kwargs)

        if (
            isinstance(ret, torch.Tensor)
            and not isinstance(ret, cls)
            and func not in dense_funcs
        ):
            ret = _make_sparse_subclass(ret)
        return ret

    @overload
    def __matmul__(self, other: "SparseTensor") -> "SparseTensor": ...

    @overload
    def __matmul__(self, other: torch.Tensor) -> "SparseTensor": ...

    def __matmul__(self, other: "SparseTensor | torch.Tensor") -> "SparseTensor":
        """Matrix multiplication which works with sparse tensors.

        Allows the user to write `a @ b` with sparse tensors and still get the
        expected behavior (like sparse gradients to sparse tensors).

        Important:
            Wrapper around `matmul()`.

        Args:
            other (SparseTensor | torch.Tensor): Left-multiply with this tensor.

        Returns:
            SparseTensor | torch.Tensor: Multiplied result tensor.
        """
        return matmul(self, other)

    @overload
    def __rmatmul__(self, other: "SparseTensor") -> "SparseTensor": ...

    @overload
    def __rmatmul__(self, other: torch.Tensor) -> "SparseTensor": ...

    def __rmatmul__(self, other: "SparseTensor | torch.Tensor") -> "SparseTensor":
        """Matrix multiplication which works with sparse tensors.

        Allows the user to write `a @ b` with sparse tensors and still get the
        expected behavior (like sparse gradients to sparse tensors).

        Important:
            Wrapper around `matmul()`.

        Args:
            other (SparseTensor | torch.Tensor): Right-multiply with this tensor.

        Returns:
            SparseTensor | torch.Tensor: Multiplied result tensor.
        """
        return matmul(self, other)

    @overload
    def __mul__(self, other: "SparseTensor") -> "SparseTensor": ...

    @overload
    def __mul__(self, other: torch.Tensor) -> "SparseTensor": ...

    @overload
    def __mul__(self, other: Number) -> "SparseTensor": ...

    def __mul__(self, other: "SparseTensor | torch.Tensor | Number") -> "SparseTensor":
        """Elementwise multiplication which works with sparse tensors.

        Important:
            Wrapper around `mul()`.

        Args:
            other (SparseTensor | torch.Tensor): Multiply with this tensor.

        Returns:
            SparseTensor | torch.Tensor: Multiplied result tensor.
        """
        if isinstance(other, torch.Tensor):
            return mul(self, other)
        else:
            return super().__mul__(other)

    @overload
    def __rmul__(self, other: "SparseTensor") -> "SparseTensor": ...

    @overload
    def __rmul__(self, other: torch.Tensor) -> "SparseTensor": ...

    @overload
    def __rmul__(self, other: Number) -> "SparseTensor": ...

    def __rmul__(self, other: "SparseTensor | torch.Tensor | Number") -> "SparseTensor":
        """Elementwise multiplication which works with sparse tensors.

        Important:
            Wrapper around `mul()`.

        Args:
            other (SparseTensor | torch.Tensor): Multiply with this tensor.

        Returns:
            SparseTensor | torch.Tensor: Multiplied result tensor.
        """
        if isinstance(other, torch.Tensor):
            return mul(other, self)
        else:
            return super().__rmul__(other)

    def reshape(self, *shape: int) -> "SparseTensor":
        """Reshapes the tensor into a specified shape.

        Important:
            Wrapper around `reshape()`.

        Args:
            *shape (int): Dimensions for the resulting tensor

        Returns:
            SparseTensor: Reshaped tensor with a view into the same data.
        """
        return reshape(self, *shape)

    def scipy(self) -> scipy.sparse.sparray:
        """Constructs a SciPy tensor with the same data (detaches autodiff graph).

        Important:
            Wrapper around `to_scipy()`.

        Returns:
            scipy.sparse.coo_array: SciPy tensor with the same data.
        """
        return to_scipy(self)

    def torch_tensor(self) -> torch.Tensor:
        result: SparseTensor = alias(self)
        result.__class__ = torch.Tensor
        return result

    def __getitem__(self, index: Any) -> "SparseTensor":
        """Slices the sparse tensor.

        Important:
            Wrapper around `get_slice()`.

        Args:
            index (Any): Slicing indices. See `get_slice()` for more information.

        Returns:
            SparseTensor: Sliced sparse tensor.
        """
        if isinstance(index, tuple):
            return get_slice(self, *index)
        else:
            return get_slice(self, index)

    def __setitem__(self, index, value):
        raise NotImplementedError("Setting sparse tensor elements not supported yet.")

    def square(self) -> "SparseTensor":
        """Elementwise square of the matrix.

        Important:
            Wrapper around `iskra.sparse.square()`.

        Returns:
            SparseTensor: Matrix with squared entries.
        """
        return square(self)


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
    """Constructs a COO `SparseTensor` from indices and values.

    Important:
        This function is a thin wrapper around `SparseTensor.from_coo()`.
    """
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
    """Constructs a CSR `SparseTensor` from compressed-row data.

    Important:
        This function is a thin wrapper around `SparseTensor.from_csr()`.
    """
    return SparseTensor.from_csr(
        crow_indices,
        col_indices,
        values,
        size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
    size: torch.Size | list[int] | tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> SparseTensor:
    """Constructs a sparse COO zero matrix.

    Args:
        size (list[int] | tuple[int, ...] | None): Size of the sparse tensor.
        dtype (torch.dtype): Matrix data type. Defaults to `torch.float32`.
        device (str | torch.device, optional): Device for the matrix. Defaults to "cpu".

    Returns:
        SparseTensor[DType, size]: Sparse `n`-by-`n` identity matrix in COO format.
    """
    return coo_tensor(
        torch.empty([len(size), 0], dtype=torch.int64, device=device),
        torch.empty([0], dtype=dtype, device=device),
        size=size,
    )


def eye(
    n: int, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"
) -> SparseTensor:
    """Constructs a sparse COO identity matrix.

    Args:
        n (int): Size (number of rows or columns) of the matrix.
        dtype (torch.dtype): Matrix data type. Defaults to `torch.float32`.
        device (str | torch.device, optional): Device for the matrix. Defaults to "cpu".

    Returns:
        SparseTensor[DType, [n, n]]: Sparse `n`-by-`n` identity matrix in COO format.
    """
    idx = torch.arange(n, device=device)
    values = torch.ones([n], dtype=dtype, device=device)
    return coo_tensor((idx, idx), values, size=[n, n], is_coalesced=True)


def diag(values: torch.Tensor | SparseTensor) -> SparseTensor:
    """Constructs a sparse COO diagonal matrix from vector of diagonal elements.

    Args:
        values (Tensor[DType, [N]] | SparseTensor[DType, [N]]): Vector with values that
            will be placed along the diagonal. Allowed to be either dense or sparse.

    Returns:
        SparseTensor[DType, [N, N]]: Sparse `N`-by-`N` identity matrix in COO format.
    """
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


def get_diag(mat: SparseTensor) -> SparseTensor:
    """Extracts diagonal from sparse COO matrix.

    This function returns the diagonal as a sparse matrix. While this may seem
    counter-intuitive at first, the reasoning behind the decision is twofold.
    First, the diagonal itself can be very sparse and making the decision to
    densify could potentially blow up memory usage. Second, sparse COO tensors
    can have more than one sparse dimension. While `iskra` is not designed with
    this usecase in mind, this function is written defensively to potentially
    support this usecase.

    Args:
        mat (SparseTensor[DType, [N, N]]): Matrix to extract diagonal from.

    Returns:
        SparseTensor[DType, [N]]: Diagonal of the matrix as a sparse COO tensor.
    """
    assert mat.ndim >= 2
    assert mat.shape[-1] == mat.shape[-2]
    idcs = mat.indices()
    values = mat.values()
    diag_mask = idcs[-1] == idcs[-2]
    return coo_tensor(
        idcs[:-1, diag_mask], values[diag_mask], size=mat.shape[:-1]
    ).coalesce()


def inv_diag(mat: SparseTensor) -> SparseTensor:
    """Inverts elements of a diagonal COO matrix.

    The function densifies the diagonal of a sparse matrix and divisions by the
    zero non-diagonal elements become NaNs.
    This default behavior is subject to change given a good real-world usecase
    that requires inverting only the non-zero elements: please leave an issue if
    you have one.

    Args:
        mat (SparseTensor): Diagonal COO matrix.

    Returns:
        SparseTensor: COO matrix of the same shape, but with
    """
    return diag(1.0 / get_diag(mat).to_dense())


def from_scipy(
    x: scipy.sparse.coo_array | scipy.sparse.csr_array,
    device: torch.device | str = "cpu",
) -> SparseTensor:
    """Converts SciPy sparse COO/CSR arrays into PyTorch sparse tensors.

    Args:
        x (scipy.sparse.coo_array | scipy.sparse.csr_array): SciPy sparse array to
            be converted.
        device (torch.device | str): Specifies the device on which to
            construct the tensor. Defaults to "cpu".

    Raises:
        ValueError: Throws an exception if the input array is neither COO nor CSR.

    Returns:
        SparseTensor: PyTorch tensor with the same data.
    """
    if x.format == "coo":
        if TYPE_CHECKING:
            assert isinstance(x, scipy.sparse.coo_array)

        row = torch.tensor(x.row, device=device)
        col = torch.tensor(x.col, device=device)
        data = torch.tensor(x.data, device=device)
        x_torch = coo_tensor((row, col), data, size=x.shape)
    elif x.format == "csr":
        if TYPE_CHECKING:
            assert isinstance(x, scipy.sparse.csr_array)

        crow = torch.tensor(x.indptr, device=device)
        col = torch.tensor(x.indices, device=device)
        data = torch.tensor(x.data, device=device)
        x_torch = csr_tensor(crow, col, data, size=x.shape)
    else:
        raise ValueError(
            f"Passed SciPy array with format={x.format}. "
            "SciPy conversion supported for COO and CSR only."
        )
    return x_torch


def to_scipy(
    x: SparseTensor | torch.Tensor,
) -> scipy.sparse.coo_array | scipy.sparse.csr_array:
    """Converts PyTorch sprase COO/CSR tensors into SciPy sparse arrays.

    If the input is on the CPU and does not require gradients, we will return a
    view into the data. If the data is on the GPU or requires gradients, the tensor
    will be cloned and a view into that tensor will be returned.

    Since SciPy does not perform type erasure when it comes to sparse layouts, this
    function can only return the type `scipy.sparse.coo_array | scipy.sparse.csr_array`,
    which might make your type checker complain if you call any functions specific to
    either layout. If this happens, you can always follow this function up with, e.g.,
    `assert isinstance(y, scipy.sparse.csr_array)` to silence the type checker.

    Args:
        x (SparseTensor | torch.Tensor): COO or CSR tensor to be converted.

    Raises:
        ValueError: Throws an exception if the input tensor is neither COO nor CSR.

    Returns:
        scipy.sparse.coo_array | scipy.sparse.csr_array: SciPy array with the same data.
    """
    if x.layout == torch.sparse_coo:
        x = x.detach().coalesce()
        if x.values().is_cuda:
            data = x.values().cpu().numpy()
            idcs = x.indices().cpu().numpy()
        else:
            # if on CPU, .numpy creates view
            data = x.values().numpy()
            idcs = x.indices().numpy()
        x_scipy = scipy.sparse.coo_array((data, (idcs[0], idcs[1])), shape=x.shape)
    elif x.layout == torch.sparse_csr:
        x = x.detach()
        if x.values().is_cuda:
            data = x.values().cpu().numpy()
            indptr = x.crow_indices().cpu().numpy().astype(np.int64)
            idcs = x.col_indices().cpu().numpy().astype(np.int64)
        else:
            # if on CPU, .numpy creates view
            data = x.values().numpy()
            indptr = x.crow_indices().numpy()
            idcs = x.col_indices().numpy()
        x_scipy = scipy.sparse.csr_array((data, idcs, indptr), shape=x.shape)
    else:
        raise ValueError(
            f"Passed tensor with layout={x.layout}. "
            "SciPy conversion supported for COO and CSR only."
        )
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
                x_idcs = x.indices()[dim]
                appears_in_idx = torch.zeros(
                    x.shape[dim], dtype=torch.bool, device=idx.device
                )
                appears_in_idx[idx] = True
                mask &= appears_in_idx[x_idcs]
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

    Warning:
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
    # The following is faster than indices()[:, mask] and values()[mask] because
    # masking with a boolean tensor calls nonzero under the hood.
    mask_idcs = mask.nonzero().squeeze(1)
    selected_idx = x.indices().index_select(1, mask_idcs)
    selected_val = x.values().index_select(0, mask_idcs)

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
    x: SparseTensor, fill_value: float, *indices: _INDEX_TYPE
) -> SparseTensor:
    """Sets all nonzero entries in a slice of a sparse COO tensor to a chosen value.

    This function leaves zero values unmodified. If you want to modify the zero
    values, use `iskra.sparse.append()`.
    See `iskra.sparse.get_slice()` to see how the slicing is performed.

    Args:
        x (SparseTensor): Sparse tensor to fill.
        fill_value (float | int): The value for the nonzero entries in the slice.
        *indices (None | slice | int | Tensor[bool | int, ...] | tuple[int, ...]):
            Slicing masks, one per dimension of input tensor.

    Returns:
        SparseTensor: Tensor with modified nonzero entries within the slice.
    """
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


def zero_slice(x: SparseTensor, *indices: _INDEX_TYPE) -> SparseTensor:
    # TODO: this function should probably be getting rid of nnz entries!
    assert x.layout == torch.sparse_coo
    return fill_slice(x, 0, *indices)


def reshape(x: SparseTensor, *shape: int) -> SparseTensor:
    """Returns a sparse COO tensor with the same data, but with the specified shape.

    Warn:
        Unlike PyTorch's default reshape, we do not support $-1$ as one of the
        dimensions.

    Args:
        x (SparseTensor): Tensor to be reshaped.
        *shape (int): The new shape.

    Returns:
        SparseTensor: Reshaped tensor.
    """
    assert x.layout == torch.sparse_coo
    assert len(x.shape) == 2, x.shape[0] == x.shape[1]
    x = x.coalesce()
    indices = x.indices()
    values = x.values()
    new_indices = unravel_index(ravel_indices(indices, x.shape), shape)

    return coo_tensor(new_indices, values, size=[*shape], is_coalesced=True)


def repdiag(x: SparseTensor, n_reps: int) -> SparseTensor:
    """Repeats a sparse COO matrix along a diagonal to make a block-diagonal matrix.

    Args:
        x (SparseTensor): Matrix to repeat.
        n_reps (int): Number of repetitions.

    Returns:
        SparseTensor: Block-diagonal matrix with `x` embedded along the diagonal
            `n_reps` times.
    """
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


def cat_diag(xs: Sequence[SparseTensor]) -> SparseTensor:
    """Concatenates sparse COO matrice along a diagonal to make a block-diagonal matrix.

    Args:
        xs (Sequence[SparseTensor]): Sequences of matrices to cocnatenate.

    Returns:
        SparseTensor: Block-diagonal matrix with `xs` embedded along the diagonal.
    """
    assert isinstance(xs, Sequence)
    assert len(xs) > 0
    for x in xs:
        assert isinstance(x, torch.Tensor)

    block_idcs_list = []
    block_values_list = []
    total_shape = None

    for x in xs:
        x = x.coalesce()
        indices = x.indices()
        values = x.values()
        shape = torch.tensor(x.shape, device=x.device)
        if total_shape is None:
            block_idcs_list.append(indices)
            block_values_list.append(values)
            total_shape = shape
        else:
            assert total_shape.nelement() == shape.nelement()
            block_idcs_list.append(indices + total_shape[:, None])
            block_values_list.append(values)
            total_shape += shape

    assert total_shape is not None

    return coo_tensor(
        torch.cat(block_idcs_list, -1),
        torch.cat(block_values_list, -1),
        size=total_shape.cpu().numpy().tolist(),
        is_coalesced=True,
    )


def cat(xs: Sequence[SparseTensor], dim: int = 0) -> SparseTensor:
    """Concatenates sparse COO matrices along a dimension.

    Args:
        xs (Sequence[SparseTensor]): Sequences of matrices to cocnatenate.
            Matrix shapes must match in all dimensions except `dim`.
        dim (int, optional): Dimension to concatenate in. Defaults to 0.

    Raises:
        ValueError: Throws an exception if shapes do not match in all dimensions
            except `dim.`

    Returns:
        SparseTensor: Concatenated sparse COO matrix.
    """
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
    """Appends new values to an existing COO tensor.

    In the case of an attempt to add values to already existing indices, the values are
    added together.

    Args:
        x (SparseTensor): Sparse COO tensor to which we will append values.
        indices (torch.Tensor): Indices for the new values.
        values (torch.Tensor): The new values.

    Returns:
        SparseTensor: Sparse COO tensor containing both the old and new values.
    """
    assert x.layout == torch.sparse_coo
    return coo_tensor(
        torch.cat([x.indices(), indices.to(x.device)], -1),
        torch.cat([x.values(), values.to(x.device)], -1),
        size=x.shape,
    ).coalesce()


@overload
def mul_sparse_sparse(a: SparseTensor, b: SparseTensor) -> SparseTensor: ...


@overload
def mul_sparse_sparse(a: torch.Tensor, b: SparseTensor) -> SparseTensor: ...


@overload
def mul_sparse_sparse(a: SparseTensor, b: torch.Tensor) -> SparseTensor: ...


@overload
def mul_sparse_sparse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def mul_sparse_sparse(
    a: SparseTensor | torch.Tensor, b: SparseTensor | torch.Tensor
) -> SparseTensor | torch.Tensor:
    """Elementwise product of two sparse COO tensors.

    Args:
        a (SparseTensor | torch.Tensor): First tensor to multiply.
        b (SparseTensor | torch.Tensor): Second tensor to multiply.

    Raises:
        ValueError: Two tensors must have the same shape.

    Returns:
        SparseTensor | torch.Tensor: Elementwise product of the two tensors.
    """
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


@overload
def mul(a: SparseTensor, b: SparseTensor) -> SparseTensor: ...


@overload
def mul(a: torch.Tensor, b: SparseTensor) -> SparseTensor: ...


@overload
def mul(a: SparseTensor, b: torch.Tensor) -> SparseTensor: ...


@overload
def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def mul(
    a: SparseTensor | torch.Tensor, b: SparseTensor | torch.Tensor
) -> SparseTensor | torch.Tensor:
    """Elementwise product of two tensors with support for sparse COO tensors.

    Args:
        a (SparseTensor | torch.Tensor): First tensor to multiply.
        b (SparseTensor | torch.Tensor): Second tensor to multiply.

    Raises:
        ValueError: If both tensors are sparse, they must have the same shape.

    Returns:
        SparseTensor | torch.Tensor: Elementwise product of the two tensors.
    """
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
    b = b.expand(out_shape)

    b_idx = []
    for dim in range(a.dim()):
        b_idx.append(0 if b.shape[dim] == 1 else idx[dim])

    new_vals = val * b[tuple(b_idx)]
    return coo_tensor(idx, new_vals, size=out_shape).coalesce()


@overload
def matmul(a: SparseTensor, b: SparseTensor) -> SparseTensor: ...


@overload
def matmul(a: torch.Tensor, b: SparseTensor) -> SparseTensor: ...


@overload
def matmul(a: SparseTensor, b: torch.Tensor) -> SparseTensor: ...


@overload
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def matmul(
    a: SparseTensor | torch.Tensor, b: SparseTensor | torch.Tensor
) -> SparseTensor | torch.Tensor:
    """Matrix-matrix product with support for sparse COO tensors.

    Unlike PyTorch's default, differentiating through this function produces sparse
    gradients through the sparse inputs. `SparseTensor.__matmul__` wraps around this
    function, meaning that, unlike PyTorch, `a @ b` is not a foot-gun in `iskra`.

    Warn:
        In case of a sparse-dense product, the sparse matrix is internally
        converted to the CSR format. This limitation stems from PyTorch's implementation
        of sparse COO matrices, which would produce dense gradients by default.
        You might want to consider converting it to CSR yourself before calling matmul.

    Args:
        a (SparseTensor | torch.Tensor): First tensor to be multiplied.
        b (SparseTensor | torch.Tensor): Second tensor to be multiplied.

    Returns:
        SparseTensor | torch.Tensor: Product `a @ b`.
    """
    # TODO: Differentiating through this function produces sparse gradients
    # if the input tensor is sparse.
    if not is_sparse_any(a) and not is_sparse_any(b):
        result = a @ b
    elif is_sparse_any(a) and is_sparse_any(b):
        with torch._C.DisableTorchFunctionSubclass():
            result = torch.sparse.mm(a, b)
    else:
        swapped = False
        if not is_sparse_any(a) and is_sparse_any(b):
            swapped = True
            a, b = b.mT, a.mT
        if a.is_sparse and not a.is_sparse_csr:
            a = a.to_sparse_csr()
        if b.ndim == 1:
            with torch._C.DisableTorchFunctionSubclass():
                result = torch.sparse.mm(a, b[..., None])[:, 0]
        elif a.ndim >= 2 and b.ndim >= 2:
            with torch._C.DisableTorchFunctionSubclass():
                result = torch.sparse.mm(a, b)
        if swapped and result.ndim == 2:
            result = result.mT

    if (
        isinstance(result, torch.Tensor)
        and not isinstance(result, SparseTensor)
        and is_sparse_any(result)
        and (isinstance(a, SparseTensor) or isinstance(b, SparseTensor))
    ):
        # ret_sparse = torch.Tensor._make_subclass(
        #     SparseTensor, result, result.requires_grad
        # )
        # ret_sparse._tensor = result
        # result = ret_sparse
        result = _make_sparse_subclass(result)
    return result


def square(x: SparseTensor) -> SparseTensor:
    """Elementwise square of sparse tensors.

    Patches a lack of support for this function in PyTorch for CSR tensors.

    Args:
        x (SparseTensor): Sparse tensor to square.

    Returns:
        SparseTensor: Sparse tensor with each entry squared.
    """
    if x.is_sparse_csr:
        return csr_tensor(
            x.crow_indices(),
            x.col_indices(),
            x.values() ** 2,
            x.shape,
            dtype=x.dtype,
            device=x.device,
        )
    else:
        with torch._C.DisableTorchFunctionSubclass():
            return torch.Tensor.square(x)

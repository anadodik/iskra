# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.


import torch


def atleast_nd(n: int, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Expands tensors to n-dimensions by prepeneding singular dimensions.

    The behavior is quite similar to torch.atleast_3d, except that unlike
    torch.atleast_3d, this function always _prepends_ singular dimensions.

    Returns:
        tuple[torch.Tensor]: Tensors which have expanded to have n dimensions.
    """
    prepended_ones = tuple((1,) * max(n - t.ndim, 0) for t in tensors)
    expanded = tuple(
        tensors[t_i].expand(*(prepended_ones[t_i] + tensors[t_i].shape))
        for t_i in range(len(tensors))
    )
    return expanded


def broadcast_tensors(*x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Typed wrapper around torch.broadcast_tensors.

    For more information see documentation of torch.broadcast_tensors.

    Returns:
        tuple[torch.Tensor]: Tensors which have been broadcast to the same shape.
    """
    return torch.broadcast_tensors(*x)  # type: ignore


def point_simplex_broadcast(
    x: torch.Tensor, simplex: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != simplex.ndim - 1:
        raise ValueError(
            "x.ndim != simplex.ndim - 1 "
            f"({x.ndim} != {simplex.ndim} - 1 = {simplex.ndim - 1})"
        )
    if x.shape[-1] != simplex.shape[-1]:
        raise ValueError(
            "Last dimensions of point and simplex do not match: "
            f"{x.shape[-1]} != {simplex.shape[-1]}."
        )
    shape = torch.broadcast_shapes(x.shape, simplex[..., 0, :].shape)  # type: ignore
    x = x.expand(*shape)
    simplex = simplex.expand(*(shape[:-1] + (-1,) + shape[-1:]))
    return x, simplex

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch


def coordinate_system(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build frame based on normal.

    Code inspired by:
    https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/vector.h
    """
    sign = torch.where(n[..., -1] <= -0.0, -1, 1)
    a = -torch.reciprocal(sign + n[..., -1])
    b = n[..., 0] * n[..., 1] * a

    tangent = torch.stack(
        [
            sign * n[..., 0] * n[..., 0] * a + 1,
            sign * b,
            -sign * n[..., 0],
        ],
        -1,
    )

    binormal = torch.stack(
        [b, n[..., 1] * n[..., 1] * a + sign, -n[..., 1]],
        -1,
    )
    return tangent, binormal

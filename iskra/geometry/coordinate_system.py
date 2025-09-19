# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch


def normal_coordinate_system(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build frame (tangent and binormal) from only a normal.

    Code based on:
    https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/vector.h


    Args:
        n (Tensor[Float, [Bs, 3]]): Normals vectors.

    Returns:
        Tensor[Float, [Bs, 3]]: Tangent vectors.
        Tensor[Float, [Bs, 3]]: Binormal vectors.
    """
    sign = torch.where(n[..., -1] <= -0.0, -1, 1)
    a = -torch.reciprocal(sign + n[..., -1])
    b = n[..., 0] * n[..., 1] * a

    tangent = torch.stack(
        [sign * n[..., 0] * n[..., 0] * a + 1, sign * b, -sign * n[..., 0]], -1
    )

    binormal = torch.stack([b, n[..., 1] * n[..., 1] * a + sign, -n[..., 1]], -1)
    return tangent, binormal


def triangle_coordinate_system(
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build frame (normal, tangent and binormal) from triangles.

    Constructs a 3D coordinate frame for triangles. It treats vertex 0 as the origin,
    normal is the normalized cross product $e_{01} \times e_{02}$,
    tangent is the normalized edge $e_{01}$, and binormal is normal-cross-tangent.

    Args:
        triangles (Tensor[Float, [Bs, 3, 3]]): Triangles

    Returns:
        Tensor[Float, [Bs, 3]]: Normal vectors.
        Tensor[Float, [Bs, 3]]: Tangent vectors.
        Tensor[Float, [Bs, 3]]: Binormal vectors.
    """
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    n = torch.nn.functional.normalize(
        torch.cross(edge_vecs[..., 0, :], edge_vecs[..., 1, :], dim=-1), p=2, dim=-1
    )
    t = torch.nn.functional.normalize(edge_vecs[..., 0, :], p=2, dim=-1)
    b = torch.nn.functional.normalize(torch.cross(n, t, dim=-1), p=2, dim=-1)
    return n, t, b

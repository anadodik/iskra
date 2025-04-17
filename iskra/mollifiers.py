# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import numpy as np
import torch

from iskra.geometry import barycentric_interpolate, point_dist
from iskra.geometry.broadcast import atleast_nd, point_simplex_broadcast


def hermite_cutoff(x: torch.Tensor, degree: int = 1) -> torch.Tensor:
    if degree == 0:
        cx = x
    elif degree == 1:
        cx = x + x**2 - x**3
    elif degree == 2:
        cx = x + 4.0 * x**3 - 7.0 * x**4 + 3.0 * x**5
    elif degree == 3:
        cx = x + 15.0 * x**4 - 39.0 * x**5 + 34.0 * x**6 - 10.0 * x**7
    elif degree == 4:
        cx = x + 56.0 * x**5 - 196.0 * x**6
        cx = cx + 260.0 * x**7 - 155.0 * x**8 + 35.0 * x**9
    elif degree == 5:
        cx = x - (-1 + x) * (x**6) * (
            210 + x * (-720 + 7 * x * (135 + 2 * x * (-40 + 9 * x)))
        )
    else:
        raise NotImplementedError(f"Degree {degree} hermite interpolant not supported.")
    return torch.where(x < 0.0, x, torch.where(x >= 1.0, 1.0, cx))


def soft_clamp(
    x: torch.Tensor, sigma: float = 0.05, left: bool = True, right: bool = True
) -> torch.Tensor:
    if sigma == 0.0 or (not left and not right):
        x = torch.clip(x, 0.0, 1.0)
    else:
        cx = 0.5 * (
            (
                (
                    -torch.exp(-((-1 + x) ** 2) / (2.0 * sigma**2))
                    + torch.exp(-(x**2) / (2.0 * sigma**2))
                )
                * np.sqrt(2.0 / np.pi)
                + np.sqrt(1.0 / sigma**2)
            )
            * sigma
            - (-1.0 + x) * torch.special.erf((-1.0 + x) / (np.sqrt(2.0) * sigma))
            + x * torch.special.erf(x / (np.sqrt(2.0) * sigma))
        )
        if left and right:
            x = cx
        elif left:
            x = torch.where(x < 0.5, cx, torch.clip(x, 0.0, 1.0))
        elif right:
            x = torch.where(x > 0.5, cx, torch.clip(x, 0.0, 1.0))
    return x


def smoothstep(x: torch.Tensor, start: float = 0.0, end: float = 1.0) -> torch.Tensor:
    x_scaled: torch.Tensor = torch.clamp((x - start) / (end - start), min=0.0, max=1.0)
    result: torch.Tensor = (-2 * x_scaled + 3) * (x_scaled**2)
    return result


def smootherstep(x: torch.Tensor, start: float = 0.0, end: float = 1.0) -> torch.Tensor:
    x_scaled: torch.Tensor = torch.clamp((x - start) / (end - start), min=0.0, max=1.0)
    result: torch.Tensor = (x_scaled**3) * (x_scaled * (x_scaled * 6.0 - 15.0) + 10.0)
    return result


def soft_edge_project(
    x: torch.Tensor, edges: torch.Tensor, sigma: float
) -> torch.Tensor:
    x, edges = point_simplex_broadcast(x, edges)
    origin = edges[..., 0, :]
    edge_vectors = edges[..., 1, :] - edges[..., 0, :]
    length = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
    edge_vectors = edge_vectors / length

    t = torch.linalg.vecdot((x - origin) / length, edge_vectors)
    t = torch.clamp(t, sigma / length[..., 0], 1 - sigma / length[..., 0])
    # t = soft_clamp(t, sigma / length[..., 0])
    bary = torch.stack([1 - t, t], -1)
    return barycentric_interpolate(edges, bary)


def soft_point_edge_dist(
    x: torch.Tensor, edges: torch.Tensor, sigma: float
) -> torch.Tensor:
    return point_dist(x, soft_edge_project(x, edges, sigma=sigma))


def soft_point_edge_dist_matrix(
    x: torch.Tensor, edges: torch.Tensor, sigma: float = 2e-1
) -> torch.Tensor:
    (x,) = atleast_nd(2, x)
    (edges,) = atleast_nd(3, edges)
    x, edges = x[..., :, None, :], edges[..., None, :, :, :]
    return soft_point_edge_dist(x, edges, sigma=sigma)


def bump(t: torch.Tensor, start: float = 0.0, end: float = 1.0) -> torch.Tensor:
    t = (t - start) / (end - start)
    t = torch.where(t > 0, torch.where(t < 1.0, t, 1.0), 0.0)
    f = torch.sigmoid((2 * t - 1) / (t * (1 - t) + 1e-7))
    return torch.where(t > 0, torch.where(t < 1.0, f, 1.0), 0.0)

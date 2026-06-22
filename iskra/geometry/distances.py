# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.


import torch
from torch import linalg

from iskra.geometry.barycentric import (
    barycentric_interpolate,
    tetrahedron_barycentric_coordinates,
    triangle_barycentric_coordinates,
)
from iskra.geometry.broadcast import (
    atleast_nd,
    broadcast_tensors,
    point_simplex_broadcast,
)
from iskra.geometry.normals import edge_normals, triangle_normals
from iskra.topology import face_to_subface_idcs


def simplex_codim(simplices: torch.Tensor) -> int:
    return simplices.shape[-1] - simplices.shape[-2] + 1


def edge_to_line(edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    origin = edges[..., 0, :]
    normal = edge_normals(edges)
    return origin, normal


def triangle_to_plane(triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    origin = triangles[..., 0, :]
    normal = triangle_normals(triangles)
    return origin, normal


def hyperplane_project(
    x: torch.Tensor, origin: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    x, origin, normal = broadcast_tensors(x, origin, normal)
    t: torch.Tensor = torch.linalg.vecdot(x - origin, normal)
    return x - t[..., None] * normal


def clamped_length_sqr(
    x: torch.Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Computes the _squared_ lengths of a set of vectors and then clamps them.

    Helps avoid numerical issues when computing lengths of vectors, such as:
        * The gradient of the square-root tends to infinitey as we get closer to zero.
        * We often wish to divide by squared length, which also explodes around zero.

    Args:
        x (Tensor[Any, ...]): Set of vectors.
        dim (int | tuple[int, ...]): The dimension(s) along which to compute lengths.
        keepdim (bool): Whether to reduce the selected dimensions
            or to keep them with the length one.
        eps (float): The _squared length_ will be clamped to this minimum value.

    Returns:
        Tensor[Any, ...]: Squared lengths of vectors along dimension dim.
    """
    sqr_distance = torch.sum(x * x, dim=dim, keepdim=keepdim)
    return sqr_distance.clamp_min(eps)


def clamped_length(
    x: torch.Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes the lengths of a set of vectors and then clamps them.

    Helps avoid numerical issues when computing lengths of vectors, such as:
        * The gradient of the square-root tends to infinitey as we get closer to zero.
        * We often wish to divide by length, which also explodes around zero.

    Args:
        x (Tensor[Any, ...]): Set of vectors.
        dim (int | tuple[int, ...]): The dimension(s) along which to compute lengths.
        keepdim (bool): Whether to reduce the selected dimensions
            or to keep them with the length one.
        eps (float): The _length_ will be clamped to this minimum value.

    Returns:
        Tensor[Any, ...]: Lengths of vectors along dimension dim.
    """
    return torch.sqrt(clamped_length_sqr(x, dim=dim, keepdim=keepdim, eps=eps * eps))


def point_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    ord: int | float | str = 2,
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the distance between vectors x_i and y_i.

    Unlike PyTorch's `torch.cdist`, this function works with `torch.func` transforms.

    Args:
        x (torch.Tensor): `[..., D]` tensor of d-dimensional vectors.
        y (torch.Tensor): `[..., D]` tensor of d-dimensional vectors.
        ord (int | float | str, optional): Order of p-norm.
        follows same convention as PyTorch's `vector_norm`. Defaults to 2.
        keepdim (bool, optional): Whether to keep the last dimension after reduction.
            Defaults to False.

    Raises:
        ValueError: Tensors x and y must have the same shape.

    Returns:
        torch.Tensor: A tensor of size `[..., 1]` if `keepdim=True`,
            otherwise with the last dimension removed.
    """
    if x.ndim != y.ndim:
        raise ValueError(
            f"Tensors have mismatching number of dimensions: {x.shape} != {y.shape}."
        )
    x, y = broadcast_tensors(x, y)
    if ord == 2:
        # Use specialized routine for 2-norm:
        diff = x - y
        sqr_distance = torch.sum(diff * diff, -1, keepdim=keepdim)
        distance = torch.sqrt(sqr_distance.clamp_min(1e-12))
    elif (isinstance(ord, int) or isinstance(ord, float)) and ord % 2 == 0:
        # Clamp other even L-norms to avoid NaNs:
        sqr_distance = torch.sum((x - y) ** ord, -1, keepdim=keepdim)
        distance = torch.pow(sqr_distance.clamp_min(1e-12), 1 / ord)
    else:
        # For other norms use default PyTorch behavior:
        distance: torch.Tensor = torch.linalg.vector_norm(
            x - y, axis=-1, ord=ord, keepdim=keepdim
        )
    return distance


def edge_project(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    x, edges = point_simplex_broadcast(x, edges)
    origin = edges[..., 0, :]
    edge_vectors = edges[..., 1, :] - edges[..., 0, :]
    length = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
    edge_vectors = edge_vectors / (length + 1e-12)

    t = torch.linalg.vecdot((x - origin) / (length + 1e-12), edge_vectors)
    t = torch.clamp(t, min=0, max=1)
    bary = torch.stack([1 - t, t], -1)
    return barycentric_interpolate(edges, bary)


def triangle_project(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    x, triangles = point_simplex_broadcast(x, triangles)
    if simplex_codim(triangles) > 0:
        # Project onto triangle plane:
        origin, normal = triangle_to_plane(triangles)
        x = hyperplane_project(x, origin, normal)

    idcs = face_to_subface_idcs(2)
    edges = torch.stack([triangles[..., idx, :] for idx in idcs], -3)
    projections = edge_project(x[..., None, :], edges)
    distances = point_dist(x[..., None, :], projections)
    min_distance, min_idx = torch.min(distances, -1, keepdim=True)
    closest_point_shape = min_distance.shape + projections.shape[-1:]
    gather_idx = min_idx[..., None].expand(closest_point_shape)
    closest_edge_point = torch.gather(projections, -2, gather_idx)
    closest_edge_point = closest_edge_point.squeeze(-2)

    # Compute barycentric coordinates and clamp to triangle interior:
    bary, valid = triangle_barycentric_coordinates(x, triangles)
    is_inside = torch.all(bary >= 0, -1) & valid
    x[~is_inside] = closest_edge_point[~is_inside]
    return x


def tetrahedron_project(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    x, tetrahedra = point_simplex_broadcast(x, tetrahedra)
    if tetrahedra.shape[-1] != 3 or x.shape[-1] != 3:
        raise ValueError("Only 3D tetrahedra are supported.")

    idcs = face_to_subface_idcs(3)
    triangles = torch.stack([tetrahedra[..., idx, :] for idx in idcs], -3)
    projections = triangle_project(x[..., None, :], triangles)
    distances = point_dist(x[..., None, :], projections)
    min_distance, min_idx = torch.min(distances, -1, keepdim=True)
    closest_point_shape = min_distance.shape + projections.shape[-1:]
    gather_idx = min_idx[..., None].expand(closest_point_shape)
    closest_triangle_point = torch.gather(projections, -2, gather_idx)
    closest_triangle_point = closest_triangle_point.squeeze(-2)

    # Compute barycentric coordinates and clamp to triangle interior:
    bary, valid = tetrahedron_barycentric_coordinates(x, tetrahedra)
    is_inside = torch.all(bary >= 0, -1) & valid
    x[~is_inside] = closest_triangle_point[~is_inside]
    return x


def simplex_project(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return tetrahedron_project(x, simplices)
    elif n_simplex_verts == 3:
        return triangle_project(x, simplices)
    elif n_simplex_verts == 2:
        return edge_project(x, simplices)
    else:
        raise NotImplementedError(
            "simplex_project only supports edges, triangles, and tetrahedra."
        )


def triangle_udf(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    x, triangles = point_simplex_broadcast(x, triangles)
    if triangles.shape[-1] != 2 or x.shape[-1] != 2:
        raise ValueError("Only 2D triangles are supported.")

    idcs = face_to_subface_idcs(2)
    edges = torch.stack([triangles[..., idx, :] for idx in idcs], -3)
    projections = edge_project(x[..., None, :], edges)
    distances = point_dist(x[..., None, :], projections)
    min_distance: torch.Tensor = torch.min(distances, -1, keepdim=False)[0]
    return min_distance


def tetrahedron_udf(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    x, tetrahedra = point_simplex_broadcast(x, tetrahedra)
    if tetrahedra.shape[-1] != 3 or x.shape[-1] != 3:
        raise ValueError("Only 3D tetrahedra are supported.")

    idcs = face_to_subface_idcs(3)
    triangles = torch.stack([tetrahedra[..., idx, :] for idx in idcs], -3)
    projections = triangle_project(x[..., None, :], triangles)
    distances = point_dist(x[..., None, :], projections)
    min_distance: torch.Tensor = torch.min(distances, -1, keepdim=False)[0]
    return min_distance


def point_edge_dist(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return point_dist(x, edge_project(x, edges))


def point_triangle_dist(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    return point_dist(x, triangle_project(x, triangles))


def point_tetrahedron_dist(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    return point_dist(x, tetrahedron_project(x, tetrahedra))


def point_simplex_dist(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return point_tetrahedron_dist(x, simplices)
    elif n_simplex_verts == 3:
        return point_triangle_dist(x, simplices)
    elif n_simplex_verts == 2:
        return point_edge_dist(x, simplices)
    else:
        raise NotImplementedError(
            "point_simplex_dist only supports edges, triangles, and tetrahedra."
        )


def point_dist_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    ord: int | float | str = 2,
) -> torch.Tensor:
    x, y = atleast_nd(2, x, y)
    return torch.cdist(x, y, p=ord)


def point_edge_dist_matrix(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    (x,) = atleast_nd(2, x)
    (edges,) = atleast_nd(3, edges)
    x, edges = x[..., :, None, :], edges[..., None, :, :, :]
    return point_edge_dist(x, edges)


def point_triangle_dist_matrix(
    x: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    (x,) = atleast_nd(2, x)
    (triangles,) = atleast_nd(3, triangles)
    x, triangles = x[..., :, None, :], triangles[..., None, :, :, :]
    return point_triangle_dist(x, triangles)


def point_tetrahedron_dist_matrix(
    x: torch.Tensor, tetrahedra: torch.Tensor
) -> torch.Tensor:
    (x,) = atleast_nd(2, x)
    (tetrahedra,) = atleast_nd(3, tetrahedra)
    x, tetrahedra = x[..., :, None, :], tetrahedra[..., None, :, :, :]
    return point_tetrahedron_dist(x, tetrahedra)


def point_simplex_dist_matrix(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return point_tetrahedron_dist_matrix(x, simplices)
    elif n_simplex_verts == 3:
        return point_triangle_dist_matrix(x, simplices)
    elif n_simplex_verts == 2:
        return point_edge_dist_matrix(x, simplices)
    else:
        raise NotImplementedError(
            "point_simplex_dist_matrix only supports edges, triangles, and tetrahedra."
        )


def closest_edge(
    x: torch.Tensor, edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (x,) = atleast_nd(2, x)
    (edges,) = atleast_nd(3, edges)
    x, edges = x[..., :, None, :], edges[..., None, :, :, :]
    projections = edge_project(x, edges)
    distances = point_dist(x, projections)
    closest_distance, prim_idx = torch.min(distances, -1)
    gather_idx = prim_idx[..., None, None].expand(
        *(-1,) * (projections.ndim - 1), projections.shape[-1]
    )
    closest_projection = torch.gather(projections, -2, gather_idx).squeeze(-2)
    return closest_projection, closest_distance, prim_idx


def closest_triangle(
    x: torch.Tensor, triangles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (x,) = atleast_nd(2, x)
    (triangles,) = atleast_nd(3, triangles)
    x, triangles = x[..., :, None, :], triangles[..., None, :, :, :]
    projections = triangle_project(x, triangles)
    distances = point_dist(x, projections)
    closest_distance, prim_idx = torch.min(distances, -1)
    gather_idx = prim_idx[..., None, None].expand(
        *(-1,) * (projections.ndim - 1), projections.shape[-1]
    )
    closest_projection = torch.gather(projections, -2, gather_idx).squeeze(-2)
    return closest_projection, closest_distance, prim_idx


def closest_simplex(
    x: torch.Tensor, simplices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return closest_triangle(x, simplices)
    elif n_simplex_verts == 2:
        return closest_edge(x, simplices)
    else:
        raise NotImplementedError("closest_simplex only supports edges and triangles.")

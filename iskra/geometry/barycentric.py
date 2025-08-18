# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.broadcast import point_simplex_broadcast


def edge_barycentric_coordinates(
    x: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the barycentric coordinates of points on edges.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Batched points of dimension Dim,
            with Dim being of dimension 1 or larger.
        edges (Tensor[Float, [Bs, 2, Dim]]): Batched edge endpoints.
        eps (float): Numerical epsilon to use when dividing by the length of
            an edge.

    Returns:
        (Tensor[Float, [Bs, 2]]): Barycentric coordinates.
        (Tensor[Bool, [Bs]]): Boolean indicating whether the barycentric
            weights were valid, i.e. whether the edge was non-degenerate.
    """
    assert x.shape[:-1] == edges.shape[:-2]
    assert edges.ndim == x.ndim + 1
    assert edges.shape[-2] == 2

    origin = edges[..., 0, :]
    direction = edges[..., 1, :] - edges[..., 0, :]
    length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    valid_edges = length[..., 0] > eps

    length_offset = length + eps
    direction = direction / length_offset
    t = torch.linalg.vecdot((x - origin) / length_offset, direction)
    bary = torch.stack([1 - t, t], -1)
    return bary, valid_edges


def triangle_barycentric_coordinates(
    x: torch.Tensor, triangles: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the barycentric coordinates of points on triangles.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Batched points of dimension Dim,
            with Dim being either 2 or 3.
        triangles (Tensor[Float, [Bs, 3, Dim]]): Batched triangle corners.
        eps (float): Numerical epsilon to use when dividing by the area of
            a triangle.

    Returns:
        (Tensor[Float, [Bs, 3]]): Barycentric coordinates.
        (Tensor[Bool, [Bs]]): Boolean indicating whether the barycentric
            weights were valid, i.e. whether the triangle was non-degenerate.
    """
    x, triangles = point_simplex_broadcast(x, triangles)

    if triangles.shape[-1] == 2:
        triangles = torch.nn.functional.pad(triangles, pad=(0, 1))
    if x.shape[-1] == 2:
        x = torch.nn.functional.pad(x, pad=(0, 1))

    assert triangles.ndim == x.ndim + 1
    assert triangles.shape[-1] == 3
    assert triangles.shape[-2] == 3

    assert x.shape[-1] == 3

    # NOTE: barycentric coordinates are ratios of the (signed)
    # determinant of the wedge products of the k-vectors
    # formed by the triangle/tet.
    relative_triangles = triangles - triangles[..., 0:1, :]
    relative_x = x - triangles[..., 0, :]

    # Compute triangle normal and area:
    triangle_cross = torch.linalg.cross(
        relative_triangles[..., 1, :], relative_triangles[..., 2, :], dim=-1
    )
    triangle_area = torch.linalg.vector_norm(triangle_cross, dim=-1, keepdim=True)
    valid_triangles = torch.abs(triangle_area[..., 0]) > eps
    area_offset = triangle_area + eps

    bary_1_cross = torch.linalg.cross(relative_x, relative_triangles[..., 2, :], dim=-1)
    bary_1_area = (
        torch.sum(triangle_cross * bary_1_cross, -1, keepdim=True) / area_offset
    )

    bary_2_cross = torch.linalg.cross(relative_triangles[..., 1, :], relative_x, dim=-1)
    bary_2_area = (
        torch.sum(triangle_cross * bary_2_cross, -1, keepdim=True) / area_offset
    )

    bary_0 = torch.empty_like(triangle_area)
    bary_1 = torch.empty_like(triangle_area)
    bary_2 = torch.empty_like(triangle_area)

    bary_1[valid_triangles] = (
        bary_1_area[valid_triangles] / area_offset[valid_triangles]
    )
    bary_2[valid_triangles] = (
        bary_2_area[valid_triangles] / area_offset[valid_triangles]
    )
    bary_0[valid_triangles] = (1 - bary_1 - bary_2)[valid_triangles]
    bary = torch.cat([bary_0, bary_1, bary_2], -1)

    return bary, valid_triangles


def _scalar_triple_product(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    return torch.sum(a * torch.cross(b, c, dim=-1), dim=-1, keepdim=True)


def tetrahedron_barycentric_coordinates(
    x: torch.Tensor, tets: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the barycentric coordinates of points in tetrahedra.

    Args:
        x (Tensor[Float, [Bs, 3]]): Batched points in 3 dimensions.
        tets (Tensor[Float, [Bs, 4, 3]]): Batched tetrahedra corners.
        eps (float): Numerical epsilon to use when dividing by the volume of
            a tetrahedron.

    Returns:
        (Tensor[Float, [Bs, 4]]): Barycentric coordinates.
        (Tensor[Bool, [Bs]]): Boolean indicating whether the barycentric
            weights were valid, i.e. whether the tetrahedron was non-degenerate.
    """
    x, tets = point_simplex_broadcast(x, tets)
    assert tets.shape[-1] == 3
    assert tets.shape[-2] == 4
    assert x.shape[-1] == 3

    tet_edges = tets[..., :-1, :] - tets[..., -1:, :]
    x_to_tet_edges = tets - x[..., None, :]

    tet_volume = _scalar_triple_product(
        tet_edges[..., 0, :],
        tet_edges[..., 1, :],
        tet_edges[..., 2, :],
    )
    valid_tets = torch.abs(tet_volume[..., 0]) > eps

    v0_tet_volume = _scalar_triple_product(
        x_to_tet_edges[..., 1, :], x_to_tet_edges[..., 3, :], x_to_tet_edges[..., 2, :]
    )
    v1_tet_volume = _scalar_triple_product(
        x_to_tet_edges[..., 0, :], x_to_tet_edges[..., 2, :], x_to_tet_edges[..., 3, :]
    )
    v2_tet_volume = _scalar_triple_product(
        x_to_tet_edges[..., 0, :], x_to_tet_edges[..., 3, :], x_to_tet_edges[..., 1, :]
    )
    v3_tet_volume = _scalar_triple_product(
        x_to_tet_edges[..., 0, :], x_to_tet_edges[..., 1, :], x_to_tet_edges[..., 2, :]
    )
    inv_tet_volume = 1.0 / (tet_volume + eps)

    coordinates = torch.cat(
        [v0_tet_volume, v1_tet_volume, v2_tet_volume, v3_tet_volume], -1
    )
    coordinates *= inv_tet_volume
    return coordinates, valid_tets


def barycentric_coordinates(
    x: torch.Tensor, simplices: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the barycentric coordinates of points on simplices.

    This is thin dispatcher that picks one of the specialized
    barycentric coordinates function based on the number of corners in the simplex.
    Works for edges, triangles, and tetrahedra.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Batched points in Dim dimensions.
        simplices (Tensor[Float, [Bs, SDim, Dim]]): Batched simplex corners,
            where SDim is either 2, 3, or 4.
        eps (float): Numerical epsilon to use when dividing by the generalized
            volume of a simplex.

    Returns:
        (Tensor[Float, [Bs, SDim]]): Barycentric coordinates.
        (Tensor[Bool, [Bs]]): Boolean indicating whether the barycentric
            weights were valid, i.e. whether the simplex was non-degenerate.
    """
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        coordinates, valid = tetrahedron_barycentric_coordinates(x, simplices, eps=eps)
    elif n_simplex_verts == 3:
        coordinates, valid = triangle_barycentric_coordinates(x, simplices, eps=eps)
    elif n_simplex_verts == 2:
        coordinates, valid = edge_barycentric_coordinates(x, simplices, eps=eps)
    else:
        raise NotImplementedError(
            "barycentric_coordinates only supports edges, triangles, and tetrahedra."
        )

    return coordinates, valid


def is_inside_pairwise(
    x: torch.Tensor, simplices: torch.Tensor, eps: float = 1e-24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checks whether each of the points in x is in any of the simplices in simplices.

    Args:
        x (Tensor[Float, [B1, Dim]]): Batched points in Dim dimensions
        simplices (Tensor[Float, [B2, SDim, Dim]]): Batched simplex corners,
            where SDim is either 2, 3, or 4.
        eps (float): Numerical epsilon to use to prevent errors in case
            of degenerate simplices.

    Returns:
        (Tensor[Bool, [B1, B2]]): Boolean tensor that indicates when a point
            is inside of a simplex.
        (Tensor[Float, [B1, B2, SDim]]): Barycentric coordinates of each point
            in each simplex.
    """
    assert x.shape[-1] == simplices.shape[-1]

    simplices, x = torch.broadcast_tensors(  # type: ignore
        simplices[None, :, :, :], x[:, None, None, :]
    )
    x = x[..., 0, :]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        coordinates, _ = tetrahedron_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= eps, -1)
    elif n_simplex_verts == 3:
        coordinates, _ = triangle_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= eps, -1)
    elif n_simplex_verts == 2:
        coordinates, _ = edge_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= eps, -1)
    else:
        raise NotImplementedError(
            "is_inside_pairwise only supports edges, triangles, and tetrahedra."
        )

    return is_inside, coordinates


def barycentric_interpolate(values: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """Interpolate values onto a position given barycentric coordinates.

    Args:
        values (Tensor[Float, [Bs, SDim, F]]): The F-dimensional per-corner values
            to be interpolated.
        bary (Tensor[Float, [Bs, Sdim]]): The barycentric coordinates.

    Returns:
        Tensor[Float, [Bs, F]]: Interpolated values.
    """
    if bary.ndim != values.ndim - 1 or bary.shape != values.shape[:-1]:
        raise ValueError(
            "Incompatible tensor shapes."
            f"bary has to be [Bs, SDim] but is {bary.shape}, "
            f"values has to be [Bs, SDim, F] but is {values.shape}."
        )
    return torch.sum(bary[..., None] * values, -2)

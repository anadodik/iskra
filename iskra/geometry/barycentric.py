# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.broadcast import point_simplex_broadcast


def edge_barycentric_coordinates(
    x: torch.Tensor, edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[:-1] == edges.shape[:-2]
    assert edges.ndim == x.ndim + 1
    assert edges.shape[-2] == 2

    origin = edges[..., 0, :]
    direction = edges[..., 1, :] - edges[..., 0, :]
    length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    direction = direction / length

    valid_edges = length[..., 0] > 1e-12
    t = torch.linalg.vecdot((x - origin) / length, direction)
    bary = torch.stack([1 - t, t], -1)
    return bary, valid_edges


def triangle_barycentric_coordinates(
    x: torch.Tensor, triangles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the barycentric coordinates of points in triangles.

    Args:
        x: torch.Tensor of shape [B, D] where B is the batch size, and
            D is the ambient dimension (either 2 or 3).
        triangles: torch.Tensor of shape [B, 3, D], where B is the batch
            size, 3 represents the three triangle vertices, and D is the ambient
            dimension (either 2 or 3).


    Returns:
        A float torch.Tensor of shape [B, 3] with the barycentric weight of each
        vertex on x in the triangles.

        A boolean torch.Tensor of shape [B] indicating whether the barycentric
        weights were valid, i.e. indicating whether the triangle was non-degenerate
        and whether x was inside of the triangle.
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

    bary_1_cross = torch.linalg.cross(relative_x, relative_triangles[..., 2, :], dim=-1)
    bary_1_area = (
        torch.sum(triangle_cross * bary_1_cross, -1, keepdim=True) / triangle_area
    )

    bary_2_cross = torch.linalg.cross(relative_triangles[..., 1, :], relative_x, dim=-1)
    bary_2_area = (
        torch.sum(triangle_cross * bary_2_cross, -1, keepdim=True) / triangle_area
    )

    valid_triangles = torch.abs(triangle_area[..., 0]) > 1e-12

    bary_0 = torch.empty_like(triangle_area)
    bary_1 = torch.empty_like(triangle_area)
    bary_2 = torch.empty_like(triangle_area)

    bary_1[valid_triangles] = (
        bary_1_area[valid_triangles] / triangle_area[valid_triangles]
    )
    bary_2[valid_triangles] = (
        bary_2_area[valid_triangles] / triangle_area[valid_triangles]
    )
    bary_0[valid_triangles] = (1 - bary_1 - bary_2)[valid_triangles]
    bary = torch.cat([bary_0, bary_1, bary_2], -1)

    return bary, valid_triangles


def _scalar_triple_product(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    return torch.sum(a * torch.cross(b, c, dim=-1), dim=-1, keepdim=True)


def tetrahedron_barycentric_coordinates(
    x: torch.Tensor, tets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    inv_tet_volume = 1.0 / tet_volume

    valid = inv_tet_volume[..., 0].isfinite()

    coordinates = torch.cat(
        [v0_tet_volume, v1_tet_volume, v2_tet_volume, v3_tet_volume], -1
    )
    coordinates *= inv_tet_volume
    return coordinates, valid


def barycentric_coordinates(
    x: torch.Tensor, simplices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        coordinates, valid = tetrahedron_barycentric_coordinates(x, simplices)
    elif n_simplex_verts == 3:
        coordinates, valid = triangle_barycentric_coordinates(x, simplices)
    elif n_simplex_verts == 2:
        coordinates, valid = edge_barycentric_coordinates(x, simplices)
    else:
        raise NotImplementedError(
            "barycentric_coordinates only supports edges, triangles, and tetrahedra."
        )

    return coordinates, valid


def is_inside_pairwise(
    x: torch.Tensor, simplices: torch.Tensor, tol: float = 1e-24
) -> torch.Tensor:
    """Checks whether each of the points in `x` is in any of the tets in `tets`.

    Args:
        x (torch.Tensor): [V, dim] tensor of point positions.
        simplices (torch.Tensor): [T, n_verts, dim] tensor of simplex vertex positions.
        tol (float): Epsilon tolerance on the barycentric coordinates for whether a
            a point is inside or outside.

    Returns:
        torch.Tensor: [V, T] binary tensor that signifies whether a point
        is contained inside of a simplex.
    """
    assert x.shape[-1] == simplices.shape[-1]

    simplices, x = torch.broadcast_tensors(  # type: ignore
        simplices[None, :, :, :], x[:, None, None, :]
    )
    x = x[..., 0, :]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        coordinates, _ = tetrahedron_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= tol, -1)
    elif n_simplex_verts == 3:
        coordinates, _ = triangle_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= tol, -1)
    elif n_simplex_verts == 2:
        coordinates, _ = edge_barycentric_coordinates(x, simplices)
        is_inside = torch.all(coordinates >= tol, -1)
    else:
        raise NotImplementedError(
            "is_inside_pairwise only supports edges, triangles, and tetrahedra."
        )

    return is_inside


def barycentric_interpolate(values: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """Interpolate values onto a position given barycentric coordinates.

    Args:
        values (torch.Tensor): A tensor of shape [..., n_simplex_vertices, D] containing
            the per-vertex values to be interpolated.
        bary (torch.Tensor): A tensor of shape [..., n_simplex_vertices] containing
            the barycentric coordinates.

    Returns:
        torch.Tensor: A tensor of shape [..., D] containing the interpolated values.
    """
    if bary.ndim != values.ndim - 1 or bary.shape[-1] != values.shape[-2]:
        raise ValueError(
            "Incompatible tensor shapes."
            "bary has to be [..., n_simplex_vertices] but is {bary.shape}, "
            f"values has to be [..., n_simplex_vertices] but is {values.shape}."
        )
    return torch.sum(bary[..., None] * values, -2)

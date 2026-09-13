# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from typing import Literal

import torch

from iskra.geometry.angles import interior_angles
from iskra.topology import face_index, reduce_on_subface


def edge_length_normals(edges: torch.Tensor) -> torch.Tensor:
    """Normals of line segments in 2D *scaled by the edge length*.

    This function is useful when computing length-weighted vertex normals.

    Args:
        edges (Tensor[Float, [Bs, 2, 2]]): Line segments in 2D; second to last dimension
            is the end points, last is the coordinates.

    Returns:
        Tensor[Float, [Bs, 2]]: *Scaled* normal vectors of each line segment.
    """
    edge_vector = edges[..., 1, :] - edges[..., 0, :]
    orth_vector = torch.tensor([1.0, -1.0], device=edges.device, dtype=edges.dtype)
    orth_vector = torch.broadcast_to(orth_vector, edge_vector.shape)
    normal = edge_vector[..., (1, 0)] * orth_vector
    assert normal.shape[:-1] == edges.shape[:-2]
    assert normal.shape[-1] == 2
    return normal


def edge_normals(edges: torch.Tensor) -> torch.Tensor:
    """Normals of line segments in 2D.

    Args:
        edges (Tensor[Float, [Bs, 2, 2]]): Line segments in 2D; second to last dimension
            is the end points, last is the coordinates.

    Returns:
        Tensor[Float, [Bs, 2]]: Normal vectors of each line segment.
    """
    return torch.nn.functional.normalize(edge_length_normals(edges), dim=-1)


def triangle_area_normals(triangles: torch.Tensor) -> torch.Tensor:
    """Normals of triangles in 2D or 3D *scaled by the triangle area*.

    This function is useful when computing area-weighted vertex normals.
    Normals of 2D triangles will always be [0, 0, area].

    Args:
        triangles (Tensor[Float, [Bs, 3, 2 | 3]]): Triangles in 2D or 3D;
            second to last dimension is the corners, last is the coordinates.

    Returns:
        Tensor[Float, [Bs, 3]]: *Scaled* normal vectors of each triangle.
    """
    if triangles.shape[-1] == 2:
        triangles = torch.nn.functional.pad(triangles, pad=(0, 1))
    assert len(triangles.shape) >= 2
    assert triangles.shape[-2] == 3
    assert triangles.shape[-1] == 3

    relative_triangles = triangles - triangles[..., 0:1, :]
    double_area_normals: torch.Tensor = torch.linalg.cross(
        relative_triangles[..., 1, :], relative_triangles[..., 2, :], dim=-1
    )
    return 0.5 * double_area_normals


def triangle_normals(triangles: torch.Tensor) -> torch.Tensor:
    """Normals of triangles in 2D or 3D.

    Normals of 2D triangles will always be [0, 0, 1].

    Args:
        triangles (Tensor[Float, [Bs, 3, 2 | 3]]): Triangles in 2D or 3D;
            second to last dimension is the corners, last is the coordinates.

    Returns:
        Tensor[Float, [Bs, 3]]: Normal vectors of each triangle.
    """
    return torch.nn.functional.normalize(triangle_area_normals(triangles), dim=-1)


def volume_face_normals(simplices: torch.Tensor) -> torch.Tensor:
    """Computes face normals weighted by the hypervolume of the element.

    Thin wrapper dispatching to either `iskra.edge_length_normals()` or
    `iskra.triangle_area_normals()`. Works on triangle meshes in 3D and polylines in 2D.

    Args:
        simplices (Tensor[Float, [Bs, S, Dim]]): Simplices tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Raises:
        NotImplementedError: Simplices must have either 2 or 3 corners.

    Returns:
        Tensor[Float, [Bs, S, Dim]]: Volume weighted face normals.
    """
    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return triangle_area_normals(simplices)
    elif n_simplex_verts == 2:
        return edge_length_normals(simplices)
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")


def face_normals(simplices: torch.Tensor) -> torch.Tensor:
    """Computes per-face normals.

    Thin wrapper dispatching to either `iskra.edge_normals()` or
    `iskra.triangle_normals()`. Works on triangle meshes in 3D and polylines in 2D.

    Args:
        simplices (Tensor[Float, [Bs, S, Dim]]): Simplices tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Raises:
        NotImplementedError: Simplices must have either 2 or 3 corners.

    Returns:
        Tensor[Float, [Bs, S, Dim]]: Face normals.
    """
    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return triangle_normals(simplices)
    elif n_simplex_verts == 2:
        return edge_normals(simplices)
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")


def vertex_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
    method: Literal["default", "graph", "area", "angle"] = "default",
) -> torch.Tensor:
    """Comptues per-vertex normals.

    Normals are computed per face and then reduced to the vertices using a
    a weighted average. The weighting is determined via the argument `method`.
    The function works for triangle meshes in 3D and polylines in 2D.

    Args:
        verts (Tensor[Float, [V, 2 | 3]]): Mesh vertex positions.
        faces (Tensor[Int, [F, 2 | 3]]): Mesh face indices.
        method (Literal["default", "graph", "area", "angle"]): Normal averaging method.
            One of:

            - **`default`**: selects angle weighing for triangle meshes and area
                weighing for polyline meshes. Chosen by default.
            - **`area`**: weighs each face normal by that face's hypervolume,
                approximating an integrated normal vector.
            - **`angle`**: angle-based normal vectors based on the heuristic from
                [TODO: cite]. By far the most visually pleasent, but only work
                on 3D meshes and a bit less theoretically justified than others.
            - **`graph`**: each face normal gets weight 1.0.

    Raises:
        ValueError: Passing `method=angle` for a polyline mesh with raise an error.
        ValueError: `method` must be in `["default", "graph", "area", "angle"`]`.
        NotImplementedError: Simplices must have either 2 or 3 corners.

    Returns:
        Tensor[Float, [V, 2 | 3]]: Vertex-based normals
    """
    simplices = face_index(verts, faces)
    volume_normals = volume_face_normals(simplices)
    if faces.shape[-1] == 2:
        if method in ("area", "default"):
            normals = reduce_on_subface(volume_normals, faces, verts.shape[0], "sum")
        elif method == "graph":
            face_normals = torch.nn.functional.normalize(volume_normals, dim=-1)
            normals = reduce_on_subface(face_normals, faces, verts.shape[0], "sum")
        elif method == "angle":
            raise ValueError("Angle normals not well defined for polylines.")
        else:
            raise ValueError(f"Unknown normal computation method {method}.")
    elif faces.shape[-1] == 3:
        if method in ("default", "angle"):
            face_normals = torch.nn.functional.normalize(volume_normals, dim=-1)
            angles = interior_angles(simplices, signed=False, face_normals=face_normals)
            angle_normals = angles[..., :, None] * face_normals[..., None, :]
            normals = reduce_on_subface(
                angle_normals, faces, verts.shape[0], "sum", data_ndim=1
            )
        elif method == "graph":
            face_normals = torch.nn.functional.normalize(volume_normals, dim=-1)
            normals = reduce_on_subface(face_normals, faces, verts.shape[0], "sum")
        elif method == "area":
            normals = reduce_on_subface(volume_normals, faces, verts.shape[0], "sum")
        else:
            raise ValueError(f"Unknown normal computation method {method}.")
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")
    return torch.nn.functional.normalize(normals, dim=-1)

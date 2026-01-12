# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from typing import Literal

import torch

from iskra.topology import face_to_subface_idcs, reduce_on_subface


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


def area_face_normals(simplices: torch.Tensor) -> torch.Tensor:
    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return triangle_area_normals(simplices)
    elif n_simplex_verts == 2:
        return edge_length_normals(simplices)
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")


def face_normals(simplices: torch.Tensor) -> torch.Tensor:
    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return triangle_normals(simplices)
    elif n_simplex_verts == 2:
        return edge_normals(simplices)
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")


def interior_angles(
    triangles: torch.Tensor,
    signed: bool = False,
    face_normals: torch.Tensor | None = None,
) -> torch.Tensor:
    # Get vertices opposite the corner vertex:
    idcs: list[tuple[int, ...]] = face_to_subface_idcs(2, 1)
    opposite_vecs = torch.stack([triangles[..., nbh_idx, :] for nbh_idx in idcs], -3)
    vecs = opposite_vecs - triangles[..., :, None, :]
    vecs = torch.nn.functional.normalize(vecs, dim=-1)

    cos_theta = torch.linalg.vecdot(vecs[..., :, 0, :], vecs[..., :, 1, :], dim=-1)
    cross = torch.linalg.cross(vecs[..., :, 0, :], vecs[..., :, 1, :], dim=-1)
    if signed:
        if face_normals is None:
            face_normals = triangle_normals(triangles)
        sin_theta = torch.linalg.vecdot(cross, face_normals[:, None, :], dim=-1)
    else:
        sin_theta = torch.linalg.vector_norm(cross, dim=-1)
    angles = torch.atan2(sin_theta, cos_theta)
    return angles


def vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    method: Literal["default", "graph", "area", "angle"] = "default",
) -> torch.Tensor:
    simplices = vertices[faces.flatten()].reshape(
        -1, faces.shape[-1], vertices.shape[-1]
    )
    face_normals = area_face_normals(simplices)
    if method == "graph":
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

    if faces.shape[-1] == 2:
        if method == "area":
            raise NotImplementedError

        normals = reduce_on_subface(face_normals, faces, vertices.shape[0], "sum")
    elif faces.shape[-1] == 3:
        if method in ("default", "angle"):
            face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

            normals = torch.zeros([vertices.shape[0], 3], device=face_normals.device)
            broadcast_faces = faces[:, :, None].expand(-1, -1, 3)
            for i in range(faces.shape[-1]):
                normals.scatter_add_(
                    0, broadcast_faces[:, i, ...], angles[:, i : i + 1] * face_normals
                )
        else:
            normals = reduce_on_subface(face_normals, faces, vertices.shape[0], "sum")
    else:
        raise NotImplementedError("Normals only supported for edges and triangles.")

    return torch.nn.functional.normalize(normals, dim=-1)

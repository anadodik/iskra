# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from typing import Literal

import torch

import iskra.sparse as sp
from iskra.geometry.volume import edge_lengths, volume_form, volume_form_intrinsic
from iskra.topology import face_index, reduce_on_subface


def mass(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return sp.diag(dual_volume)


def mass_inv(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    embedded_faces = face_index(vertices, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, vertices.shape[0], "sum")
    return sp.diag(1 / dual_volume)


def mass_intrinsic(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return sp.diag(vertex_areas)


def mass_intrinsic_inv(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    area = volume_form_intrinsic(edge_lengths, face_to_edge)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_vertices, "sum")
    return sp.diag(1 / vertex_areas)


def grad_triangle_3d(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Finite element gradient matrix for 3d triangles.

    Given a triangle mesh in 3d, computes the finite element gradient matrix
    assuming piecewise linear hat function basis.

    Args:
    vertices (torch.Tensor):(n,d) tensor vertex list a triangle mesh
    faces (torch.Tensor): tensor of ints of shape (m, 3) interpreted as face index
    list of a triangle mesh

    Returns:
    torch.Tensor: three (m, n) sparse coo tensor of the finite element gradient matrix
    """
    if faces.shape[-1] != 3:
        raise ValueError("grad_3d() implemented only for triangle meshes.")
    if vertices.shape[-1] != 3:
        raise ValueError("grad_3d() implemented only for triangle meshes in 3d")
    # TODO(anadodik): make it into a batched [3, n_faces, n_vertices] tensor.
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    device = vertices.device

    triangles = face_index(vertices, faces)
    edge_01 = triangles[:, 1, :] - triangles[:, 0, :]
    edge_20 = triangles[:, 0, :] - triangles[:, 2, :]

    face_normals = torch.linalg.cross(edge_01, -edge_20)
    double_face_areas = torch.linalg.vector_norm(face_normals, dim=-1, keepdim=True)
    face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=-1)

    rot_edge_01 = torch.linalg.cross(face_normals, edge_01) / double_face_areas
    rot_edge_20 = torch.linalg.cross(face_normals, edge_20) / double_face_areas

    idx_i = torch.cat([torch.arange(0, n_faces, device=device)] * 4)
    idx_j = torch.cat([faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 0]])
    idx = torch.stack([idx_i, idx_j])
    values = torch.cat([rot_edge_20, -rot_edge_20, rot_edge_01, -rot_edge_01])

    grad_x = sp.coo_tensor(idx, values[:, 0], size=[n_faces, n_vertices])
    grad_y = sp.coo_tensor(idx, values[:, 1], size=[n_faces, n_vertices])
    grad_z = sp.coo_tensor(idx, values[:, 2], size=[n_faces, n_vertices])
    return grad_x.coalesce(), grad_y.coalesce(), grad_z.coalesce()


def grad_triangle_2d(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite element gradient matrix for 2d triangles.

    Given a triangle mesh in 2d, computes the finite element gradient matrix
    assuming piecewise linear hat function basis.

    Args:
    vertices (torch.Tensor):(n,d) tensor vertex list a triangle mesh
    faces (torch.Tensor): tensor of ints of shape (m, 2) interpreted as face index
    list of a triangle mesh

    Returns:
    torch.Tensor: two (m, n) sparse coo tensor of the finite element gradient matrix

    Notes:
    Taken from https://github.com/sgsellan/gpytoolbox/blob/main/src/gpytoolbox/grad.py
    """
    if faces.shape[-1] != 3:
        raise ValueError("grad_2d() implemented only for triangle meshes.")
    if vertices.shape[-1] != 2:
        raise ValueError("grad_2d() implemented only for triangle meshes in 2d")

    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    device = vertices.device

    triangles = face_index(vertices, faces)

    edge_21 = triangles[:, 1, :] - triangles[:, 0, :]
    edge_13 = triangles[:, 0, :] - triangles[:, 2, :]

    # signed triangle areas
    double_face_areas = 0.5 * (
        edge_21[:, 1] * edge_13[:, 0] - edge_21[:, 0] * edge_13[:, 1]
    )

    # Rotate edge vectors by 90 degrees and normalize by area
    # In 2D, rotating by 90 degrees is done by (x, y) -> (-y, x)
    rot_edge_21 = torch.stack([-edge_21[:, 1], edge_21[:, 0]], dim=-1) / (
        2 * double_face_areas.unsqueeze(-1)
    )
    rot_edge_13 = torch.stack([-edge_13[:, 1], edge_13[:, 0]], dim=-1) / (
        2 * double_face_areas.unsqueeze(-1)
    )

    idx_i = torch.cat([torch.arange(0, n_faces, device=device)] * 4)
    idx_j = torch.cat([faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 0]])
    idx = torch.stack([idx_i, idx_j])
    values = torch.cat([rot_edge_13, -rot_edge_13, rot_edge_21, -rot_edge_21])

    grad_x = sp.coo_tensor(idx, values[:, 0], size=[n_faces, n_vertices])
    grad_y = sp.coo_tensor(idx, values[:, 1], size=[n_faces, n_vertices])

    return grad_x.coalesce(), grad_y.coalesce()


def grad_edges(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Finite element gradient matrix for edges.

    Given a polyline, computes the finite element gradient matrix assuming piecewise
    linear hat function basis.

    Args:
    vertices (torch.Tensor): (n,d) tensor vertex list of a polyline where
    edges (torch.Tensor): tensor of ints of shape (m, 2) interpreted as edge
    index list of a polyline

    Returns:
    torch.Tensor: (m, n) sparse coo tensor of the finite element gradient matrix
    """
    if edges.shape[-1] != 2:
        raise ValueError("grad_1d() implemented only for edges only.")

    edge_len = edge_lengths(
        torch.stack((vertices[edges[:, 1], :], vertices[edges[:, 0], :]), dim=1)
    )

    edge_len[edge_len == 0] = 1e-8

    inv_len = 1.0 / edge_len

    n_vertices = vertices.shape[0]
    n_faces = edges.shape[0]
    device = vertices.device

    idx_i = torch.arange(edges.shape[0], dtype=torch.long, device=device)
    idx_i = torch.cat((idx_i, idx_i))
    idx_j = torch.cat((edges[:, 0], edges[:, 1]))
    idx = torch.stack([idx_i, idx_j])

    values = torch.cat([-inv_len, inv_len])

    grad = sp.coo_tensor(idx, values, size=[n_faces, n_vertices])

    return grad


def grad(
    vertices: torch.Tensor, faces: torch.Tensor, stack: bool = False
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Finite element gradient matrix.

    Given a triangle mesh or a polyline, computes the finite element gradient matrix
    assuming piecewise linear hat function basis.

    Args:
    vertices (torch.Tensor):(n,d) tensor vertex list of a polyline or triangle mesh
    faces (torch.Tensor): tensor of ints
        if (m, 2),  interpret as edge index list of a polyline
        if (m, 3),  interpret as face index list of a triangle mesh

    Returns:
    torch.Tensor: S (m, n) sparse coo tensor of the finite element gradient matrix
    where S is 1 for polyline, 2 for triangle mesh in 2d, and 3 for triangle mesh in 3d
    """
    if faces.shape[-1] == 2:
        grads = grad_edges(vertices, faces)
    if vertices.shape[-1] == 2:
        grads = grad_triangle_2d(vertices, faces)
    elif vertices.shape[-1] == 3:
        grads = grad_triangle_3d(vertices, faces)
    else:
        raise ValueError(f"Gradients not implemented for dim = {vertices.shape[-1]}")

    if stack:
        return sp.cat(grads, -2)
    else:
        return grads


def grad_to_div(g: torch.Tensor, tri_areas: torch.Tensor) -> torch.Tensor:
    g_dim = g.shape[0] // tri_areas.shape[-1]
    assert g_dim == 3
    areas = torch.cat(g_dim * [tri_areas])
    return sp.mul(areas[..., None, :], g.mT)


def laplacian(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weight_type: Literal["cotan", "uniform"] = "cotan",
) -> torch.Tensor:
    pass


def laplacian_intrinsic(
    edge_lengths: torch.Tensor,
    face_to_edge: torch.Tensor,
    n_vertices: int,
    faces: torch.Tensor,
    weight_type: Literal["cotan", "uniform"] = "cotan",
) -> torch.Tensor:
    pass

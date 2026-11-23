# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

import iskra.sparse as sp
from iskra.geometry.normals import triangle_area_normals
from iskra.geometry.volume import (
    signed_tetrahedron_volumes,
    volume_form,
    volume_form_intrinsic,
)
from iskra.topology import (
    face_index,
    face_to_subface_idcs,
    get_subfaces,
    reduce_on_subface,
)


def mass(verts: torch.Tensor, faces: torch.Tensor) -> sp.SparseTensor:
    """FEM lumped mass matrix with barycentric dual volumes.

    The finite element `V` times `V` mass matrix where each diagonal entry corresponds
    to the volume of the dual element at that vertex. It uses the barycentric dual cell
    volume.

    Args:
        verts (Tensor[Float, [V, Dim]]): Vertex positions.
        faces (Tensor[Int64, [F, FV]]): Face index.

    Returns:
        SparseTensor[Float, [V, V]]: Diagonal sparse COO mass matrix.
    """
    embedded_faces = face_index(verts, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, verts.shape[0], "sum")
    return sp.diag(dual_volume)


def mass_inv(verts: torch.Tensor, faces: torch.Tensor) -> sp.SparseTensor:
    """FEM lumped inverse mass matrix with barycentric dual volumes.

    The finite element `V` times `V` mass matrix where each diagonal entry corresponds
    to one over the volume of the dual element at that vertex. It uses the barycentric
    dual cell volume.

    Warning:
        You rarely want to use this function. Most often when solving linear systems,
        you will want to call `iskra.fem.mass()` and then premultiply the right-hand
        side with the mass matrix.

    Args:
        verts (Tensor[Float, [V, Dim]]): Vertex positions.
        faces (Tensor[Int64, [F, FV]]): Face index.

    Returns:
        SparseTensor[Float, [V, V]]: Diagonal sparse COO inverse mass matrix.
    """
    embedded_faces = face_index(verts, faces)
    volume = volume_form(embedded_faces)
    n_corners = faces.shape[-1]
    dual_volume = reduce_on_subface(volume / n_corners, faces, verts.shape[0], "sum")
    return sp.diag(1 / dual_volume)


def mass_intrinsic(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_subface: torch.Tensor,
    n_verts: int,
) -> sp.SparseTensor:
    """FEM lumped mass matrix with barycentric dual volumes, from edge lengths.

    The finite element `V` times `V` mass matrix where each diagonal entry corresponds
    to the volume of the dual element at that vertex. It uses the barycentric
    dual cell volume.

    This function computes it from edge lengths alone, meaning we do not have to have an
    embedding of the mesh.

    See Also:
        To obtain the `face_subface` tensor, use `iskra.topology.get_subfaces()`.

    Args:
        edge_lengths (Tensor[Float, [E]]): Lengths of mesh edges.
        faces (Tensor[Int64, [F, FV]]): Face index.
        face_subface (Tensor[Int64, [F, FS]]): Face-to-subface index.
        n_verts (int): Number of mesh vertices.

    Returns:
        SparseTensor[Float, [V, V]]: Diagonal sparse COO mass matrix.
    """
    area = volume_form_intrinsic(edge_lengths, face_subface)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_verts, "sum")
    return sp.diag(vertex_areas)


def mass_intrinsic_inv(
    edge_lengths: torch.Tensor,
    faces: torch.Tensor,
    face_subface: torch.Tensor,
    n_verts: int,
) -> sp.SparseTensor:
    """FEM lumped inverse mass matrix with barycentric dual volumes, from edge lengths.

    The finite element `V` times `V` mass matrix where each diagonal entry corresponds
    to one over the volume of the dual element at that vertex. It uses the barycentric
    dual cell volume.

    This function computes it from edge lengths alone, meaning we do not have to have an
    embedding of the mesh.

    Warning:
        You rarely want to use this function. Most often when solving linear systems,
        you will want to call `iskra.fem.mass_intrinsic()` and then premultiply the
        right-hand side with the mass matrix.

    See Also:
        To obtain the `face_subface` tensor, use `iskra.topology.get_subfaces()`.

    Args:
        edge_lengths (Tensor[Float, [E]]): Lengths of mesh edges.
        faces (Tensor[Int64, [F, FV]]): Face index.
        face_subface (Tensor[Int64, [F, FS]]): Face-to-subface index.
        n_verts (int): Number of mesh vertices.

    Returns:
        SparseTensor[Float, [V, V]]: Diagonal sparse COO inverse mass matrix.
    """
    area = volume_form_intrinsic(edge_lengths, face_subface)
    n_corners = faces.shape[-1]
    vertex_areas = reduce_on_subface(area / n_corners, faces, n_verts, "sum")
    return sp.diag(1 / vertex_areas)


def grad_edge(
    verts: torch.Tensor, edges: torch.Tensor, eps: float | None = None
) -> sp.SparseTensor:
    """Finite element gradient matrix for edges.

    Given a line segment mesh, the function computes the finite element gradient matrix
    for a piecewise linear hat function basis.

    Args:
        verts (Tensor[Float, [V, Dim]] | Tensor[Float, [V]]): Vertex positions.
        edges (Tensor[Int64, [E, 2]]): Edge indices.
        eps (float | None): Zero out entries at edges with lengths smaller
            than eps. `None` means no zeroing.

    Returns:
        SparseTensor[Float, [Dim * E, V]]: Finite element gradient matrix.
    """
    if edges.shape[-1] != 2:
        raise ValueError(
            f"grad_edges implemented only for edges, got edges.shape={edges.shape}."
        )
    if eps is None:
        eps = -float("inf")
    if verts.ndim == 1:
        verts = verts[:, None]

    n_faces, n_verts, dim = edges.shape[0], verts.shape[0], verts.shape[-1]

    line_segments = face_index(verts, edges)  # E, 2, Dim
    edge_vecs = line_segments[..., 1, :] - line_segments[..., 0, :]  # E, Dim
    length_sq = (edge_vecs * edge_vecs).sum(-1, keepdim=True)  # E, 1
    valid_edges = length_sq > eps
    inv_length_sq = torch.zeros_like(length_sq)  # E, 1
    inv_length_sq[valid_edges] = 1.0 / length_sq[valid_edges]
    values = inv_length_sq * edge_vecs  # E, Dim
    values = torch.cat([-values, values])  # 2E, Dim

    idx_i = torch.cat(2 * [torch.arange(edges.shape[0], device=verts.device)])
    idx_j = edges.mT.flatten()
    idx = torch.stack([idx_i, idx_j])

    grad_list = []
    for dim_i in range(dim):
        grad_dim_i = sp.coo_tensor(idx, values[..., dim_i], size=[n_faces, n_verts])
        grad_list.append(grad_dim_i)
    return sp.cat(grad_list, -1).reshape(dim * n_faces, n_verts)


def grad_triangle(
    verts: torch.Tensor, faces: torch.Tensor, eps: float | None = None
) -> sp.SparseTensor:
    """Finite element gradient matrix for 2D and 3D triangles.

    Given a triangle mesh in 2D or 3D, the function computes the finite element
    gradient matrix for a piecewise linear hat function basis.

    Example:
        To compute the gradients of a 2D function uv defined on `V` over a 3D mesh,
        we can do the following:
        ```
        uv = torch.randn([verts.shape[0], 2])
        grad = grad_triangle(verts, faces)
        grad_uv = (grad @ uv)  # [3F, 2]
        grad_uv = grad_uv.reshape(faces.shape[0], 3, 2)  # [F, 3, 2]
        ```
        where the columns of `grad_uv` represent the gradients of `u` and `v`.


    Args:
        verts (Tensor[Float, [V, Dim]]): Vertices, where `Dim` is either 2 or 3.
        faces (Tensor[Int64, [F, 3]]): Triangle face index.
        eps (float | None): Zero out entries at triangles with areas smaller
            than eps. `None` means no zeroing.

    Returns:
        SparseTensor[Float, [Dim * F, V]]: Sparse COO finite element gradient matrix.
    """
    if faces.shape[-1] != 3:
        raise ValueError(
            "grad_triangle implemented only for triangles, "
            f"got faces.shape={faces.shape}."
        )

    n_faces, n_verts, dim = faces.shape[0], verts.shape[0], verts.shape[-1]

    if verts.shape[-1] == 2:
        verts = torch.nn.functional.pad(verts, pad=(0, 1))

    if eps is None:
        eps = -float("inf")

    tris = face_index(verts, faces)
    area_normals = triangle_area_normals(tris)
    normals = torch.nn.functional.normalize(area_normals, dim=-1)
    double_area = 2.0 * torch.linalg.vector_norm(area_normals, dim=-1)
    valid_tris = double_area > eps
    inv_area = torch.zeros_like(double_area)
    inv_area[valid_tris] = 1.0 / double_area[valid_tris]

    opposite_verts = face_to_subface_idcs(2, 1)
    values_list = []
    for i in range(3):
        opposite_edges = tris[:, opposite_verts[i], :]
        edge_vecs = opposite_edges[:, 1, :] - opposite_edges[:, 0, :]
        edge_vecs_rot = torch.linalg.cross(normals, edge_vecs, dim=-1)
        grad_i = inv_area[..., None] * edge_vecs_rot
        values_list.append(grad_i)
    values = torch.cat(values_list)
    values = values[..., :dim]

    idx_i = torch.arange(faces.shape[0], dtype=torch.int64, device=verts.device)
    idx_i = torch.cat(3 * [idx_i])
    idx_j = faces.mT.flatten()
    idx = torch.stack([idx_i, idx_j])

    grad_list = []
    for dim_i in range(dim):
        grad_dim_i = sp.coo_tensor(idx, values[..., dim_i], size=[n_faces, n_verts])
        grad_list.append(grad_dim_i)
    return sp.cat(grad_list, -1).reshape(dim * n_faces, n_verts)


def grad_tetrahedron(
    verts: torch.Tensor, faces: torch.Tensor, eps: float | None = None
) -> sp.SparseTensor:
    """Finite element gradient matrix tetrahedra in 3D.

    Given a tetrahedral mesh in 3D, the function computes the finite element
    gradient matrix for a piecewise linear hat function basis.

    Example:
        To compute the gradients of a 3D function uvw defined on `V` over a 3D mesh,
        we can do the following:
        ```
        uvw = torch.randn([verts.shape[0], 3])
        grad = grad_tetrahedron(verts, faces)
        grad_uvw = (grad @ uvw)  # [3F, 3]
        grad_uvw = grad_uvw.reshape(faces.shape[0], 3, 3)  # [F, 3, 3]
        ```
        where the columns of `grad_uvw` represent the gradients of `u`, `v`, and `w`.


    Args:
        verts (Tensor[Float, [V, 3]]): Vertex positions.
        faces (Tensor[Int64, [F, 4]]): Tetrahedral face index.
        eps (float | None): Zero out entries at tets with volume smaller
            than eps. `None` means no zeroing.

    Returns:
        SparseTensor[Float, [3 * F, V]]: Sparse COO finite element gradient matrix.
    """
    if faces.shape[-1] != 4:
        raise ValueError(
            "grad_tetrahedron implemented only for tetrahedra, "
            f"got faces.shape={faces.shape}."
        )
    if eps is None:
        eps = -float("inf")
    n_faces, n_verts, dim = faces.shape[0], verts.shape[0], verts.shape[-1]

    tets = face_index(verts, faces)
    tri_faces, tet_tri_faces, tet_tri_sign = get_subfaces(faces, 2)
    tris = face_index(verts, tri_faces)
    signed_volume = signed_tetrahedron_volumes(tets)
    valid_tris = signed_volume.abs() > eps
    inv_volume = torch.zeros_like(signed_volume)
    inv_volume[valid_tris] = 1.0 / (3.0 * signed_volume[valid_tris])

    area_normals = triangle_area_normals(tris)
    tet_tri_area_normals = face_index(area_normals, tet_tri_faces)
    values = (
        -inv_volume[..., None, None] * tet_tri_sign[..., None] * tet_tri_area_normals
    )
    values = values.transpose(-3, -2).flatten(-3, -2)

    idx_i = torch.arange(faces.shape[0], dtype=torch.int64, device=verts.device)
    idx_i = torch.cat(4 * [idx_i])
    idx_j = faces.mT.flatten()
    idx = torch.stack([idx_i, idx_j])

    grad_list = []
    for dim_i in range(dim):
        grad_dim_i = sp.coo_tensor(idx, values[..., dim_i], size=[n_faces, n_verts])
        grad_list.append(grad_dim_i)
    return sp.cat(grad_list, -1).reshape(dim * n_faces, n_verts)


def grad(
    verts: torch.Tensor, faces: torch.Tensor, eps: float | None = None
) -> sp.SparseTensor:
    """Finite element gradient matrix.

    Given a line segment, triangle or tetrahedral mesh, the function computes the
    finite element gradient matrix for a piecewise linear hat function basis.

    Args:
        verts (Tensor[Float, [V, Dim]]): Vertex positions.
        faces (Tensor[Int64, [F, FV]]): Face indices.
        eps (float | None): Zero out entries at tets with volume smaller
            than eps. `None` means no zeroing.

    Returns:
        SparseTensor[Float, [Dim * F, V]]: Sparse COO finite element gradient matrix.
    """
    if faces.shape[-1] == 2:
        grads = grad_edge(verts, faces, eps=eps)
    elif faces.shape[-1] == 3:
        grads = grad_triangle(verts, faces, eps=eps)
    elif faces.shape[-1] == 4:
        grads = grad_tetrahedron(verts, faces, eps=eps)
    else:
        raise ValueError(f"Gradients not implemented for dim = {verts.shape[-1]}")
    return grads


def grad_to_div(g: sp.SparseTensor, volumes: torch.Tensor) -> sp.SparseTensor:
    """Converts a finite element gradient matrix into the divergence matrix.

    See Also:
        `iskra.fem.grad()`, `iskra.fem.div()`, `iskra.geometry.volume_form`.

    Args:
        g (SparseTensor[Float, [Dim * F, V]]): Finite element gradient matrix.
        volumes (Tensor[Float, [F]]): Generalized face volumes,
            i.e., edge lengths, triangle areas, or tet volumes.

    Returns:
        SparseTensor[Float, [V, Dim * F]]: The finite element divergence operator
            matrix.
    """
    g_dim = g.shape[0] // volumes.shape[-1]
    volumes = torch.stack(g_dim * [volumes], -1).flatten(-2, -1)
    return sp.mul(volumes[..., None, :], g.mT)


def div(
    verts: torch.Tensor, faces: torch.Tensor, g: sp.SparseTensor | None = None
) -> sp.SparseTensor:
    """Computes the finite element divergence matrix.

    Internally computes the generalized volumes of the finite elements and the
    gradient matrix (if not provided) and calls `iskra.fem.grad_to_div()`.
    If you already have the gradient matrix and element volumes computed, use
    `iskra.fem.grad_to_div`.

    See Also:
        `iskra.fem.grad()`, `iskra.fem.grad_to_div()`, `iskra.geometry.volume_form`.

    Args:
        verts (Tensor[Float, [V, Dim]]): Vertex positions.
        faces (Tensor[Int64, [F, FV]]): Face indices.
        g (SparseTensor[Float, [Dim * F, V]] | None): Optional finite element gradient
            matrix, if the user happens to have it handy.

    Returns:
        SparseTensor[Float, [V, Dim * F]]: The finite element divergence operator
            matrix.
    """
    volumes = volume_form(face_index(verts, faces))
    if g is None:
        g = grad(verts, faces)
    return grad_to_div(g, volumes)


def laplacian(verts: torch.Tensor, faces: torch.Tensor) -> sp.SparseTensor:
    """Computes the *integrated* finite element Laplacian.

    The function is a wrapper around `d @ g` where `g` is the gradient operator and
    `d` the divergence. To convert the integrated quantity into a pointwise one, use the
    mass matrix (or its inverse) provided by the FEM module.

    See Also:
        `iskra.fem.grad()`, `iskra.fem.div()`, `iskra.fem.mass()`,
        `iskra.fem.mass_inv()`, `iskra.fem.mass_intrinsic()`,
        `iskra.fem.mass_intrinsic_inv()`.

    Args:
        verts (Tensor[Float, [V, Dim]]): Vertex positions.
        faces (Tensor[Int64, [F, FV]]): Face index.

    Returns:
        SparseTensor[Float, [V, V]]: Integrated Laplacian as a sparse COO matrix.
    """
    g = grad(verts, faces)
    d = div(verts, faces, g=g)
    return d @ g

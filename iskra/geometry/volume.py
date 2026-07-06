# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.normals import triangle_area_normals


def edge_lengths(edges: torch.Tensor) -> torch.Tensor:
    """Computes the lengths of a list of edges.

    Args:
        edges (Tensor[Float, [Bs, 2, Dim]]): Line segments tensor s.t. second to
            last dimension represent end points, last represents coordinates.

    Returns:
        Tensor[Float, [Bs]]: Line segment lengths.
    """
    edge_dir = edges[..., 1, :] - edges[..., 0, :]
    length: torch.Tensor = torch.linalg.vector_norm(edge_dir, dim=-1)
    return length


def triangle_areas(triangles: torch.Tensor) -> torch.Tensor:
    """Computes the areas of a list of triangles.

    Args:
        triangles (Tensor[Float, [Bs, 3, 2 | 3]]): Triangles tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Returns:
        Tensor[Float, [Bs]]: Triangle areas.
    """
    area_normals = triangle_area_normals(triangles)
    areas: torch.Tensor = torch.linalg.vector_norm(area_normals, dim=-1)
    return areas


def _scalar_triple_product(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    return torch.sum(a * torch.linalg.cross(b, c), dim=-1, keepdim=True)


def tetrahedron_volumes(tets: torch.Tensor) -> torch.Tensor:
    """Computes the volumes of a list of tetrahedra in 3D.

    Args:
        tets (Tensor[Float, [Bs, 4, 3]]): Tetrahedra tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Returns:
        Tensor[Float, [Bs]]: Tetrahedra volumes.
    """
    batch_shape = tets.shape[:-2]
    assert tets.shape[-2] == 4
    assert tets.shape[-1] == 3
    tets = tets.reshape(-1, 4, 3)
    edge_dirs = tets[..., 1:, :] - tets[..., 0:1, :]
    volume = _scalar_triple_product(
        edge_dirs[..., 0, :],
        edge_dirs[..., 1, :],
        edge_dirs[..., 2, :],
    )
    return 1 / 6 * volume.reshape(*batch_shape)


def volume_form(simplices: torch.Tensor) -> torch.Tensor:
    """Computes the generalized volumes of a list of simplices.

    This is a convenience function that checks the number of corners in the simplex
    and calls into one of `edge_lengths`, `triangle_areas`, or `tetrahedron_volumes`.

    Args:
        simplices (Tensor[Float, [Bs, S, Dim]]): Simplices tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Returns:
        Tensor[Float, [Bs]]: Simplices generalized volumes.
    """
    assert simplices.ndim >= 2

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return tetrahedron_volumes(simplices)
    elif n_simplex_verts == 3:
        return triangle_areas(simplices)
    elif n_simplex_verts == 2:
        return edge_lengths(simplices)
    elif n_simplex_verts == 1:
        return torch.ones(simplices.shape[:-1], device=simplices.device)
    else:
        raise NotImplementedError(
            "volume_form only supports vertices, edges, triangles, and tetrahedra."
        )


def triangle_areas_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    """Computes the areas of a list of triangles from edge lengths alone.

    Args:
        edge_lengths (Tensor[Float, [E]]): Tensor of edge lengths where Edges is
            the number of edges in the mesh.
        face_to_edge (Tensor[Float, [Tris, 3]]): Indexes into `edge_lengths`.
            Usually obtained using `iskra.topology.get_subfaces`.

    Returns:
        Tensor[Float, [Tris]]: Triangle areas.
    """
    face_edge_lengths = edge_lengths[face_to_edge.flatten()].reshape(
        *face_to_edge.shape, *edge_lengths.shape[1:]
    )
    semiperimeters = 0.5 * face_edge_lengths.sum(-1, keepdim=True)
    areas = torch.sqrt(
        semiperimeters[:, 0] * torch.prod(semiperimeters - face_edge_lengths, dim=-1)
    )
    return areas


def tetrahedron_volumes_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    """Computes the volume of tetrahedra from edge lengths alone.

    Implements [https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron](https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron).

    Args:
        edge_lengths (Tensor[Float, [Edges]]): Tensor of edge lengths where Edges is
            the number of edges in the mesh.
        face_to_edge (Tensor[Float, [Tets, 6]]): Indexes into `edge_lengths`.
            Usually obtained using `iskra.topology.get_subfaces`.

    Returns:
        Tensor[Float, [Tets]]: Tetrahedra volumes.
    """
    face_edge_lengths = edge_lengths[face_to_edge.flatten()].reshape(
        *face_to_edge.shape, *edge_lengths.shape[1:]
    )

    # here be dragons, truly:
    u0 = face_edge_lengths[:, 0]
    v0 = face_edge_lengths[:, 1]
    w0 = face_edge_lengths[:, 2]
    u = face_edge_lengths[:, 3]
    v = face_edge_lengths[:, 4]
    w = face_edge_lengths[:, 5]

    x0 = (w - u0 + v) * (u0 + v + w)
    x = (u0 - v + w) * (v - w + u0)
    y0 = (u - v0 + w) * (v0 + w + u)
    y = (v0 - w + u) * (w - u + v0)
    z0 = (v - w0 + u) * (w0 + u + v)
    z = (w0 - u + v) * (u - v + w0)

    a = torch.sqrt(x * y0 * z0)
    b = torch.sqrt(y * z0 * x0)
    c = torch.sqrt(z * x0 * y0)
    d = torch.sqrt(x * y * z)

    p = -a + b + c + d
    q = a - b + c + d
    r = a + b - c + d
    s = a + b + c - d

    volumes = torch.sqrt(p * q * r * s) / (192 * u * v * w)
    return volumes


def volume_form_intrinsic(
    edge_lengths: torch.Tensor, face_to_edge: torch.Tensor
) -> torch.Tensor:
    """Computes the generalized volume of simplices from edge lengths alone.

    This is a convenience function that checks the number of corners in the simplex
    and is returns the identity in case of edges, and calls either
    `triangle_areas`, or `tetrahedron_volumes` for triangles and tetrahedra resp.

    Args:
        edge_lengths (Tensor[Float, [Edges]]): Tensor of edge lengths where Edges is
            the number of edges in the mesh.
        face_to_edge (Tensor[Float, [S, 1 | 3 | 6]]): Indexes into `edge_lengths`.
            Usually obtained using `iskra.topology.get_subfaces`.

    Returns:
        Tensor[Float, [S]]: Simplices generlized volumes.
    """
    n_simplex_verts = face_to_edge.shape[-1]
    if n_simplex_verts == 6:
        return tetrahedron_volumes_intrinsic(edge_lengths, face_to_edge)
    elif n_simplex_verts == 3:
        return triangle_areas_intrinsic(edge_lengths, face_to_edge)
    elif n_simplex_verts == 1:
        assert edge_lengths.ndim == 2
        assert edge_lengths.shape[-1] == 1

        return edge_lengths.squeeze(-1)
    else:
        raise NotImplementedError(
            "volume_form_intrinsic only supports edges, triangles, and tetrahedra."
        )

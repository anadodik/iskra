# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry import coordinate_system
from iskra.geometry.normals import triangle_normals
from iskra.topology import face_index, get_subfaces


def complex_nth_root(z: torch.Tensor, n: int) -> torch.Tensor:
    """Compute the nth roots of a complex number z.

    Args:
        z (torch.Tensor): Tensor of any shape of complex numbers.
        n (int): Order of root to compute.

    Returns:
        torch.Tensor: `[..., n]` tensor containing the nth roots of z.
    """
    r = torch.abs(z)
    theta = torch.angle(z)

    k = torch.arange(n, dtype=torch.float32)
    root_magnitude = r[..., None] ** (1 / n)
    root_angle = (theta[..., None] + 2 * torch.pi * k) / n

    roots = root_magnitude * (torch.cos(root_angle) + 1j * torch.sin(root_angle))
    return roots


def to_intrinsic(
    v: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor
) -> torch.Tensor:
    """Projects 3D vector onto 2D basis spanned by `tangent` and `binormal`.

    !!! warning
        This function assumes that `v` is already in the plane spanned by the
        basis vectors.

    Args:
        v (torch.Tensor): `[B, 3]` tensor to be projected.
        tangents (torch.Tensor): `[B, 3]` tensor containing the first basis vector.
        binormals (torch.Tensor): `[B, 3]` tensor containing the second basis vector.

    Returns:
        torch.Tensor: `[B]` complex tensor containing the projection of `v`.
    """
    return torch.linalg.vecdot(v, tangents) + 1j * torch.linalg.vecdot(v, binormals)


def to_extrinsic(
    u: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor
) -> torch.Tensor:
    """Projects intrinsic complex directional into 3D.

    Args:
        u (torch.Tensor): `[B]` complex tensor to be projected.
        tangents (torch.Tensor): `[B, 3]` tensor containing the first basis vector.
        binormals (torch.Tensor): `[B, 3]` tensor containing the second basis vector.

    Returns:
        torch.Tensor: `[B, 3]` real tensor containing the embedding of `u`.
    """
    return u.real[:, None] * tangents + u.imag[:, None] * binormals


def to_intrinsic_n_rosy(
    v: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor, n: int
) -> torch.Tensor:
    """Projects 3D vector onto 2D basis spanned by `tangent` and `binormal`.

    !!! warning
        This function assumes that `v` is already in the plane spanned by the
        basis vectors.

    Args:
        v (torch.Tensor): `[B, 3]` tensor to be projected. It corresponds to any of the
            vectors in the N-RoSy field and the remaining ones
            are computed from the degree of symmetry.
        tangents (torch.Tensor): `[B, 3]` tensor containing the first basis vector.
        binormals (torch.Tensor): `[B, 3]` tensor containing the second basis vector.
        n (int): The degree of symmetry of the N-RoSy field.

    Returns:
        torch.Tensor: `[B]` complex tensor containing the encoding of `v`.
    """
    return to_intrinsic(v, tangents, binormals) ** n


def to_extrinsic_n_rosy(
    u: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor, n: int
) -> torch.Tensor:
    """Projects intrinsic complex N-RoSy field into 3D vectors.

    Args:
        u (torch.Tensor): `[B]` complex tensor to be projected.
        tangents (torch.Tensor): `[B, 3]` tensor containing the first basis vector.
        binormals (torch.Tensor): `[B, 3]` tensor containing the second basis vector.
        n (int): The degree of symmetry of the N-RoSy field.

    Returns:
        torch.Tensor: `[B, n, 3]` real tensor containing the embedding of `u`.
    """
    roots = complex_nth_root(u, n)
    extrinsic = torch.stack(
        [to_extrinsic(roots[..., i], tangents, binormals) for i in range(n)], -2
    )
    return extrinsic


def face_tangent_bundle(
    vertices: torch.Tensor, faces: torch.Tensor, edge_flaps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a tangent bundle for a face-based vector field.

    Args:
        vertices (torch.Tensor): `[V, D]` tensor, where V is the number of vertices
            and D is either 2 or 3.
        faces (torch.Tensor): `[F, 3]` tensor, where F is the number of triangle faces.
        edge_flaps (torch.Tensor): `[E, 2]` tensor, where E is the number of
            unique edges in the mesh. Use `iskra.edge_flaps` to obtain it.

    Returns:
        torch.Tensor: `[F, 3]` tensor of face tangents, where F is the number of faces.
        torch.Tensor: `[F, 3]` tensor of face binormals, where F is the number of faces.
        torch.Tensor: `[E]` complex tensor with the discrete Levi-Civita connection that
            transports vectors from face `edge_flaps[:, 0]` to face `edge_flaps[:, 1]`.
    """
    device = vertices.device

    # Compute an arbitrary basis for each face:
    triangles = face_index(vertices, faces)
    face_normals = triangle_normals(triangles)
    tangents, binormals = coordinate_system(face_normals)

    # Compute the connection between neighboring faces:
    edges, _, _ = get_subfaces(faces)
    connection = torch.zeros([edges.shape[0]], dtype=torch.cfloat, device=device)

    line_segments = face_index(vertices, edges)
    edge_vectors = line_segments[..., 1, :] - line_segments[..., 0, :]
    edge_vectors = torch.nn.functional.normalize(edge_vectors, p=2, dim=-1)

    # Represent non-boundary edges in the tangent bases of its neighboring faces:
    is_flap = (edge_flaps[:, 0] != -1) & (edge_flaps[:, 1] != -1)
    left_edge_proj = to_intrinsic(
        edge_vectors[is_flap],
        tangents[edge_flaps[is_flap, 0]],
        binormals[edge_flaps[is_flap, 0]],
    )
    right_edge_proj = to_intrinsic(
        edge_vectors[is_flap],
        tangents[edge_flaps[is_flap, 1]],
        binormals[edge_flaps[is_flap, 1]],
    )
    connection[is_flap] = right_edge_proj.conj() * left_edge_proj

    return tangents, binormals, connection


def transport_from_face(
    source: int,
    intrinsic: complex | torch.Tensor,
    n_faces: int,
    flaps: torch.Tensor,
    connection: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Parallelly transports of a vector from face source to all neighboring faces.

    Args:
        source (int): Index of source face.
        intrinsic (complex | torch.Tensor): Vector to be transported
            in a complex representation. Use `to_intrinsic` to project
        extrinsic vectors to intrinsic ones.
        n_faces (int): Number of faces in the mesh.
        flaps (torch.Tensor): `[E, 2]` tensor specifying the edge-to-face
            connectivity in the mesh. See `iskra.edge_flaps`.
        connection (torch.Tensor): Discrete connection. You can use
            `face_tangent_bundle` to obtain a connection from a mesh.
        n (int): The degree of symmetry of the N-RoSy field.
            Is simply 1 for vector fields.

    Returns:
        torch.Tensor: `[F]` complex tensor containing the transported vectors.
    """
    connection = connection**n
    transported = torch.zeros(
        [n_faces], dtype=torch.complex64, device=connection.device
    )
    transported[source] = intrinsic

    is_source_left = flaps[:, 0] == source
    lr_target_face = flaps[is_source_left][:, 1]
    lr_edges = (is_source_left).nonzero().flatten()

    is_source_right = flaps[:, 1] == source
    rl_target_face = flaps[is_source_right][:, 0]
    rl_edges = (is_source_right).nonzero().flatten()

    transported_rl = connection[rl_edges] * intrinsic
    transported_lr = connection[lr_edges].conj() * intrinsic
    transported[rl_target_face] = transported_rl
    transported[lr_target_face] = transported_lr
    return transported

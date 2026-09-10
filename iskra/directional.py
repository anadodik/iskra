# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from collections.abc import Iterable

import torch

import iskra.sparse as sp
from iskra.geometry import normal_coordinate_system
from iskra.geometry.normals import triangle_normals
from iskra.geometry.volume import edge_lengths, triangle_areas, triangle_areas_intrinsic
from iskra.sparse_linalg import min_quadratic_energy
from iskra.topology import face_index, get_subfaces


def complex_nth_root(z: torch.Tensor, n: int) -> torch.Tensor:
    r"""Compute the nth roots of a complex number z.

    Args:
        z (Tensor[Complex, [...]]): Complex numbers.
        n (int): Order of root to compute.

    Returns:
        Tensor[Complex, [..., n]]: The $n\text{th}$ roots of z.
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
    """Projects 3D vector onto 2D basis spanned by `tangents` and `binormals`.

    Important:
        This function assumes that `v` is already in the plane spanned by the
        basis vectors.

    Args:
        v (Tensor[Float, [B, 3]]): Vector to be projected.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.

    Returns:
        Tensor[Complex, [B]]: Intrinsic projection of `v`.
    """
    return torch.linalg.vecdot(v, tangents) + 1j * torch.linalg.vecdot(v, binormals)


def to_extrinsic(
    u: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor
) -> torch.Tensor:
    """Reconstructs 3D vector from intrinsic complex directional.

    Args:
        u (Tensor[Complex, [B]]): Directional to be embedded.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.

    Returns:
        Tensor[Float, [B, 3]]: 3D embedding of `u`.
    """
    # TODO: fix typing to address 1D vs nD u.
    is_1d = False
    if u.ndim == 1:
        is_1d = True
        u = u[:, None]
    extrinsic = (
        u.real[..., None] * tangents[..., None, :]
        + u.imag[..., None] * binormals[..., None, :]
    )
    if is_1d:
        extrinsic = extrinsic[:, 0, :]
    return extrinsic


def to_intrinsic_n_rosy(
    v: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor, n: int
) -> torch.Tensor:
    """Constructs an N-RoSy field from a 3D vector.

    Important:
        This function assumes that `v` is already in the plane spanned by the
        basis vectors.

    Args:
        v (Tensor[Float, [B, 3]]): Vector to be projected. It can be any of the vectors
            in the N-RoSy field. The rest follow from the degree of symmetry.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.
        n (int): The degree of symmetry of the N-RoSy field.

    Returns:
        Tensor[Complex, [B]]: Intrinsic N-RoSy field with `v` as one of its roots.
    """
    return to_intrinsic(v, tangents, binormals) ** n


def to_extrinsic_n_rosy(
    u: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor, n: int
) -> torch.Tensor:
    """Reconstructs 3D vectors from intrinsic complex N-RoSy field.

    Args:
        u (Tensor[Complex, [B]]): N-RoSy field to be embedded.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.
        n (int): The degree of symmetry of the N-RoSy field.

    Returns:
        Tensor[Float, [B, n, 3]]: 3D embedding of `u`.
    """
    roots = complex_nth_root(u, n)
    extrinsic = torch.stack(
        [to_extrinsic(roots[..., i], tangents, binormals) for i in range(n)], -2
    )
    return extrinsic


def to_intrinsic_frame_field(
    u: torch.Tensor, v: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor
) -> torch.Tensor:
    """Constructs a frame field from two 3D vectors.

    Important:
        This function assumes that `u` and `v` are already in the plane spanned by the
        basis vectors.

    Args:
        u (Tensor[Float, [B, 3]]): First direction defining the frame field.
        v (Tensor[Float, [B, 3]]): Second direction defining the frame field.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.

    Returns:
        Tensor[Complex, [B, 2]]: Intrinsic frame field with $u$ and $v$ as two of
            its roots (remaining ones will be $-u$ and $-v$).
    """
    u_sq = to_intrinsic(u, tangents, binormals) ** 2
    v_sq = to_intrinsic(v, tangents, binormals) ** 2
    coeff_2 = -(u_sq + v_sq)
    coeff_0 = u_sq * v_sq
    return torch.stack([coeff_2, coeff_0], -1)


def to_extrinsic_frame_field(
    uv: torch.Tensor, tangents: torch.Tensor, binormals: torch.Tensor
) -> torch.Tensor:
    """Reconstructs 3D vectors from complex frame field.

    Args:
        uv (Tensor[Complex, [B, 2]]): Frame field to be unprojected.
        tangents (Tensor[Float, [B, 3]]): First basis vector.
        binormals (Tensor[Float, [B, 3]]): Second basis vector.

    Returns:
        Tensor[Float, [B, 4, 3]]: 3D embedding of `uv`.
    """
    coeff_2 = uv[..., 0]
    coeff_0 = uv[..., 1]
    companion = torch.zeros([*uv.shape[:-1], 4, 4], dtype=uv.dtype, device=uv.device)
    companion[..., 1:, :-1] = torch.eye(3, dtype=uv.dtype, device=uv.device)
    companion[..., 0, -1] = -coeff_0
    companion[..., 2, -1] = -coeff_2
    # companion[..., 3, -1] = -1
    roots = torch.linalg.eigvals(companion)
    # roots = complex_nth_root(uv, 2)
    extrinsic = torch.stack(
        [to_extrinsic(roots[..., i], tangents, binormals) for i in range(4)], -2
    )
    return extrinsic


def face_tangent_bundle(
    vertices: torch.Tensor, faces: torch.Tensor, edge_flaps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a tangent bundle for a face-based vector field.

    Args:
        vertices (Tensor[Float, [V, Dim]]): Mesh vertices. Dim is either 2 or 3.
        faces (Tensor[Int64, [F, 3]]): Triangle faces.
        edge_flaps (Tensor[Int64, [E, 2]]): Unique edges in the mesh. Use
            `iskra.edge_flaps` to obtain it.

    Returns:
        Tensor[Float, [F, 3]]: Face tangents.
        Tensor[Float, [F, 3]]: Face binormals.
        Tensor[Complex, [E]]: Discrete Levi-Civita connection that
            transports vectors from face `edge_flaps[:, 0]` to face `edge_flaps[:, 1]`.
        Tensor[Complex, [E, 2]]: Projection operator. Given a directional in the tangent
            space of a dual edge, it transports it into the tangent spaces
            of `edge_flaps[:, 0]`, resp. `edge_flaps[:, 1]`.
    """
    device = vertices.device

    # Compute an arbitrary basis for each face:
    triangles = face_index(vertices, faces)
    face_normals = triangle_normals(triangles)
    tangents, binormals = normal_coordinate_system(face_normals)

    # Use the edge vector to compute mutual basis for neighboring faces:
    edges, _, _ = get_subfaces(faces)
    line_segments = face_index(vertices, edges)
    edge_vectors = line_segments[..., 1, :] - line_segments[..., 0, :]
    edge_vectors = torch.nn.functional.normalize(edge_vectors, p=2, dim=-1)

    connection = torch.zeros([edges.shape[0]], dtype=torch.cfloat, device=device)
    edge_proj = torch.zeros([edges.shape[0], 2], dtype=torch.cfloat, device=device)

    # Represent non-boundary edges in the tangent bases of its neighboring faces:
    is_flap = (edge_flaps[:, 0] != -1) & (edge_flaps[:, 1] != -1)
    edge_proj_0 = to_intrinsic(
        edge_vectors[is_flap],
        tangents[edge_flaps[is_flap, 0]],
        binormals[edge_flaps[is_flap, 0]],
    )
    edge_proj_1 = to_intrinsic(
        edge_vectors[is_flap],
        tangents[edge_flaps[is_flap, 1]],
        binormals[edge_flaps[is_flap, 1]],
    )

    # Compute the connection between neighboring faces:
    edge_proj[is_flap] = torch.stack([edge_proj_0, edge_proj_1], -1)
    connection[is_flap] = edge_proj_0 * edge_proj_1.conj()

    return tangents, binormals, connection, edge_proj


def transport_from_face(
    source: int,
    intrinsic: complex | torch.Tensor,
    n_faces: int,
    flaps: torch.Tensor,
    connection: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Parallelly transports of a vector from face source to all neighboring faces.

    Note:
        This function only tansports stuff within a 1-ring. It can be useful for visual
        debugging, e.g., making sure your connection works well with your directionals.

    Args:
        source (int): Index of source face.
        intrinsic (complex | Tensor[Complex, [...]]): Vector to be transported.
            Use `to_intrinsic` to project extrinsic vectors to intrinsic ones.
        n_faces (int): Number of faces in the mesh.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `iskra.edge_flaps`.
        connection (Tensor[Complex, [E]]): Discrete connection.
            You can use `face_tangent_bundle` to obtain a connection from a mesh.
        n (int): The degree of symmetry of the N-RoSy field.
            Is simply 1 for vector fields.

    Returns:
        Tensor[Complex, [F]]: Transported vectors.
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


def face_connection_d_01(
    n_faces: int, flaps: torch.Tensor, connection: torch.Tensor
) -> sp.SparseTensor:
    """Construct the face-based connection differetial for an N-RoSy field.

    Args:
        n_faces (int): Number of faces in the mesh.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `edge_flaps()`.
        connection (Tensor[Complex, [E]]): Discrete connection.
            You can use `face_tangent_bundle()` to obtain a connection from a mesh.

    Returns:
        SparseTensor[Complex, [E, F]]: Face-based connection differetial.
    """
    is_flap = (flaps[:, 0] != -1) & (flaps[:, 1] != -1)
    int_flaps = flaps[is_flap]
    int_conn = connection[is_flap]

    i = torch.cat(2 * [is_flap.nonzero().flatten()])
    j = torch.cat([int_flaps[:, 0], int_flaps[:, 1]])
    idcs = torch.stack([i, j])
    values = torch.cat([torch.full_like(int_conn, -1), int_conn])

    d_01 = sp.coo_tensor(idcs, values, size=[flaps.shape[0], n_faces])
    return d_01.coalesce()


def face_connection_mass(
    verts: torch.Tensor,
    edges: torch.Tensor,
    face_to_edge: torch.Tensor,
    flaps: torch.Tensor,
) -> sp.SparseTensor:
    """Makes the face-based connection mass matrix for an N-RoSy field via edge lengths.

    Given edge $e$ with adjacent faces $f$ and $g$, this matrix is the length of the
    primal edge over the length of the dual edge. See "Modeling n-Symmetry Vector Fields
    using Higher-Order Energies" by Brandt et al. 2018.for more information.

    Args:
        verts (Tensor[Float, [V, 3]]): Mesh vertices.
        edges (Tensor[Int64, [E, 2]]): Mesh edges.
        face_to_edge (Tensor[Int64, [F, 3]]): Face-to-edge connections.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `edge_flaps()`.

    Returns:
        SparseTensor[Complex, [E, E]]: Mass matrix for a face-based N-RoSy field.
    """
    edge_vecs = face_index(verts, edges)
    lengths = edge_lengths(edge_vecs)
    areas = triangle_areas_intrinsic(lengths, face_to_edge)
    areas = torch.where(
        (flaps[:, 0] != -1) & (flaps[:, 1] != -1),
        areas[flaps[:, 0]] + areas[flaps[:, 1]],
        0,
    )
    mass = 3 * (lengths**2) / areas
    mass = sp.diag(mass.to(dtype=torch.cfloat))
    return mass


def face_connection_laplacian(
    verts: torch.Tensor,
    faces: torch.Tensor,
    flaps: torch.Tensor,
    connection: torch.Tensor,
) -> tuple[sp.SparseTensor, sp.SparseTensor]:
    """Construct the face-based connection laplacian for an N-RoSy field.

    Args:
        verts (Tensor[Float, [V, 3]]): Mesh vertices.
        faces (Tensor[Int64, [F, 2]]): Mesh faces.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `edge_flaps()`.
        connection (Tensor[Complex, [E]]): Discrete connection. You can use
            `face_tangent_bundle()` to obtain a connection from a mesh.

    Returns:
        SparseTensor[Complex, [F, F]]: Face-based connection Laplacian.
    """
    edges, face_to_edge, _ = get_subfaces(faces)
    mass = face_connection_mass(verts, edges, face_to_edge, flaps)
    d_01 = face_connection_d_01(faces.shape[0], flaps, connection)
    laplacian = d_01.mT @ mass @ d_01
    return laplacian, mass


def smooth_n_rosy(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    flaps: torch.Tensor,
    connection: torch.Tensor,
    n: int,
    sources: int | Iterable[int] | torch.Tensor,
    intrinsic: complex | Iterable[complex] | torch.Tensor,
) -> torch.Tensor:
    """Smooths an N-RoSy field.

    Args:
        vertices (Tensor[Float, [V, Dim]]): Mesh vertices. Dim is either 2 or 3.
        faces (Tensor[Int64, [F, 3]]): Triangle faces.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `edge_flaps()`.
        connection (Tensor[Complex, [E]]): Discrete connection.
            You can use `face_tangent_bundle()` to obtain a connection from a mesh.
        n (int): The degree of symmetry of the N-RoSy field.
            Is simply 1 for vector fields.
        sources (int | Iterable[int] | Tensor[Int64, [...]]): Array of indices of
            faces with hard constraints.
        intrinsic (complex | Tensor[Complex, [...]]): Array of complex N-RoSy
            coefficients to be transported. Use `to_intrinsic_n_rosy()` to project
            extrinsic vectors to intrinsic N-RoSy ones.

    Returns:
        Tensor[Complex, [F]]: Smoothed vectors.
    """
    device = vertices.device

    if isinstance(sources, int):
        sources = torch.tensor([sources], device=device)
    elif not isinstance(sources, torch.Tensor):
        sources = torch.tensor(sources, device=device)

    if isinstance(intrinsic, complex):
        intrinsic = torch.tensor([intrinsic], device=device)
    elif not isinstance(intrinsic, torch.Tensor):
        intrinsic = torch.tensor(intrinsic, device=device)
    assert isinstance(intrinsic, torch.Tensor)

    laplacian, _ = face_connection_laplacian(vertices, faces, flaps, connection**n)
    rhs = torch.zeros([faces.shape[0]], dtype=intrinsic.dtype)
    _, transported = min_quadratic_energy(laplacian, rhs, sources, intrinsic)
    return transported


def smooth_frame_field(
    verts: torch.Tensor,
    faces: torch.Tensor,
    flaps: torch.Tensor,
    connection: torch.Tensor,
    source_idcs: int | Iterable[int] | torch.Tensor | None = None,
    source_vals: complex | Iterable[complex] | torch.Tensor | None = None,
    partial_idcs: int | Iterable[int] | torch.Tensor | None = None,
    partial_vals: complex | Iterable[complex] | torch.Tensor | None = None,
    ortho_reg_lambda: float = 0.0,
) -> torch.Tensor:
    """Smooth a frame field with sparse hard constraints.

    Smoothing energy based on the paper "Modeling n-Symmetry Vector Fields using
    Higher-Order Energies" by Brandt et al. 2018.

    Args:
        verts (Tensor[Float, [V, 2 | 3]]): Mesh vertices.
        faces (Tensor[Int64, [F, 3]]): Triangle faces.
        flaps (Tensor[Int64, [E, 2]]): Edge-to-face connectivity in the mesh.
            See `edge_flaps()`.
        connection (Tensor[Complex, [E]]): Discrete connection.
            You can use `face_tangent_bundle()` to obtain a connection from a mesh.
        source_idcs (int | Iterable[int] | Tensor[Int64, [S]]): Indices of
            faces with hard constraints.
        source_vals (tuple[complex, complex] | Tensor[Complex, [S, 2]]): Complex
            polyvector coefficients to be transported. Use `to_intrinsic_frame_field()`
            to project extrinsic vectors to intrinsic ones.
        partial_idcs (int | Iterable[int] | Tensor[Int64, [P]]): Indices of
            faces with hard partial constraints.
        partial_vals (tuple[complex, complex] | Tensor[Complex, [P]]): Complex 2-RoSy
            polyvector coefficients to be transported.  Use `to_intrinsic_frame_field()`
            with `n=2` to project extrinsic vectors to intrinsic ones.
        ortho_reg_lambda (float): Strength of an orthogonality regularizer.
            Regularization disabled when `ortho_reg_lambda=0.0`.

    Returns:
        Tensor[Complex, [F, 2]]: Smoothed vectors.
    """
    sources_exist = source_idcs is not None and source_vals is not None
    partial_exist = partial_idcs is not None and partial_vals is not None
    if not sources_exist and not partial_exist:
        raise ValueError("Must specify either partial or full hard constraints.")

    device = verts.device
    if sources_exist:
        dtype = source_vals.dtype
    else:
        dtype = partial_vals.dtype

    if isinstance(source_idcs, int):
        source_idcs = torch.tensor([source_idcs], device=device)
    elif source_idcs is None:
        source_idcs = torch.tensor([], dtype=torch.long, device=device)
    elif not isinstance(source_idcs, torch.Tensor):
        source_idcs = torch.tensor(source_idcs, device=device)

    if isinstance(source_vals, tuple):
        source_vals = torch.tensor([source_vals], device=device)
    elif source_vals is None:
        source_vals = torch.empty([0, 2], dtype=dtype, device=device)
    elif not isinstance(source_vals, torch.Tensor):
        source_vals = torch.tensor(source_vals, device=device)

    if isinstance(partial_idcs, int):
        partial_idcs = torch.tensor([partial_idcs], device=device)
    elif partial_idcs is None:
        partial_idcs = torch.tensor([], dtype=torch.long, device=device)
    elif not isinstance(partial_idcs, torch.Tensor):
        partial_idcs = torch.tensor(partial_idcs, device=device)

    if isinstance(partial_vals, complex):
        partial_vals = torch.tensor([partial_vals], device=device)
    elif partial_vals is None:
        partial_vals = torch.tensor([], dtype=dtype, device=device)
    elif not isinstance(partial_vals, torch.Tensor):
        partial_vals = torch.tensor(partial_vals, device=device)

    laplacian_2, mass_e = face_connection_laplacian(verts, faces, flaps, connection**2)
    laplacian_4, _ = face_connection_laplacian(verts, faces, flaps, connection**4)

    n_faces = faces.shape[0]
    block_laplacian = sp.cat_diag([laplacian_2, laplacian_4])
    block_rhs = torch.cat([torch.zeros([2 * n_faces], dtype=dtype)])

    partial_proj = sp.eye(2 * n_faces, dtype=dtype, device=device)
    partial_proj = sp.fill_slice(partial_proj, -1, partial_idcs, partial_idcs)
    partial_proj = sp.zero_slice(
        partial_proj, n_faces + partial_idcs, n_faces + partial_idcs
    )
    partial_proj = sp.append(
        partial_proj,
        torch.stack([n_faces + partial_idcs, partial_idcs]),
        partial_vals,
    )
    partial_proj = sp.append(
        partial_proj,
        torch.stack([partial_idcs, n_faces + partial_idcs]),
        partial_vals,
    )

    system = block_laplacian / mass_e.sum()
    if ortho_reg_lambda > 0:
        areas = triangle_areas(face_index(verts, faces)).to(dtype=laplacian_2.dtype)
        total_area = areas.sum()
        ortho_reg = sp.cat_diag([sp.diag(areas), sp.zeros([n_faces, n_faces])])
        # 3 * total_area comes from https://github.com/avaxman/Directional/blob/25738730958f0bfa68289ae628a5cd3c4b719d61/include/directional/polyvector_field.h#L192
        # 3 because (n - 1).
        system = system + 1 / (3 * total_area) * ortho_reg_lambda * ortho_reg

    system = partial_proj.adjoint() @ system @ partial_proj
    rhs = torch.cat([block_rhs])
    _, transported = min_quadratic_energy(
        system,
        rhs,
        torch.cat([source_idcs, n_faces + source_idcs, n_faces + partial_idcs]),
        torch.cat([source_vals.mT.flatten(), -torch.ones_like(partial_vals)]),
    )
    transported = partial_proj @ transported
    transported = transported.reshape(2, -1).mT
    return transported

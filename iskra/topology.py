# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import itertools
from itertools import combinations
from typing import Literal

import networkx as nx
import scipy.sparse
import torch

import iskra.sparse as sp


def face_to_subface_idcs(face_dim: int, subface_dim: int = -1) -> list[tuple[int, ...]]:
    r"""Returns canonical indices for subfaces within faces.

    When requesting subfaces one dimension lower than the faces
    (e.g., triangles from tets or edges from triangles), the function
    ensures that the subfaces are oriented correctly
    and that the $i^{\text{th}}$ subface is opposite to vertex $i$.

    Tip:
        A pattern you might need at some point is using this function to get
        the face-subfaces-vertices tensor:

        .. code-block:: python

            idcs: list[tuple[int, ...]] = face_to_subface_idcs(face_dim, subface_dim)
            half_subfaces = torch.stack([faces[:, nbh_idx] for nbh_idx in idcs], -2)

        For example using this pattern with `face_dim=2`, `subface_dim=1` would create
        a `[F, 3, 2]` tensor with the "triangle to half-edge vertices" relationship.
        Ideally, this should only be necessary in rare occasions.

    Caution:
        The dimension of a face is one less than the number of vertices in the face,
        e.g., edges are 1-faces, triangles 2-faces, etc.

    Args:
        face_dim (int): Intrinsic dimension of face to be indexed into.
        subface_dim (int): Intrinsic dimension of desired subface.
            Passing a negative value makes the subface dimension relative to
            the face dimension: on a k-dimensional mesh, passing `subface_dim=-1`
            asks for (k-1)-dimensional simplices.

    Returns:
        list[tuple[int, ...]]: List of the indices used to get each subface.

    Example:
        .. csv-table::
            :header: "Description", "`face_dim`", "`subface_dim`", "`idcs`"
            :widths: 28, 8, 10, 40

            "tet → triangles (oriented/opposite vertex $i$ convention)", "`3`", "`2` (default)", "`[(1,2,3), (0,3,2), (0,1,3), (0,2,1)]`"
            "tet → edges (`Heron's formula / opposite edge $i$ <https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron>`_)", "`3`", "`1`", "`[(0,1), (1,2), (2,0), (2,3), (0,3), (1,3)]`"
            "triangle → edges (oriented/opposite vertex $i$ convention)", "`2`", "`1` (default)", "`[(1,2), (2,0), (0,1)]`"
            "triangle → vertices", "`2`", "`0`", "`[(0,), (1,), (2,)]`"
            "edge → vertices (opposite vertex $i$ convention)", "`1`", "`0` (default)", "`[(1,), (0,)]`"
            "other dims (fallback)", "`d`", "`k`", "`combinations(range(d+1), k+1)`"
    """
    if subface_dim < 0:
        subface_dim = face_dim + subface_dim

    idcs: list[tuple[int, ...]]
    if face_dim == 3 and subface_dim == 2:
        idcs = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    elif face_dim == 3 and subface_dim == 1:
        # The edge ordering is s.t. UVW form a triangle,
        # and U is opposite to u, V to v and W to w.
        # U = (0, 1), V = (1, 2), W = (2, 0)
        # u = (2, 3), v = (0, 3), w = (1, 3)
        # i.e., it follows the convention of [https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron](https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron).
        idcs = [(0, 1), (1, 2), (2, 0), (2, 3), (0, 3), (1, 3)]
    elif face_dim == 2 and subface_dim == 1:
        idcs = [(1, 2), (2, 0), (0, 1)]
    elif face_dim == 2 and subface_dim == 0:
        idcs = [(0,), (1,), (2,)]
    elif face_dim == 1 and subface_dim == 0:
        idcs = [(1,), (0,)]
    else:
        idcs = list(combinations(range(face_dim + 1), subface_dim + 1))
    return idcs


def simplex_parity(faces: torch.Tensor) -> torch.Tensor:
    r"""Parity of each simplex's vertex ordering relative to sorted order.

    Treats the vertex indices of each simplex as a permutation of their
    sorted values and returns the `parity of that permutation
    <https://en.wikipedia.org/wiki/Parity_of_a_permutation>`_: $0$ if even
    (same orientation as ascending index order), $1$ if odd (opposite).
    Implemented by selection-sorting each simplex into ascending vertex
    order and counting swaps modulo $2$. The number of transpositions is
    not unique, but its parity is.

    Tip:
        `get_subfaces()` maps this parity to the orientation signs
        $\\{+1, -1\\}$ used in the face-subface hierarchy.

    Caution:
        Vertex indices within each simplex are assumed distinct.

    Args:
        faces (Tensor[Int64, [Bs, F, FV]]): Face-vertex indices. Any number
            of leading batch dimensions is allowed.

    Returns:
        Tensor[Int64, [Bs, F]]: $0$ for even parity, $1$ for odd parity.
    """
    faces = faces.clone()
    transpositions = torch.zeros_like(faces[..., 0])
    for i in range(faces.shape[-1] - 1):
        min_i = i + faces[..., i:].argmin(-1)
        # Swap smallest and current:
        smallest = torch.gather(faces, -1, min_i[..., None])
        torch.scatter(faces, -1, min_i[..., None], faces[..., i : i + 1])
        faces[..., i] = smallest[..., 0]

        # If swapped, increment number of transpositions:
        transpositions += (min_i > i).to(torch.int64)
    transpositions = transpositions % 2
    return transpositions


def get_subfaces(
    faces: torch.Tensor, subface_dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Finds all `subface_dim`-dimensional subfaces contained in a set of faces.

    The function constructs one level of the face-subface hierarchy via the
    three tensors connecting faces to subfaces: subface-vertex, face-subface,
    and subface-orientation.
    It assigns each subface element a unique index.
    `subface_vertex[i, 0]` is the index of the first vertex of subface $i$;
    thus, $i$ is the unique index assigned to the subface with those vertices.
    `face_subface[j, 0]` is said unique index of the first subface in face $j$.
    Now, notice that subface $i$ has a canonical orientation assigned to it via
    the vertex ordering in `subface_vertex`, which may be flipped when the subface
    appears as a part of face $j$.
    Therefore, `subface_orientation[j, 0]` tells us whether the subface at
    `face_subface[j, 0]` is flipped (-1) or not (+1) as it appears in face $j$.

    Hint:
        The function obtains the list of unique subfaces by getting all of
        the half-subface vertices via `face_to_subface_idcs()`, sorting the
        vertex indices and then picking out the unique ones.
        It finds the orientation using `simplex_parity()`.

    Tip:
        This function is one of the three building blocks of `iskra`'s
        scatter-gather framework, together with `iskra.topology.reduce_on_subface()`
        which performs the scatter-reduce operation and `iskra.topology.face_index()`,
        which performs the gather operation.

        This function constructs the face hierarchy.

        See :doc:`/guide/scatter-gather` for an explanation of `iskra`'s
        tensor-based scatter-gather framework.

    Caution:
        The dimension of a face is one less than the number of vertices in the face,
        e.g., edges are 1-faces, triangles 2-faces, etc.

    Args:
        faces (Tensor[Int64, [F, FV]]): Face-vertex indices.
        subface_dim (int): The dimension of the requested subfaces.
            Passing a negative value makes the subface dimension relative to
            the face dimension: on a k-dimensional mesh, passing `subface_dim=-1`
            asks for (k-1)-dimensional simplices.

    Returns:
        subface_vertex (Tensor[Int64, [S, SV]]): Subface-vertex indices.
        face_subface (Tensor[Int64, [F, FS]]): Face-subface indices.
        subface_orientation (Tensor[Float, [F, FS]]): Sign (-1 or +1) signaling a subface in the
            face-subface indices is flipped *with regards to its canonical orientation*
            as dictated by the subface-vertex indices. The sign for vertex subfaces is
            *always* +1, as they can only have one canonical orientation.
            This is a different notion of orientation from DEC's `d_{0, 1}` operator.
    """
    if faces.ndim != 2:
        raise ValueError(
            "faces must be of shape [n_faces, n_corners], "
            f"but face.shape is {faces.shape}."
        )
    face_dim = faces.shape[-1] - 1
    if subface_dim != -1 and face_dim < subface_dim:
        raise ValueError(
            f"Cannot find a {subface_dim}-subsimplex of a {face_dim}-simplex."
        )

    if face_dim == subface_dim:
        subfaces = faces.clone()[..., None, :]
        n_subfaces = 1
    else:
        idcs: list[tuple[int, ...]] = face_to_subface_idcs(face_dim, subface_dim)
        subfaces = torch.stack([faces[:, nbh_idx] for nbh_idx in idcs], -2)
        n_subfaces = len(idcs)

    # Is simplex pairity really what we want here?
    if face_dim == 1:  # whyyyyyyyy is this necessary?!?!?!?!!?
        ones = torch.ones([faces.shape[0]], device=faces.device)
        subface_sign = torch.stack([ones, -ones], -1)
    else:
        subface_flipped = simplex_parity(subfaces)
        subface_sign = torch.where(subface_flipped.bool(), -1.0, 1.0)
        subface_sign = subface_sign.reshape(-1, n_subfaces)

    subfaces = torch.flatten(subfaces, -3, -2)
    subfaces, _ = torch.sort(subfaces, -1)
    subfaces, face_to_subface = torch.unique(subfaces, dim=-2, return_inverse=True)

    face_to_subface = face_to_subface.reshape(-1, n_subfaces)
    return subfaces, face_to_subface, subface_sign


def edge_flaps(faces: torch.Tensor) -> torch.Tensor:
    """Returns a tensor denoting the right and left faces of an edge.

    Computes a tensor `flaps`, such that `flaps[i, 0]` is the index of the left face
    and `edge_flaps[i, 1]` the index of the right face containing edge $i$.
    If an edge is a boundary edge and only has a triangle on one of its sides,
    the other side will be equal to -1.

    Caution:
        Assumes a manifold mesh.

    Args:
        faces (Tensor[Int64, [F, 3]]): Triangle faces.

    Returns:
        flaps (Tensor[Int64, [E, 2]]): The edge flaps tensor.
    """
    # TODO: rename to dual edges.
    device = faces.device

    idcs: list[tuple[int, ...]] = face_to_subface_idcs(2, 1)
    subsimplex_list = [faces[:, nbh_idx] for nbh_idx in idcs]
    face_half_edge_vert = torch.stack(subsimplex_list, -2)

    edges = torch.flatten(face_half_edge_vert, -3, -2)
    edges, _ = torch.sort(edges, -1)

    face_edge_vert = edges.reshape(face_half_edge_vert.shape)
    same = (face_half_edge_vert == face_edge_vert).all(-1)  # F x 3
    flipped = (face_half_edge_vert == torch.flip(face_edge_vert, (-1,))).all(-1)

    edges, face_edge = torch.unique(edges, dim=-2, return_inverse=True)
    face_edge = face_edge.reshape(-1, 3)

    edge_flaps = torch.full([edges.shape[0], 2], -1, device=device)

    face_idcs = torch.arange(faces.shape[0], device=device)
    for v_i in range(3):
        edge_flaps[face_edge[same[:, v_i], v_i], 0] = face_idcs[same[:, v_i]]
        edge_flaps[face_edge[flipped[:, v_i], v_i], 1] = face_idcs[flipped[:, v_i]]
    return edge_flaps


def assemble_incidence_matrix(
    n_faces: int,
    n_subfaces: int,
    face_to_subface: torch.Tensor,
    subface_sign: torch.Tensor,
    signed: bool = False,
) -> sp.SparseTensor:
    """Assembles face-subface relationships into a matrix.

    Returns a sparse `[F, S]` matrix where a non-zero entry at $(i, j)$ indicates
    that face $i$ contains subface $j$. The entry is $1$ if `signed=False` and is
    equal to the orientation of the subface $j$ as seen from face $i$
    (see `get_faces()`) if `signed=True`.
    The inputs needed for this function usually come from `get_subfaces()`, or you
    can use the high-level wrapper `incidence_matrix()`.

    Args:
        n_faces (int): Number of faces (`F`).
        n_subfaces (int): Number of subfaces (`S`).
        face_to_subface (Tensor[Int64, [F, FS]]): Face-subface matrix.
        subface_sign (Tensor[Float, [F, FS]]): Subface orientation matrix.
        signed (bool): Whether the matrix should be signed or not. Defaults to False.

    Returns:
        SparseTensor[Float, [F, S]]: Incidence matrix.
    """
    device = face_to_subface.device
    n_subfaces_per_face = face_to_subface.shape[-1]
    i = torch.cat(n_subfaces_per_face * [torch.arange(n_faces, device=device)])
    j = face_to_subface.mT.flatten()
    idcs = torch.stack([i, j])
    if signed:
        values = subface_sign.mT.flatten()
    else:
        values = torch.ones_like(subface_sign.mT.flatten())
    return sp.coo_tensor(idcs, values, [n_faces, n_subfaces]).coalesce()


def incidence_matrix(
    faces: torch.Tensor, subface_dim: int = -1, signed: bool = False
) -> sp.SparseTensor:
    """Constructs the face-subface incidence matrix.

    This function is a high-level wrapper that calls `get_subfaces()` and passes
    the outputs to `assemble_incidence_matrix()`.
    Returns a sparse `[F, S]` matrix where a non-zero entry at $(i, j)$ indicates
    that face $i$ contains subface $j$. The entry is $1$ if `signed=False` and is
    equal to the orientation of the subface $j$ as seen from face $i$
    (see `get_faces()`) if `signed=True`.

    Caution:
        The dimension of a face is one less than the number of vertices in the face,
        e.g., edges are 1-faces, triangles 2-faces, etc.

    See Also:
        `iskra.topology.assemble_incidence_matrix()`, `iskra.topology.get_subfaces()`.
        Also used in `iskra.dec.d_01()`, `iskra.dec.d_12()`, etc.

    Args:
        faces (Tensor[Int64, [F, 3]]): Face index.
        subface_dim (int): The dimension of the requested subfaces.
            Passing a negative value makes the subface dimension relative to
            the face dimension: on a k-dimensional mesh, passing `subface_dim=-1`
            asks for (k-1)-dimensional simplices.
        signed (bool): Whether the matrix should be signed or not.

    Returns:
        SparseTensor[Float, [F, S]]: Incidence matrix.
    """
    subfaces, face_to_subface, subface_sign = get_subfaces(faces, subface_dim)
    return assemble_incidence_matrix(
        faces.shape[0],
        subfaces.shape[0],
        face_to_subface,
        subface_sign,
        signed=signed,
    )


def get_vert_vert(faces: torch.Tensor) -> torch.Tensor:
    """Vertex-vertex adjacencies.

    The function returns vertex-to-vertex adjacency pairs $(i, j)$ and
    ensures both $(i, j)$ and $(j, i)$ are included.
    This function can be helpful when we want to perform operations
    over vertex one-rings.

    Tip:
        The faces can be of arbitrary dimension. Tets, triangles, and edges all work.

    Args:
        faces (Tensor[Int64, [F, FV]]): Face-vertex indices.

    Returns:
        Tensor[Int64, [2 * E, 2]]: Vertex-to-vertex adjacencies.
    """
    edges, _, _ = get_subfaces(faces, subface_dim=1)
    idx = torch.cat([edges, edges.flip(-1)], -2)
    return idx


def scatter_edge_to_vert_vert(values: torch.Tensor) -> torch.Tensor:
    """Scatters scalar edge data to vertex-vertex data.

    Under the hood, this simply duplicates scalar edge data
    so it becomes scalar vert-vert data:

    .. code-block:: python

        return torch.cat([values, values], -1)

    See Also:
        `iskra.topology.get_vert_vert()`.

    Args:
        values (Tensor[DType, [Bs, E]]): Scalar per-edge data.

    Returns:
        Tensor[DType, [Bs, 2E]]: Duplicated edge data.
    """
    return torch.cat([values, values], -1)


def vertex_adjacency_matrix(n_vertices: int, faces: torch.Tensor) -> sp.SparseTensor:
    """*Undirected* vertex-vertex adjacency matrix.

    Tip:
        The faces argument can be an arbitrary simplex. Tets, triangles, and edges all work.

    Args:
        n_vertices (int): Number of vertices in your mesh.
        faces (torch.Tensor): Tensor  representing the mesh topology with shape `[n_faces, n_face_corners]`,
            where `n_faces` is the number of faces and `n_face_corners` is the number of simplex corners,.

    Returns:
        SparseTensor[Float, []]: A sparse COO tensor of shape `[n_vertices, n_vertices]`.
            An entry is 1 if two vertices share an edge.
    """
    edges, _, _ = get_subfaces(faces, subface_dim=1)
    idx = torch.cat([edges, edges.flip(-1)]).mT
    values = torch.ones([2 * edges.shape[0]], device=faces.device)
    return sp.coo_tensor(idx, values, [n_vertices, n_vertices])


def boundary(faces: torch.Tensor) -> torch.Tensor:
    """Finds all boundary subfaces of a mesh.

    Boundary subfaces are defined as those that appear in one face only.
    A boundary subface inherits the orientation from their parent face
    (see `face_subface_idcs()` for more information on the subface vertex ordering).

    Tip:
        The faces can be of arbitrary dimension. Tets, triangles, and edges all work.

    See Also:
        `iskra.topology.ordered_boundary_vertices()`,
        `iskra.topology.ordered_boundary_edges()`,
        `iskra.topology.face_subface_idcs()`.

    Args:
        faces (Tensor[Int64, [F, FV]]): Face-vertex indices.

    Returns:
        Tensor[Int64, [BS, SV]]: Boundary subface-vertex indices, one row per
            boundary subface. `SV` is one less than `FV`.
    """
    idcs: list[tuple[int, ...]] = face_to_subface_idcs(faces.shape[-1] - 1)
    half_faces = torch.cat([faces[:, idx] for idx in idcs], 0)
    sorted_half_faces, _ = torch.sort(half_faces, dim=-1)
    _, unique_idcs, counts = torch.unique(
        sorted_half_faces, dim=0, return_inverse=True, return_counts=True
    )
    inverse_counts = counts[unique_idcs]
    return half_faces[inverse_counts == 1, :]


def connected_components(
    n_vertices: int, faces: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Finds the connected components of a mesh.

    Tip:
        The faces can be of arbitrary dimension. Tets, triangles, and edges all work.

    Warning:
        This function is executed on the CPU.

    Args:
        n_vertices (int): Number of vertices in your mesh.
        faces (Tensor[Int64, [F, FV]]): Face-vertex indices.

    Returns:
        n_components (int): Number of connected components in the mesh.
        vertex_labels (Tensor[Int64, [V]]): Integer labels signifying
            the connected component of each vertex.
        face_labels (Tensor[Int64, [F]]): Integer labels signifying
            the connected component of each face.
    """
    device = faces.device
    adjacency = vertex_adjacency_matrix(n_vertices, faces)
    labels = torch.zeros(n_vertices, device=device)
    adjacency_scipy = sp.to_scipy(adjacency)
    n_comp, labels = scipy.sparse.csgraph.connected_components(adjacency_scipy)
    labels = torch.from_numpy(labels).to(device=device, dtype=torch.long)

    # all vertices in a face must belong to the same component:
    face_labels = labels[faces[:, 0]]
    return n_comp, labels, face_labels


def ordered_boundary_edges(edges: torch.Tensor) -> list[torch.Tensor]:
    """Orders boundary edges into contiguous loops.

    Given a set of undirected edges (typically the output of `boundary()`),
    finds each connected component of the edge graph and returns the edges
    of that component in a depth-first traversal order. Each component
    therefore forms a contiguous walk along a boundary loop.

    Warning:
        This function is executed on the CPU.

    See Also:
        `iskra.topology.ordered_boundary_vertices()`,
        `iskra.topology.boundary()`

    Args:
        edges (Tensor[Int64, [E, 2]]): Edge-vertex indices, e.g. boundary
            edges from `boundary()`.

    Returns:
        list[Tensor[Int64, [Ec, 2]]]: One tensor of ordered edges per
            connected component of the edge graph. `Ec` is the number of
            edges in that component. Components with no edges are omitted.
    """
    device = edges.device
    max_vertex = -1
    if edges.numel() > 0:
        max_vertex = edges.max().cpu().item()
    graph = nx.from_scipy_sparse_array(
        scipy.sparse.coo_array(
            (
                torch.ones([edges.shape[0]], device="cpu").numpy(),
                (edges.cpu().numpy().T),
            ),
            shape=[max_vertex + 1, max_vertex + 1],
        )
    )
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    component_edges = []
    for component in components:
        ordered_edges_list = list(nx.edge_dfs(component))
        if len(ordered_edges_list) == 0:
            continue
        ordered_edges = torch.tensor(
            ordered_edges_list, dtype=torch.long, device=device
        )
        component_edges.append(ordered_edges)
    return component_edges


def ordered_boundary_vertices(edges: torch.Tensor) -> list[torch.Tensor]:
    """Orders boundary vertices into contiguous loops.

    Thin wrapper around `ordered_boundary_edges()`: for each connected
    component of the edge graph, returns the ordered sequence of vertex
    indices along that walk. Assumes that the edges belong to the boundary
    of a 2-manifold mesh.

    Warning:
        This function is executed on the CPU.

    See Also:
        `iskra.topology.ordered_boundary_edges()`,
        `iskra.topology.boundary()`

    Args:
        edges (Tensor[Int64, [E, 2]]): Edge-vertex indices, e.g. boundary
            edges from `boundary()`.

    Returns:
        list[Tensor[Int64, [Vc]]]: One tensor of ordered vertex indices
            per connected component where `Vc` is the number of boundary
            vertices of that component.
    """
    component_edges = ordered_boundary_edges(edges)
    return [ordered_edges[:, 0] for ordered_edges in component_edges]


def face_index(
    data: torch.Tensor,
    faces: torch.Tensor,
    *,
    face_ndim: int = 2,
    squeeze: bool = True,
) -> torch.Tensor:
    """Gathers subface values onto the faces that contain them.

    This function collects, i.e., gathers, data (scalar, vector, tensor, etc.)
    defined on subfaces onto the faces they belong to. So, e.g., triangle faces
    can gather 3 edge-based values or 3 vertex-based values onto themselves.
    Alternatively, one can see this function as using the subface indices
    stored in `faces` to index into `data`: given data associated with each
    subface and a tensor of subface indices, this function outputs a tensor
    that contains data entries in positions defined by the indices.

    Tip:
        This function is one of the three building blocks of `iskra`'s
        scatter-gather framework, together with `iskra.topology.get_subfaces()`
        which constructs the face hierarchy and
        `iskra.topology.reduce_on_subface()`, which performs the
        scatter-reduce operation.

        This function moves data up the face hierarchy.

        See :doc:`/guide/scatter-gather` for an explanation of `iskra`'s
        tensor-based scatter-gather framework.

    Args:
        data (Tensor[DType, [Bs, S, Ds]]): Data stored on each subface.
        faces (Tensor[Int64, [Bs, F] | [Bs, F, FS] | [Bs, F, FSs]]): Face-to-subface
            indices, possibly nested. See examples below.
        face_ndim (int): The dimensionality of the face index. By default
            the function assumes it is working with (possibly batched) face-vertex
            relations, which are stored as `2D` tensors. Hence `face_ndim=2` by default.
            If we were, e.g., gathering from vertices using a nested tet-tri-vert index
            we would specify `faces_ndim=3` and for a 1D list of vertices `face_ndim=1`.
        squeeze (bool): If faces shape is `[Bs, F, FS]` and `FS` == 1 (i.e. the list of
            faces is a 1D list of vertices), `squeeze` dictates whether the output
            will have the size-1 dimension corresponding to the face vertices
            squeezed. If faces shape is already `[Bs, F]`, `squeeze` does nothing.
            Default: `True`.

    Raises:
        ValueError: Throws an exception if `faces.ndim` is smaller than `face_ndim`.
            Everything to the left of the face index is read as the batch, so, e.g.,
            handing a 1D list of vertices to the default `face_ndim=2` would ask for
            `-1` batch dimensions. We have to specify `face_ndim=1` instead.
        ValueError: Throws an exception if the batch dimensions of `faces` and `data`
            disagree. It is `face_ndim` that decides where the batch stops and the
            face index starts, so this usually means `face_ndim` is wrong.

    Returns:
        An Tensor with the shape `[Bs, F, FSs, Ds]`.

    Example:
        .. csv-table::
            :header: "`values.shape`", "`faces.shape`", "`result.shape`", "Gathering"
            :widths: 12, 12, 14, 30

            "`[V, 2]`", "`[Tris, 3]`", "`[Tris, 3, 2]`", "2D triangle positions"
            "`[V, 3]`", "`[Tris, 3]`", "`[Tris, 3, 3]`", "3D triangle positions"
            "`[V, 3]`", "`[Tets, 4]`", "`[Tets, 4, 3]`", "3D tet positions"
            "`[V, 4]`", "`[Tets, 4]`", "`[Tets, 4, 4]`", "4D tet positions"
            "`[E, 2, 2]`", "`[Tets, 4, 3]`", "`[Tets, 4, 3, 2, 2]`", "2x2 tet-tri-edge matrices."

        Next, assume a face tensor with the size of `[8, 4]`. There is no way to know
        whether this is `8` polygons with `4` corners each or whether an batch of `8`
        1D vectors of vertex indices. This function always assumes the first, i.e., it
        assumes it is working with a face-vertex input. To change the default and have
        the function interpret `8` as a batch dimension, we have to be explicit and
        specify the face-index is 1D, i.e., face_ndim=1.
    """
    if faces.ndim < face_ndim:
        raise ValueError(
            f"faces.shape={faces.shape} ({len(faces.shape)} dims), but "
            f"face_ndim={face_ndim}. face_ndim must be bigger than number of face dims."
        )

    batch_ndim = faces.ndim - face_ndim
    data_shape = data.shape[batch_ndim + 1 :]
    data_ndim = len(data_shape)
    result_shape = faces.shape + data_shape

    face_batches = faces.shape[:batch_ndim]
    data_batches = data.shape[:batch_ndim]
    if face_batches != data_batches:
        raise ValueError(
            "Face batch dimensions do not match data batch dimensions. "
            f"This is likely due to an incorrect face_ndim (={face_ndim}) argument."
        )

    # The following is basically a fancy way of writing the following,
    # but in a way that supports batching:
    # result = data[faces.flatten(), ...].reshape(result_shape)
    index = faces.flatten(batch_ndim, -1)
    index = index[..., *((None,) * data_ndim)]
    index = index.expand(*((-1,) * (batch_ndim + 1)), *data_shape)
    result = torch.gather(data, batch_ndim, index).reshape(result_shape)

    if squeeze and face_ndim >= 2 and faces.shape[-1] == 1:
        result = result.squeeze(faces.ndim - 1)

    return result


def reduce_on_subface(
    data: torch.Tensor,
    faces: torch.Tensor,
    n_subfaces: int,
    reduce: Literal["sum", "prod", "mean", "amax", "amin"],
    *,
    data_ndim: int | None = None,
    batch_ndim: int = 0,
) -> torch.Tensor:
    """Scatter-reduce data from faces onto constitutive subfaces.

    This function distributes, i.e., scatters, data (scalar, vector, tensor,
    etc.) defined on faces to each face's subfaces.
    A naive scatter would result in each subface receiving multiple competing
    values, one for each face that contains it. Therefore, each operation is
    actually a scatter-reduce (see `reduce` argument).
    So, e.g., a triangle face can scatter its value to its 3 edges. That value
    is then, e.g., averaged onto each edge with the values of
    the other triangles that share that same edge.

    The function slices up the data tensor into three parts: `[Bs, Fs, Ds]`,
    where `Bs` is the batch shape, `Ds` are the data payload dimensions;
    `Fs` represents the "domain" of the data.
    Most commonly `Fs` is either `[F]` or `[F, FS]`.
    For example, on a triangle mesh, `Fs = [F]` implies one data payload
    per triangle, whereas `Fs = [F, 3]` implies one data payload per
    triangle-corner or triangle-side.

    Data dimensions are greedy when `data_ndim` is `None`: we assume
    `Fs = [F]` and that everything to the right of `Fs` is the data payload.
    E.g., if `data.shape = [B, F, 3, 3]` and `Fs = [F]`, we have one data
    payload per face, whereas if `Fs = [F, 3]`, we have one data payload per
    triangle-corner.

    Caution:
        The function should generalize to nested subfaces indices, i.e.,
        `Fs=[F, FSs]`. For example, it should handle data defined on a
        tetrahedron's side-triangles' corners (`Fs=[Tets, 3, 3]`), but
        this is untested! If you have a real-world example of data stored
        on subfaces of subfaces, let me know and I can look into it!

    Tip:
        This function is one of the three building blocks of `iskra`'s
        scatter-gather framework, together with `iskra.topology.get_subfaces()`
        which constructs the face hierarchy and `iskra.topology.face_index()`,
        which performs the gather operation.

        This function moves data down the face hierarchy.

        See :doc:`/guide/scatter-gather` for an explanation of `iskra`'s
        tensor-based scatter-gather framework.

    Args:
        data (Tensor[DType, [Bs, Fs, Ds]]): Data defined on mesh faces
            (`Fs=[F]`) or face-subfaces (`Fs=[F, FS]`).
        faces (Tensor[Int64, [Bs, F, FS]]): Face-subface indices.
        n_subfaces (int): Total number of subfaces in the mesh (e.g., total
            number of vertices or edges in a triangle mesh).
        reduce (Literal["sum", "prod", "mean", "amax", "amin"]): Reduction
            operation, see `torch.scatter_reduce()` for more details.
        data_ndim (int | None): Num. dimensions of the per-face
            (or per-face-subface) payload. See above for more details
            on default behavior.
        batch_ndim (int): Num. batch dimensions.

    Returns:
        Tensor[DType, [Bs, S, Ds]]: Data reduced onto the `S` subfaces,
        where `S` equals the value of the argument `n_subfaces`.

    Example:
        .. csv-table::
            :header: "`data.shape`", "`faces.shape`", "`data_ndim`", "`result.shape`", "Example: from → to"
            :widths: 14, 5, 5, 14, 30

            "`[B, F]`", "`[B, F, 3]`", "`0`", "`[B, V]`", "face scalars → vertices"
            "`[B, F, 3]`", "`[B, F, 3]`", "`0`", "`[B, V]`", "corner scalars → vertices"
            "`[B, F, 2]`", "`[B, F, 3]`", "`1` | `None`", "`[B, V, 2]`", "face 2-vectors → vertices"
            "`[B, F, 3, 2]`", "`[B, F, 3]`", "`1`", "`[B, V, 2]`", "corner 2-vectors → vertices"
            "`[B, F, 3]`", "`[B, F, 3]`", "`1` | `None`", "`[B, V, 3]`", "face normals → vertices"
            "`[B, F, 3, 3]`", "`[B, F, 3]`", "`1`", "`[B, V, 3]`", "corner normals → vertices"
            "`[B, F, 3, 3]`", "`[B, F, 3]`", "`2` | `None`", "`[B, V, 3, 3]`", "face covariances → vertices"
    """
    # Data dims is assumed to be greedy by default, i.e.,
    # we assume everything that is to the right of the face index is part
    # of the data payload.
    if data_ndim is None:
        data_ndim = data.ndim - 1 - batch_ndim
    assert data_ndim is not None  # for type checking

    flatten_scalar_dim = False
    if data_ndim == 0:
        flatten_scalar_dim = True
        data_ndim = 1
        data = data[..., None]

    data_shape = data.shape[-data_ndim:]
    batch_shape = data.shape[:batch_ndim]
    # matched_ndim, first n dims that are shared between data and faces:
    matched_ndim = data.ndim - data_ndim
    assert data.shape[:matched_ndim] == faces.shape[:matched_ndim]

    result_shape = (*batch_shape, n_subfaces, *data_shape)
    result = torch.zeros(result_shape, dtype=data.dtype, device=data.device)
    subface_gen = itertools.product(
        *(range(faces.shape[dim]) for dim in range(batch_ndim + 1, faces.ndim))
    )
    for subface_idcs in subface_gen:
        face_slice = faces[..., :, *subface_idcs]
        face_slice = face_slice[..., :, *((None,) * data_ndim)].expand(
            *((-1,) * (batch_ndim + 1)), *data_shape
        )
        data_slice = data[
            ...,
            :,
            *subface_idcs[: matched_ndim - 1 - batch_ndim],
            *((slice(None, None),) * data_ndim),
        ]

        result = result.scatter_reduce(
            batch_ndim, face_slice, data_slice, reduce=reduce
        )
    if flatten_scalar_dim:
        result = result.squeeze(-1)
    return result


def find_cliques(edges: torch.Tensor, max_d: int) -> list[torch.Tensor]:
    """Finds all cliques in a graph for all sizes up to max_d.

    Given an edge soup, this helps us find all simplices up that can be formed
    by combining the different edges that have common vertices.
    Taken from https://stackoverflow.com/questions/48081912/converting-adjacency-matrix-to-abstract-simplicial-complex.

    Args:
        edges (Tensor[Int64, [E, 2]]): Edge-vertex indices.
        max_d (int, optional): The number of vertices in the largest
            requested simplex. E.g. max_d=4 will return all possible
            simplices up to and including tetrahedra.

    Returns:
        list[torch.Tensor]: list of tensors such that the Nth tensor contains
            the simplices with N vertices.
    """
    edge_list: list[tuple[int, int]] = edges.cpu().numpy().tolist()
    edge_set = {frozenset(edge) for edge in edge_list if edge[0] != edge[1]}
    vertices = {vertex for edge in edge_set for vertex in edge}

    neighbors = {
        v: frozenset(({v} ^ e).pop() for e in edge_set if v in e) for v in vertices
    }

    simplices = [set(), [vertices], edge_set]
    shared_neighbors = {frozenset({v}): nb for v, nb in neighbors.items()}
    for _ in range(2, max_d):
        next_degree = set()  # type: ignore
        for smplx in simplices[-1]:
            # Split off random vertex
            rem = set(smplx)
            rv = rem.pop()
            rem = frozenset(rem)  # type: ignore
            # Find shared neighbors
            shrd_nb = shared_neighbors[rem] & neighbors[rv]  # type: ignore
            shared_neighbors[smplx] = shrd_nb  # type: ignore
            # Build containing simplices
            next_degree.update(smplx | {vtx} for vtx in shrd_nb)
        if not next_degree:
            break
        simplices.append(next_degree)

    simplices_tensors = []
    for simplices_list in simplices:
        simplex_idcs_list = [list(simplex) for simplex in simplices_list]
        simplex_idcs = torch.tensor(
            simplex_idcs_list, dtype=torch.long, device=edges.device
        )
        simplex_idcs = torch.sort(simplex_idcs, -1)[0]
        simplex_idcs = torch.unique(simplex_idcs, dim=0)
        simplices_tensors.append(simplex_idcs)
    return simplices_tensors

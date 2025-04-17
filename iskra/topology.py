# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from itertools import combinations
from typing import Literal

import networkx as nx
import scipy.sparse
import torch

from iskra.sparse import torch_to_scipy


def face_to_subface_idcs(face_dim: int, subface_dim: int = -1) -> list[tuple[int, ...]]:
    """Returns indices used to find subfaces of intrinsic dimension `subface_dim` within faces of dimension `face_dim`.

    Makes sure triangles/edges are oriented correctly,
    and that they are opposite to the vertex indexed by
    their position.

    Args:
        face_dim (int): _description_
        subface_dim (int): _description_

    Returns:
        list[tuple[int, ...]]: _description_
    """
    if subface_dim < 0:
        subface_dim = face_dim + subface_dim
    idcs: list[tuple[int, ...]]
    if face_dim == 3 and subface_dim == 2:
        idcs = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    if face_dim == 3 and subface_dim == 1:
        # The edge ordering is s.t. UVW form a triangle,
        # and U is opposite to u, V to v and W to w.
        # U = (0, 1), V = (1, 2), W = (2, 0)
        # u = (2, 3), v = (0, 3), w = (1, 3)
        # i.e., it follows the convention of [https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron](https://en.wikipedia.org/wiki/Heron%27s_formula#Volume_of_a_tetrahedron).
        idcs = [(0, 1), (1, 2), (2, 0), (2, 3), (0, 3), (1, 3)]
    elif face_dim == 2 and subface_dim == 1:
        idcs = [(1, 2), (2, 0), (0, 1)]
    elif face_dim == 1 and subface_dim == 0:
        idcs = [(1,), (0,)]
    else:
        idcs = list(combinations(range(face_dim + 1), face_dim))
    return idcs


def simplex_parity(faces: torch.Tensor) -> torch.Tensor:
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


def get_subfaces(faces: torch.Tensor, subface_dim: int = -1) -> torch.Tensor:
    """Finds all subsimplices of dimension d in a set of higher-dimensional faces.

    Args:
        faces (torch.Tensor): A [n_faces, simplex_dim] tensor containing the
            vertices of the higher-dimensional faces.
        subface_dim (int): The dimension of the requested simplex. Note that the simplex
            dimension is one less than the number of its vertices, e.g. edges are
            1-simplices, triangles 2-simplices, etc.

    Returns:
        torch.Tensor: A tensor containing all subsimplex indices. Shape is
            [n_simplices, n_subfaces_per_simplex, d].

        torch.Tensor: The sign says whether the subface, as it appears in the subfaces,
            is flipped _with regards to some canonical orientation of the subface!_.
            This means that sign for vertices will _always_ be +1, as they can only
            have one canonical orientation. This makes it slightly different
            from the orientation in DEC's `d_{0, 1}` operator in this case.

        [???].
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
        n_subfaces = len(idcs)
        subsimplex_list = [faces[:, nbh_idx] for nbh_idx in idcs]
        subfaces = torch.stack(subsimplex_list, -2)
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

    Args:
        faces (torch.Tensor): An `[F, 3]` tensor with triangle faces.

    Returns:
        torch.Tensor: An `[E, 2]` tensor, where E is the number of
            unique edges in the mesh. `edge_flaps[:, 0]` is the index of the left face
            and `edge_flaps[:, 1]` the index of the right face of the mesh. If an edge
            is a boundary edge and only has a triangle on one of its sides,
            the other side will be equal to -1.
    """
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
    face_dim: int,
    n_subfaces: int,
    face_to_subface: torch.Tensor,
    subface_sign: torch.Tensor,
    signed: bool = False,
) -> torch.Tensor:
    device = face_to_subface.device
    n_subfaces_per_face = face_to_subface.shape[-1]
    i = torch.cat(n_subfaces_per_face * [torch.arange(n_faces, device=device)])
    j = face_to_subface.mT.flatten()
    idcs = torch.stack([i, j])
    if signed:
        values = subface_sign.mT.flatten()
        if face_dim == 1:  # whyyyyyyyy is this necessary?!?!?!?!!?
            values = torch.cat(
                [
                    -torch.ones([n_faces], device=device),
                    torch.ones([n_faces], device=device),
                ]
            )
    else:
        values = torch.ones_like(subface_sign.mT.flatten())
    return torch.sparse_coo_tensor(idcs, values, [n_faces, n_subfaces]).coalesce()


def incidence_matrix(
    faces: torch.Tensor, subface_dim: int = -1, signed: bool = False
) -> torch.Tensor:
    n_faces, face_dim = faces.shape[0], faces.shape[-1] - 1

    subfaces, face_to_subface, subface_sign = get_subfaces(faces, subface_dim)
    return assemble_incidence_matrix(
        n_faces,
        face_dim,
        subfaces.shape[0],
        face_to_subface,
        subface_sign,
        signed=signed,
    )


def vertex_adjacency_matrix(n_vertices: int, faces: torch.Tensor) -> torch.Tensor:
    """*Undirected* vertex-vertex adjacency matrix.

    !!! tip

        The faces argument can be an arbitrary simplex. Tets, triangles, and edges all work.

    Args:
        n_vertices (int): Number of vertices in your mesh.
        faces (torch.Tensor): Tensor  representing the mesh topology with shape `[n_faces, n_face_corners]`,
            where `n_faces` is the number of faces and `n_face_corners` is the number of simplex corners,.

    Returns:
        torch.Tensor: A sparse COO tensor of shape `[n_vertices, n_vertices]`.
            An entry is 1 if two vertices share an edge.
    """
    edges, _, _ = get_subfaces(faces, subface_dim=1)
    idx = torch.cat([edges, edges.flip(-1)]).mT
    values = torch.ones([2 * edges.shape[0]], device=faces.device)
    return torch.sparse_coo_tensor(idx, values, [n_vertices, n_vertices])


def boundary(faces: torch.Tensor) -> torch.Tensor:
    idcs: list[tuple[int, ...]] = face_to_subface_idcs(faces.shape[-1] - 1)
    half_faces = torch.cat([faces[:, idx] for idx in idcs], 0)
    sorted_edges, _ = torch.sort(half_faces, dim=-1)
    _, unique_idcs, counts = torch.unique(
        sorted_edges, dim=0, return_inverse=True, return_counts=True
    )
    inverse_counts = counts[unique_idcs]
    return half_faces[inverse_counts == 1, :]


def connected_components(
    n_vertices: int, faces: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Finds the connected components of a mesh.

    !!! tip

        The faces argument can be an arbitrary simplex. Tets, triangles, and edges all work.

    Args:
        n_vertices (int): Number of vertices in your mesh.
        faces (torch.Tensor): Tensor  representing the mesh topology with shape `[n_faces, n_face_corners]`,
            where `n_faces` is the number of faces and `n_face_corners` is the number of simplex corners,.

    Returns:
        n_components (int): Number of connected components in the mesh.
        vertex_labels (torch.Tensor): A tensor of shape `[n_vertices]`
            with integer labels signifying the connected component of each vertex.
        face_labels (torch.Tensor): A tensor of shape `[n_faces]`
            with integer labels signifying the connected component of each face.
    """
    device = faces.device
    adjacency = vertex_adjacency_matrix(n_vertices, faces)
    labels = torch.zeros(n_vertices, device=device)
    adjacency_scipy = torch_to_scipy(adjacency)
    n_comp, labels = scipy.sparse.csgraph.connected_components(adjacency_scipy)
    labels = torch.from_numpy(labels).to(device=device, dtype=torch.long)

    # all vertices in a face must belong to the same component:
    face_labels = labels[faces[:, 0]]
    return n_comp, labels, face_labels


def select_linked(start_vertex: int, faces: torch.Tensor) -> torch.Tensor:
    pass


def loose_vertices(n_vertices: int, faces: torch.Tensor) -> torch.Tensor:
    pass


def flip_edges() -> torch.Tensor:
    pass


def ordered_boundary_edges(edges: torch.Tensor) -> list[torch.Tensor]:
    device = edges.device
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


def face_index(
    values: torch.Tensor, faces: torch.Tensor, squeeze: bool = True
) -> torch.Tensor:
    """Scatters vertex values according to a tensor of vertex indices.

    This function takes in a function value associated with each vertex
    and a high-dimensional tensor of vertex indices, and outputs a tensor
    that contains those function values in positions defined by the index.

    !!! example

        Some concrete examples are:

        - Scattering 2D triangle positions:
            - input tensor shapes: `[n_vertices, 2]`, `[n_triangles, 3]`
            - output tensor shape: `[n_triangles x 3 x 2]`.
        - Scattering 3D triangle positions:
            - input tensor shapes: `[n_vertices, 3]`, `[n_triangles, 3]`
            - output tensor shape: `[n_triangles x 3 x 3]`.
        - Scattering 3D tet positions:
            - input tensor shapes: `[n_vertices, 3]`, `[n_tets, 4]`
            - output tensor shape: `[n_triangles x 4 x 3]`.
        - Scattering 4D tet positions:
            - input tensor shapes: `[n_vertices, 4]`, `[n_tets, 4]`
            - output tensor shape: `[n_triangles x 4 x 4]`.
        Works with higher dimensional indices too.

    Args:
        values: A tensor of shape [n_vertices, value_dim] assigning a
            value_dim-dimensional value to each vertex.
        faces: A tensor of shape [n_faces, intrinsic_dim + 1] containing
            the n_faces faces each with dim + 1 vertices.
        squeeze: If intrinsic_dim == 0 (i.e. the list of faces is just
            a 1D list of vertices) squeeze dictates whether the output
            will have the size-1 dimension corresponding to the face vertices
            squeezed. Default: True.

    Returns:
        An Tensor with the shape [n_simplices, intrinsic_dim + 1, value_dim].
    """
    if faces.ndim == 1:
        faces = faces[:, None]

    result_shape = faces.shape + values.shape[1:]
    result = values[faces.flatten(), ...].reshape(result_shape)
    if squeeze and faces.shape[-1] == 1:
        result = result.squeeze(faces.ndim - 1)

    return result


def reduce_on_subface(
    values: torch.Tensor,
    faces: torch.Tensor,
    n_subfaces: int,
    reduce: Literal["sum", "prod", "mean", "amax", "amin"],
) -> torch.Tensor:
    """Take values defined on mesh faces and average them onto its vertices.

    Args:
        values (torch.Tensor): Values defined on the mesh faces.
        faces (torch.Tensor): Face indices.
        n_subfaces (int): Total number of subfaces in the mesh.
        reduce (Literal["sum", "prod", "mean", "amax", "amin"]): Reduction operation.

    Returns:
        torch.Tensor: _description_
    """
    assert values.shape[0] == faces.shape[0]
    assert faces.ndim == 2

    values_shape = values.shape[1:]
    scattered = torch.zeros(
        [n_subfaces, *values_shape], dtype=values.dtype, device=values.device
    )
    broadcast_faces = faces[(...,) + (None,) * len(values_shape)]
    broadcast_faces = broadcast_faces.expand(-1, -1, *values_shape)
    for i in range(faces.shape[-1]):
        scattered.scatter_reduce_(0, broadcast_faces[:, i, ...], values, reduce=reduce)
    return scattered


def find_cliques(edges: torch.Tensor, max_d: int) -> list[torch.Tensor]:
    """Finds all cliques in a graph for all sizes up to max_d.

    Given an edge soup, this helps us find all simplices up that can be formed
    by combining the different edges that have common vertices.
    Taken from https://stackoverflow.com/questions/48081912/converting-adjacency-matrix-to-abstract-simplicial-complex.

    Args:
        edges (torch.Tensor): A tensor of shape [n_edges, 2] containing
            the vertex indices that make each edge.
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

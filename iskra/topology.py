# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from itertools import combinations
from typing import Literal

import networkx as nx
import scipy.sparse
import torch


def face_to_subface_idcs(face_dim: int, subface_dim: int = -1) -> list[tuple[int, ...]]:
    """_summary_.

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
        subface_dim = face_dim - subface_dim
    idcs: list[tuple[int, ...]]
    if face_dim == 3 and subface_dim == 2:
        idcs = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
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
    face_dim = faces.shape[-1] - 1
    if subface_dim != -1 and face_dim < subface_dim:
        raise ValueError(
            f"Cannot find a {subface_dim}-subsimplex of a {face_dim}-simplex."
        )

    if face_dim == subface_dim:
        return faces

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


def incidence_matrix(
    faces: torch.Tensor, subface_dim: int = -1, signed: bool = False
) -> torch.Tensor:
    device = faces.device
    n_faces, face_dim = faces.shape[0], faces.shape[-1]

    subfaces, face_to_subface, subface_sign = get_subfaces(faces, subface_dim)
    n_subfaces = subfaces.shape[0]
    n_subfaces_per_face = face_to_subface.shape[-1]
    i = torch.cat(n_subfaces_per_face * [torch.arange(n_faces, device=device)])
    j = face_to_subface.mT.flatten()
    idcs = torch.stack([i, j])
    if signed:
        values = subface_sign.mT.flatten()
        if face_dim == 1:  # whyyyyyyyy is this necessary?!?!?!?!!?
            values = torch.cat(
                [
                    torch.ones([n_faces], device=device),
                    -torch.ones([n_faces], device=device),
                ]
            )
    else:
        values = torch.ones_like(subface_sign.mT.flatten())
    return torch.sparse_coo_tensor(idcs, values, [n_faces, n_subfaces])


def boundary(faces: torch.Tensor) -> torch.Tensor:
    idcs: list[tuple[int, ...]] = face_to_subface_idcs(faces.shape[-1] - 1)
    half_faces = torch.cat([faces[:, idx] for idx in idcs], 0)
    sorted_edges, _ = torch.sort(half_faces, dim=-1)
    _, unique_idcs, counts = torch.unique(
        sorted_edges, dim=0, return_inverse=True, return_counts=True
    )
    inverse_counts = counts[unique_idcs]
    return half_faces[inverse_counts == 1, :]


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
    Some concrete examples are:
    - Scattering 2D triangle positions:
        - input tensor shapes: [n_vertices, 2], [n_triangles, 3]
        - output tensor shape: [n_triangles x 3 x 2].
    - Scattering 3D triangle positions:
        - input tensor shapes: [n_vertices, 3], [n_triangles, 3]
        - output tensor shape: [n_triangles x 3 x 3].
    - Scattering 3D tet positions:
        - input tensor shapes: [n_vertices, 3], [n_tets, 4]
        - output tensor shape: [n_triangles x 4 x 3].
    - Scattering 4D tet positions:
        - input tensor shapes: [n_vertices, 4], [n_tets, 4]
        - output tensor shape: [n_triangles x 4 x 4].
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
    scattered = torch.zeros([n_subfaces, *values_shape], device=values.device)
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

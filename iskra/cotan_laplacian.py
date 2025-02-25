import torch
import scipy.sparse

from iskra.geometry.cotan_weights import cotan_weights


def edge_local_index(faces: torch.Tensor):
    """Return the (local) indexes of the opposite edge in each face

    Args:
        faces (torch.Tensor): #F x {1|3|4} tensor of mesh elements (edges, tri, or tet)
    """
    simplex_size = faces.shape[1]
    if simplex_size == 2:
        return torch.tensor([[0, 1]], dtype=faces.dtype)
    elif simplex_size == 3:
        return torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=faces.dtype)
    elif simplex_size == 4:
        return torch.tensor(
            [[1, 2], [2, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=faces.dtype
        )
    else:
        raise ValueError(
            f"Only accept edges (2), triangles (3), and tet (4) meshes. Input mesh simplex size ={simplex_size}"
        )


def squared_edge_lengths(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute squared lengths of edges opposite each index in a face (triangle/tet) tensor

    Args:
        vertices (torch.Tensor): #V x dim tensor of vertex position
        faces (torch.Tensor): #F x {2|3|4} tensor of mesh elements (edges, tri, or tet)

    Returns:
        torch.Tensor: #F x {1|3|6} tensor of squared edge lengths. For triangles and tet, the edge local index for each face (every row) follows the same convention in edge_local_index
    """
    edges = edge_local_index(faces)
    diff = vertices[faces[:, edges[:, 0]]] - vertices[faces[:, edges[:, 1]]]
    return (diff**2).sum(dim=2)


def edge_lengths(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute lengths of edges opposite each index in a face (triangle/tet) tensor

    Args:
        vertices (torch.Tensor): #V x dim tensor of vertex position
        faces (torch.Tensor): #F x {2|3|4} tensor of mesh elements (edges, tri, or tet)

    Returns:
        torch.Tensor: #F x {1|3|6} tensor of edge lengths. For triangles and tet, the edge local index for each face (every row) follows the same convention in edge_local_index
    """
    return torch.sqrt(squared_edge_lengths(vertices, faces))


def doublearea(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Computes twice the area of input triangles

    Args:
        vertices (torch.Tensor): #V x dim tensor of vertex position
        faces (torch.Tensor): #F x 3 tensor of mesh elements (tri)

    Returns:
        torch.Tensor: #F x 1 tensor of twice the face area
    """
    dim = vertices.shape[1]
    simplex_size = faces.shape[1]
    if simplex_size != 3:
        raise ValueError(
            f"Only accept triangles (3) meshes. Input mesh simplex size ={simplex_size}"
        )

    if dim != 2 and dim != 3:
        raise ValueError(
            f"Vertex dimension should be either 2 or 3. Input vertex dimension is ={dim}"
        )

    if dim == 3:
        c = torch.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 2]],
            vertices[faces[:, 0]] - vertices[faces[:, 2]],
            dim=1,
        )
        dblA = torch.norm(c, dim=1)

    if dim == 2:
        dblA = (vertices[faces[:, 0], 0] - vertices[faces[:, 2], 0]) * (
            vertices[faces[:, 1], 1] - vertices[faces[:, 2], 1]
        ) - (vertices[faces[:, 0], 1] - vertices[faces[:, 2], 1]) * (
            vertices[faces[:, 1], 0] - vertices[faces[:, 2], 0]
        )

    return dblA


# def face_areas(edge_len: torch.Tensor) -> torch.Tensor:
#    """Calculate a tet mesh face area using the edge length information
#
#    Args:
#        edge_len (torch.Tensor): #T x 6 tensor of tet mesh edge lengths where the index is derived from edge_local_index
#
#    Returns:
#        torch.Tensor: #T x 4 tensor of the tet's face area
#    """
#    if edge_len.shape[1] != 6:
#        raise ValueError(
#            f"Input shape of edge length is not correct. It should have dimension of 6. Input dimension is {edge_len.shape[1]}"
#        )
#
#    l0 = edge_len[:, [1, 2, 3]]
#    l1 = edge_len[:, [0, 2, 4]]
#    l2 = edge_len[:, [0, 1, 5]]
#    l3 = edge_len[:, [3, 4, 5]]
#
#    a0 = 0.5 * doublearea(l0)
#    a1 = 0.5 * doublearea(l1)
#    a2 = 0.5 * doublearea(l2)
#    a3 = 0.5 * doublearea(l3)
#    return torch.stack([a0, a1, a2, a3], dim=1)


def face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Calculate a tet mesh face area

    Args:
        vertices (torch.Tensor): #V x dim tensor of vertex position
        faces (torch.Tensor): #F x 4 tensor of mesh elements (tet)

    Returns:
        torch.Tensor: #T x 4 tensor of the tet's face area corresponding to faces opposite vertices
    """
    return face_areas(edge_lengths(vertices, faces))


def cotmatrix_entries(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute cotangent weights for each edge int the mesh

    Args:
        vertices (torch.Tensor): #V x dim tensor of vertex position
        faces (torch.Tensor): #F x {3|4} tensor of mesh elements (tri or tet)

    Returns:
        torch.Tensor: #F x {3|6} tensor of cotangent values
    """

    simplex_size = faces.shape[1]

    if simplex_size == 3:
        l2 = squared_edge_lengths(vertices, faces)
        dblA = doublearea(vertices, faces)

        C = torch.zeros((faces.shape[0], 3), dtype=vertices.dtype)
        C[:, 0] = (l2[:, 1] + l2[:, 2] - l2[:, 0]) / (4.0 * dblA)
        C[:, 1] = (l2[:, 2] + l2[:, 0] - l2[:, 1]) / (4.0 * dblA)
        C[:, 2] = (l2[:, 0] + l2[:, 1] - l2[:, 2]) / (4.0 * dblA)
    # TODO (ahmed) finish tet mesh
    # elif simplex_size == 4:
    #    l = edge_lengths(vertices, faces)
    #    s = face_areas(l)
    #    theta, cos_theta = dihedral_angles_intrinsic(l, s)
    #    vol = volume(l)
    #    sin_theta = torch.zeros_like(cos_theta)
    #    for i in range(6):
    #        sin_theta[:, i] = vol / (
    #            (2.0 / (3.0 * l[:, i])) * s[:, i // 3] * s[:, (i % 3) + 1]
    #        )
    #    C = (1.0 / 6.0) * l * cos_theta / sin_theta
    else:
        raise ValueError(
            f"Only accept triangles (3) and tet (4) meshes. Input mesh simplex size ={simplex_size}"
        )

    return C


def triangle_cot_laplacian(
    vertices: torch.Tensor, faces: torch.Tensor
) -> scipy.sparse.coo_matrix:
    """Build the cotangent Laplacian matrix for triangle mesh

    Args:
        vertices (torch.Tensor):
        faces (torch.Tensor):
    Taken from https://github.com/libigl/libigl/blob/main/include/igl/cotmatrix.h
    """
    num_vertices = vertices.shape[0]
    simplex_size = faces.shape[1]

    if simplex_size != 3 and simplex_size != 4:
        raise ValueError(
            f"Only accept triangles and tet meshes! Input mesh simplex size ={simplex_size}"
        )

    # List how to index through faces to get the edges
    edges = edge_local_index(faces)

    C = cotmatrix_entries(vertices, faces)

    num_entries = faces.shape[0] * edges.shape[0] * 4
    I = []
    J = []
    Vals = []
    for i in range(faces.shape[0]):
        for e in range(edges.shape[0]):
            source = faces[i, edges[e, 0]].item()
            dest = faces[i, edges[e, 1]].item()

            I.extend([source, dest, source, dest])
            J.extend([dest, source, source, dest])
            Vals.extend(
                [C[i, e].item(), C[i, e].item(), -C[i, e].item(), -C[i, e].item()]
            )

    L = scipy.sparse.coo_matrix((Vals, (I, J)), shape=(num_vertices, num_vertices))
    return L

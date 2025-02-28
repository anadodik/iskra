import torch
from scipy.sparse import coo_matrix

from iskra.geometry.volume import edge_lengths


def grad_edges(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Finite element gradient matrix for edges.

    Given a polyline, computes the finite element gradient matrix assuming piecewise
    linear hat function basis.

    Args:
    vertices (torch.Tensor): (n,d) tensor vertex list of a polyline where
    faces (torch.Tensor): tensor of ints of shape (m, 2) interpreted as edge index list of
    a polyline

    Returns:
    torch.Tensor: (d*m, n) sparse coo tensor of the finite element gradient matrix

    Notes:
    Taken from https://github.com/sgsellan/gpytoolbox/blob/main/src/gpytoolbox/grad.py
    """
    simplex_size = faces.shape[1]

    if simplex_size != 2:
        raise ValueError(
            f"grad_edges only accepts edges (2) Input's simplex size ={simplex_size}."
        )
    # FD with varying edge lengths
    edge_len = edge_lengths(
        torch.stack((vertices[faces[:, 1], :], vertices[faces[:, 0], :]), dim=1)
    )

    i = torch.arange(faces.shape[0], dtype=torch.long)
    i = torch.cat((i, i))
    j = torch.cat((faces[:, 0], faces[:, 1]))
    vals = torch.ones(faces.shape[0], dtype=vertices.dtype) / edge_len
    vals = torch.cat((-vals, vals))
    g = coo_matrix(
        (vals.numpy(), (i.numpy(), j.numpy())),
        shape=(faces.shape[0], vertices.shape[0]),
    )
    return g


def grad_triangles(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Finite element gradient matrix for triangles.

    Given a triangle mesh, computes the finite element gradient matrix assuming piecewise
    linear hat function basis.

    Args:
    vertices (torch.Tensor):(n,d) tensor vertex list a triangle mesh
    faces (torch.Tensor): tensor of ints of shape (m, 3) interpreted as face index
    list of a triangle mesh

    Returns:
    torch.Tensor: (d*m, n) sparse coo tensor of the finite element gradient matrix

    Notes:
    Taken from https://github.com/sgsellan/gpytoolbox/blob/main/src/gpytoolbox/grad.py
    """
    dim = vertices.shape[1]
    simplex_size = faces.shape[1]

    if simplex_size != 3:
        raise ValueError(
            f"grad_triangles only accepts triangles (3). "
            f"Input's simplex size ={simplex_size}."
        )

    # If the input is 2D, add a zero dimension to make it 3D
    if dim == 2:
        vertices = torch.cat(
            (
                vertices,
                torch.zeros(
                    (vertices.shape[0], 1),
                    dtype=vertices.dtype,
                    device=vertices.device,
                ),
            ),
            dim=1,
        )
    # Gradient of scalar function defined on piecewise linear elements (mesh) is
    # constant on each triangle i, j,k:
    # Renaming indices of triangle vertices
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    # Fx3 matrices of triangles edge vectors, named after opposite vertices
    v21 = vertices[i2] - vertices[i1]
    v02 = vertices[i0] - vertices[i2]
    v10 = vertices[i1] - vertices[i0]
    # area of parallelogram is twice area of triangle
    n = torch.cross(v21, v02, dim=1)
    # Twice the area of each triangle (L2 norm of normal vectors)
    dbl_area = torch.norm(n, dim=1)
    # unit normal vector
    u = n / (dbl_area[:, None] + 1e-14)
    # perpendicular edge vectors
    cross_u_v10 = torch.cross(u, v10, dim=1)
    cross_u_v02 = torch.cross(u, v02, dim=1)
    eperp10 = cross_u_v10 * (
        torch.norm(v10, dim=1)[:, None]
        / (dbl_area[:, None] * torch.norm(cross_u_v10, dim=1)[:, None])
    )
    eperp02 = cross_u_v02 * (
        torch.norm(v02, dim=1)[:, None]
        / (dbl_area[:, None] * torch.norm(cross_u_v02, dim=1)[:, None])
    )
    # triangle indices for the sparse matrix construction
    f_ind = torch.arange(faces.shape[0], dtype=torch.long, device=vertices.device)
    i = torch.cat(
        (
            f_ind,
            f_ind,
            f_ind,
            f_ind,
            faces.shape[0] + f_ind,
            faces.shape[0] + f_ind,
            faces.shape[0] + f_ind,
            faces.shape[0] + f_ind,
            2 * faces.shape[0] + f_ind,
            2 * faces.shape[0] + f_ind,
            2 * faces.shape[0] + f_ind,
            2 * faces.shape[0] + f_ind,
        )
    )
    j = torch.cat((faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 0]))
    j = torch.cat((j, j, j))
    vals = torch.cat(
        (
            eperp02[:, 0],
            -eperp02[:, 0],
            eperp10[:, 0],
            -eperp10[:, 0],
            eperp02[:, 1],
            -eperp02[:, 1],
            eperp10[:, 1],
            -eperp10[:, 1],
            eperp02[:, 2],
            -eperp02[:, 2],
            eperp10[:, 2],
            -eperp10[:, 2],
        )
    )
    g = coo_matrix(
        (
            vals.cpu().detach().numpy(),
            (i.cpu().detach().numpy(), j.cpu().detach().numpy()),
        ),
        shape=(3 * faces.shape[0], vertices.shape[0]),
    )
    # if the original input was 2D, keep only the first two rows
    if dim == 2:
        # TODO this is unnecessarily ugly
        g_csr = g.tocsr()[0 : (2 * faces.shape[0]), :]
        g = g_csr.tocoo()
    return g


def grad(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Finite element gradient matrix.

    Given a triangle mesh or a polyline, computes the finite element gradient matrix
    assuming piecewise linear hat function basis.

    Args:
    vertices (torch.Tensor):(n,d) tensor vertex list of a polyline or triangle mesh
    faces (torch.Tensor): tensor of ints
        if (m, 2),  interpret as edge index list of a polyline
        if (m, 3),  interpret as face index list of a triangle mesh

    Returns:
    torch.Tensor: (d*m, n) sparse coo tensor of the finite element gradient matrix

    Notes:
    Taken from https://github.com/sgsellan/gpytoolbox/blob/main/src/gpytoolbox/grad.py
    """
    simplex_size = faces.shape[1]

    # polyline
    if simplex_size == 2:
        return grad_edges(vertices, faces)

    # triangles
    if simplex_size == 3:
        return grad_triangles(vertices, faces)

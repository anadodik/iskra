import torch
from scipy.sparse import coo_matrix


def grad(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Finite element gradient matrix
    Given a triangle mesh or a polyline, computes the finite element gradient matrix assuming piecewise linear hat function basis.

    Parameters:
    V:(n,d) tensor
        vertex list of a polyline or triangle mesh
    F: tensor of ints
        if (m, 2),  interpret as edge index list of a polyline
        if (m, 3),  interpret as face index list of a triangle mesh

    Returns:
    G: (d*m, n) sparse coo tensor of the finite element gradient matrix

    Notes:
    Taken from https://github.com/sgsellan/gpytoolbox/blob/main/src/gpytoolbox/grad.py
    """

    dim = V.shape[1]
    simplex_size = F.shape[1]

    # polyline
    if simplex_size == 2:
        # FD with varying edge lengths
        edge_len = torch.norm(V[F[:, 1], :] - V[F[:, 0], :], dim=1)

        I = torch.arange(F.shape[0], dtype=torch.long)
        I = torch.cat((I, I))
        J = torch.cat((F[:, 0], F[:, 1]))
        vals = torch.ones(F.shape[0], dtype=V.dtype) / edge_len
        vals = torch.cat((-vals, vals))
        G = coo_matrix(
            (vals.numpy(), (I.numpy(), J.numpy())), shape=(F.shape[0], V.shape[0])
        )
        return G

    # triangles
    if simplex_size == 3:
        # If the input is 2D, add a zero dimension to make it 3D

        if dim == 2:
            V = torch.cat(
                (V, torch.zeros((V.shape[0], 1), dtype=V.dtype, device=V.device)), dim=1
            )

        # Gradient of scalar function defined on piecewise linear elements (mesh) is constant on each triangle i, j,k:
        # Renaming indices of triangle vertices
        i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

        # Fx3 matrices of triangles edge vectors, named after opposite vertices
        v21 = V[i2] - V[i1]
        v02 = V[i0] - V[i2]
        v10 = V[i1] - V[i0]

        # area of parallelogram is twice area of triangle
        n = torch.cross(v21, v02, dim=1)

        # Twice the area of each triangle (L2 norm of normal vectors)
        dblA = torch.norm(n, dim=1)

        # unit normal vector
        u = n / (dblA[:, None] + 1e-14)

        # perpendicular edge vectors
        cross_u_v10 = torch.cross(u, v10, dim=1)
        cross_u_v02 = torch.cross(u, v02, dim=1)

        eperp10 = cross_u_v10 * (
            torch.norm(v10, dim=1)[:, None]
            / (dblA[:, None] * torch.norm(cross_u_v10, dim=1)[:, None])
        )

        eperp02 = cross_u_v02 * (
            torch.norm(v02, dim=1)[:, None]
            / (dblA[:, None] * torch.norm(cross_u_v02, dim=1)[:, None])
        )

        # triangle indices for the sparse matrix construction
        Find = torch.arange(F.shape[0], dtype=torch.long, device=V.device)

        I = torch.cat(
            (
                Find,
                Find,
                Find,
                Find,
                F.shape[0] + Find,
                F.shape[0] + Find,
                F.shape[0] + Find,
                F.shape[0] + Find,
                2 * F.shape[0] + Find,
                2 * F.shape[0] + Find,
                2 * F.shape[0] + Find,
                2 * F.shape[0] + Find,
            )
        )

        J = torch.cat((F[:, 1], F[:, 0], F[:, 2], F[:, 0]))
        J = torch.cat((J, J, J))

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

        G = coo_matrix(
            (
                vals.cpu().detach().numpy(),
                (I.cpu().detach().numpy(), J.cpu().detach().numpy()),
            ),
            shape=(3 * F.shape[0], V.shape[0]),
        )

        # if the original input was 2D, keep only the first two rows
        if dim == 2:
            # TODO this is unnecessarily ugly
            G_csr = G.tocsr()[0 : (2 * F.shape[0]), :]
            G = G_csr.tocoo()

        return G

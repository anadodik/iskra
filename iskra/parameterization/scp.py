# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from typing import Literal

import torch

import iskra.sparse as sp
from iskra import dec
from iskra.sparse_linalg import eigsh
from iskra.topology import boundary


def vertex_area_matrix(
    n_vertices: int, faces: torch.Tensor, dtype: torch.dtype = torch.float32
) -> sp.SparseTensor:
    bdr_edges = boundary(faces)
    bdr_edges_bwd = bdr_edges[:, (1, 0)]
    n_bdr_edges = bdr_edges.shape[0]
    idcs_i = torch.cat([bdr_edges, bdr_edges_bwd + n_vertices], -2).flatten(-2, -1)
    idcs_j = torch.cat([bdr_edges_bwd + n_vertices, bdr_edges], -2).flatten(-2, -1)
    values = torch.tensor([0.25, -0.25], device=faces.device, dtype=dtype)
    values = values[None, :].expand(2 * n_bdr_edges, -1).flatten(-2, -1)
    return sp.coo_tensor(
        torch.stack([idcs_i, idcs_j], -2), values, size=[2 * n_vertices, 2 * n_vertices]
    )


def boundary_matrix(
    n_verts: int, faces: torch.Tensor, dtype: torch.dtype = torch.float32
) -> sp.SparseTensor:
    bdr_idx = boundary(faces).flatten().unique()
    bdr_ii = torch.stack([bdr_idx, bdr_idx], 0)
    bdr_val = torch.ones(bdr_ii.shape[1], dtype=dtype, device=faces.device)
    boundary_mat_block = sp.coo_tensor(bdr_ii, bdr_val, size=[n_verts, n_verts])
    boundary_mat = sp.repdiag(boundary_mat_block, 2)
    return boundary_mat


def assemble_conformal_laplacian(
    lap: sp.SparseTensor, vertex_area_mat: sp.SparseTensor
) -> sp.SparseTensor:
    r"""Assembles a conformal laplacian from a cotan laplacian and a vertex area matrix.

    The conformal laplacian is equal to:

    $$
    L_c = 
    \begin{bmatrix}
    L & 0 \\
    0 & L \\
    \end{bmatrix}
    -
    2 A,
    $$
    
    where $L$ represents the cotan laplacian, and $A$ is the vertex area matrix.
    See Spectral Conformal Parameterization by Mullen et al. 2008 for more details.

    Args:
        lap (sp.SparseTensor): Cotan laplacian, e.g., from `iskra.dec.laplacian`.
        vertex_area_mat (sp.SparseTensor): Vertex area matrix, see `vertex_area_matrix`.

    Returns:
        sp.SparseTensor:
    """
    return (sp.repdiag(lap, 2) - 2.0 * vertex_area_mat).coalesce()


def conformal_laplacian(
    verts: torch.Tensor, faces: torch.Tensor, clamp_min: float | None = 0.0
) -> sp.SparseTensor:
    lap, _ = dec.laplacian(verts, faces, clamp_min=clamp_min)
    vertex_area_mat = vertex_area_matrix(verts.shape[0], faces, dtype=verts.dtype)
    return assemble_conformal_laplacian(lap, vertex_area_mat)


def conformal_laplacian_from_weights(
    d_01: torch.Tensor, weights: torch.Tensor, vertex_area_mat: sp.SparseTensor
) -> sp.SparseTensor:
    lap = sp.matmul(d_01.mT, sp.matmul(sp.diag(weights), d_01)).coalesce()
    return assemble_conformal_laplacian(lap, vertex_area_mat)


def scp_solve(
    conformal_lap: sp.SparseTensor,
    boundary_mat: sp.SparseTensor,
    *,
    eps: float = 0.0,
    sigma: float = -1e-12,
    bwd_method: Literal[
        "unroll", "individual", "truncate", "dodik-fixedpoint", "dodik-invert"
    ] = "individual",
    bwd_max_iter: int = 25,
) -> torch.Tensor:
    if boundary_mat.nelement() == 0:
        raise ValueError(
            "Boundary matrix empty. SCP doesn't make sense on meshs without boundaries."
        )
    if eps != 0.0:
        conformal_lap = conformal_lap + eps * sp.eye(
            conformal_lap.shape[0],
            dtype=conformal_lap.dtype,
            device=conformal_lap.device,
        )
    _, evecs = eigsh(
        conformal_lap,
        M=boundary_mat,
        k=3,
        sigma=sigma,
        bwd_method=bwd_method,
        bwd_max_iter=bwd_max_iter,
    )
    # eigsh sorts descending in shift-inverse mode:
    uv_opt = evecs[:, 0:1].reshape(2, -1).mT
    return uv_opt


def scp_solve_from_weights(
    d_01: torch.Tensor,
    weights: torch.Tensor,
    vertex_area_mat: sp.SparseTensor,
    boundary_mat: sp.SparseTensor,
    *,
    eps: float = 0.0,
    sigma: float = -1e-12,
    bwd_method: Literal[
        "unroll", "individual", "truncate", "dodik-fixedpoint", "dodik-invert"
    ] = "individual",
    bwd_max_iter: int = 25,
) -> torch.Tensor:
    conformal_lap = conformal_laplacian_from_weights(d_01, weights, vertex_area_mat)
    return scp_solve(
        conformal_lap,
        boundary_mat,
        eps=eps,
        sigma=sigma,
        bwd_method=bwd_method,
        bwd_max_iter=bwd_max_iter,
    )

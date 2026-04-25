# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import json
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, cast

import igl
import numpy as np
import torch

import iskra.sparse as sp
import iskra.sparse_linalg as spla
from iskra.adjoint import make_solver_layer
from iskra.dec import d_01, d_10, laplacian, laplacian_from_weights
from iskra.geometry import cotan_weights
from iskra.logging.logging import getLogger
from iskra.mesh import Mesh
from iskra.profiling import global_profiler, profile_block, profile_fn
from iskra.signed_svd import closest_rot_3x3, polar_3x3, signed_svd
from iskra.topology import (
    edge_to_vertex_adjacency,
    face_index,
    reduce_on_subface,
    vertex_adjacency,
)

LOGGER = getLogger(__name__)


@profile_fn
def arap_step(
    verts_deformed,
    verts_rest,
    cots,
    vert_vert,
    lap,
    bc_idx,
    bc_vals,
    solver: spla.SolverT | None = None,
):
    n_vertices = verts_rest.shape[0]

    with profile_block("covs"):
        lines = face_index(verts_rest, vert_vert)
        vecs = lines[..., 1, :] - lines[..., 0, :]
        lines_deformed = face_index(verts_deformed, vert_vert)
        vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]
        covs = cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]
        vert_covs = reduce_on_subface(covs, vert_vert[:, 0:1], n_vertices, "sum")

    with profile_block("svd"):
        vert_rot = closest_rot_3x3(vert_covs)

    with profile_block("rotation"):
        halfedge_rot = face_index(vert_rot.mT, vert_vert).mean(1)
        rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]
        rhs = reduce_on_subface(
            rotated_halfedge_vecs, vert_vert[:, 0:1], n_vertices, "sum"
        )

    with profile_block("solve"):
        verts_deformed = spla.min_quadratic_energy(
            lap, -rhs, bc_idx, bc_vals, solver=solver
        )[1]
    return verts_deformed


def _arap_step_paper(deformed, verts, cots, vert_vert, lap, handle_idx, handles):
    nv = verts.shape[0]

    # Find covariances
    lines = face_index(verts, vert_vert)
    vecs = lines[..., 1, :] - lines[..., 0, :]
    lines_def = face_index(deformed, vert_vert)
    vecs_def = lines_def[..., 1, :] - lines_def[..., 0, :]
    covs = cots[..., None, None] * vecs_def[..., None, :] * vecs[..., :, None]
    vert_covs = reduce_on_subface(covs, vert_vert[..., 0:1], nv, "sum")

    # Find closest rotation
    vert_rot = closest_rot_3x3(vert_covs)

    # Rotate using closest rotation
    vert_vert_rot = face_index(vert_rot.mT, vert_vert).mean(1)
    rotated = cots[:, None] * (vert_vert_rot @ vecs[..., None])[..., 0]
    rhs = reduce_on_subface(rotated, vert_vert[..., 0:1], nv, "sum")

    # Solve for deformation
    _, deformed = spla.min_quadratic_energy(lap, -rhs, handle_idx, handles)
    return deformed


def arap_step_with_energy(
    verts_deformed: torch.Tensor,
    verts: torch.Tensor,
    cots: torch.Tensor,
    vert_vert: torch.Tensor,
    lap: torch.Tensor,
    bc_idx: torch.Tensor,
    bc_vals: torch.Tensor,
    solver: spla.SolverT | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_vertices = verts.shape[0]

    lines = face_index(verts, vert_vert)
    vecs = lines[..., 1, :] - lines[..., 0, :]

    lines_deformed = face_index(verts_deformed, vert_vert)
    vecs_deformed = lines_deformed[..., 1, :] - lines_deformed[..., 0, :]

    halfedge_covs = (
        cots[..., None, None] * vecs_deformed[..., None, :] * vecs[..., :, None]
    )

    vert_covs = reduce_on_subface(halfedge_covs, vert_vert[:, 0:1], n_vertices, "sum")
    # Uncomment to debug SVD:
    # vert_u, _, vert_vt = signed_svd(vert_covs)
    # vert_rot = vert_vt.mT @ vert_u.mT
    # vert_rot = vert_covs * 0.0 + torch.eye(3, dtype=vert_u.dtype)[None, :, :].expand(
    #     n_vertices, -1, -1
    # )
    vert_rot = closest_rot_3x3(vert_covs)
    assert (torch.linalg.det(vert_rot) > 0).all()

    # Following lines are energy only:
    halfedge_vert_rot = face_index(vert_rot.mT, vert_vert).mean(1)
    diff = vecs_deformed - (halfedge_vert_rot @ vecs[..., None])[..., 0]
    weighted_dist = cots * torch.linalg.vector_norm(diff, dim=-1, ord=2) ** 2

    vert_energy = reduce_on_subface(weighted_dist, vert_vert[:, 0:1], n_vertices, "sum")

    # THIS IS INTERPOLATING ROTATIONS WEIRDLY??? SHRINKWRAP ARTIFACTS?
    halfedge_rot = face_index(vert_rot.mT, vert_vert).mean(1)
    rotated_halfedge_vecs = cots[:, None] * (halfedge_rot @ vecs[..., None])[..., 0]

    rhs = reduce_on_subface(rotated_halfedge_vecs, vert_vert[:, 0:1], n_vertices, "sum")
    _, verts_deformed = spla.min_quadratic_energy(
        lap, -rhs, bc_idx, bc_vals, solver=solver
    )
    return verts_deformed, vert_energy


@profile_fn
def arap_solve(
    verts: torch.Tensor,
    handle_idx: torch.Tensor,
    handles: torch.Tensor,
    vert_vert: torch.Tensor,
    vert_vert_weights: torch.Tensor,
    lap: torch.Tensor,
    lap_factors: spla.SolverT | None = None,
    fwd_max_iter: int = 1000,
    compute_fwd_energy: bool = False,
    fwd_error_ord: int | float | Literal["fro", "nuc"] = 2,
    fwd_abs_tol: float = 1e-7,
    fwd_rel_tol: float = 1e-4,
    bwd_max_iter: int = 200,
    bwd_abs_tol: float = 1e-7,
    bwd_rel_tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: fix `profile_fn` not type-checking correctly
    if handle_idx.numel() < 4:
        LOGGER.warning(
            "ARAP gradients may be incorrect with fewer than 4 boundary conditions."
        )
    arap_fn = arap_step
    fwd_error_metric = "delta"
    if compute_fwd_energy:
        arap_fn = arap_step_with_energy
        fwd_error_metric = 1
    solver = make_solver_layer(
        partial(arap_fn, solver=lap_factors),
        [(0, 0)],
        (1, 2, 4, 6),
        fwd_method="fixed-point",
        fwd_max_iter=fwd_max_iter,
        fwd_error_metric=fwd_error_metric,
        fwd_error_ord=fwd_error_ord,
        fwd_abs_tol=fwd_abs_tol,
        fwd_rel_tol=fwd_rel_tol,
        bwd_method="gmres",
        bwd_max_iter=bwd_max_iter,
        bwd_abs_tol=bwd_abs_tol,
        bwd_rel_tol=bwd_rel_tol,
        verbose=verbose,
    )

    init = verts.clone()
    # TODO: Next line only necessary because of bad gradients with identity matrix?
    init[handle_idx] = handles
    result = solver(init, verts, vert_vert_weights, vert_vert, lap, handle_idx, handles)
    if compute_fwd_energy:
        return result
    else:
        # TODO: fix returns
        return result, torch.full_like(result[:, 0], float("inf"))


def arap_precompute(
    verts: torch.Tensor,
    faces: torch.Tensor,
    handle_idx: torch.Tensor,
    lap_clamp: float | None = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, spla.SolverT]:
    """Precomputes data necessary to run ARAP deformations.

    !!! note
        It is necessary to clamp the ARAP weights to be positive.
        Otherwise, the SVD procedure is not guaranteed to lead to the closest rotation.
        We allow setting it to None to match the libigl implementation,
        but this is likely to break your gradients in the backward pass.


    Args:
        verts (Tensor[V, 3]): Mesh vertex positions.
        faces (Tensor[F, 3]): Mesh face indices.
        handle_idx (Tensor[H]): Indices signifying which vertices are control handles.
        lap_clamp (float | None): Clamp minimum for cotan weights. Defaults to 1e-5.

    Returns:
        Tensor[V, 2]: Vertex-to-vertex adjacency from `vertex_adjacency`.
        Tensor[2V]: Cotan weights per vertex-vertex relationship.
        SparseTensor[V, V]: Cotan (integrated) Laplacian matrix.
        spla.SolverT: Factorized Laplacian solver.
    """
    n_verts = verts.shape[0]
    vert_vert = vertex_adjacency(faces)
    weights = cotan_weights(verts, faces, clamp_min=lap_clamp)
    vert_vert_weights = edge_to_vertex_adjacency(weights)
    lap = laplacian_from_weights(weights, faces)
    unknown_idx = sp.index_complement(n_verts, handle_idx)
    lap_uk = spla.quad_energy_mat(lap, unknown_idx)
    lap_factors = spla.default_solver(lap_uk)
    return vert_vert, vert_vert_weights, lap, lap_factors

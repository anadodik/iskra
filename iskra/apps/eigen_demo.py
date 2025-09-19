# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
import torch

import iskra.sparse as sp
from iskra.dec import laplacian
from iskra.geometry import normal_coordinate_system, triangle_areas, triangle_normals
from iskra.mesh import Mesh
from iskra.sparse_linalg import (
    eigsh,
)
from iskra.topology import boundary, face_index, get_subfaces, ordered_boundary_edges


def kron(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]
    )


def solve_triu_transposed(a, b):
    return torch.linalg.solve_triangular(a.mT, b.mT, upper=False).mT


def qr_jacobian_system(Q, R, L):
    n, k = Q.shape
    dtype = Q.dtype
    # Kronecker helpers
    # kron = torch.kron
    I_n, I_k = torch.eye(n, dtype=dtype), torch.eye(k, dtype=dtype)

    # Commutation matrix K (k^2 x k^2)
    idx = torch.arange(k * k).reshape(k, k)
    K = torch.zeros((k * k, k * k), dtype=dtype)
    for i in range(k):
        for j in range(k):
            K[idx[i, j], idx[j, i]] = 1.0

    # Mask diag for L
    M_L = torch.diag(L.reshape(-1).to(dtype=dtype))

    A11 = kron(R.T, I_n)
    A12 = kron(I_k, Q)
    A21 = (torch.eye(k * k, dtype=dtype) + K) @ kron(I_k, Q.T)
    A22 = torch.zeros((k * k, k * k), dtype=dtype)
    A31 = torch.zeros((k * k, n * k), dtype=dtype)
    A32 = M_L

    # Assemble block system matrix
    top = torch.cat([A11, A12], dim=1)
    mid = torch.cat([A21, A22], dim=1)
    bot = torch.cat([A31, A32], dim=1)
    A = torch.cat([top, mid, bot], dim=0)

    return A  # multiplies [vec(dQ); vec(dR)] = [vec(dB); 0; 0]


def qr_jacobian_Q(Q, R, L):
    n, k = Q.shape
    dtype = Q.dtype
    I_n, I_k = torch.eye(n, dtype=dtype), torch.eye(k, dtype=dtype)
    # kron = torch.kron

    # Mask for strict-lower entries
    M_L = torch.diag(L.reshape(-1).to(dtype))
    print(M_L.shape)

    # Elimination matrix E for skew-symmetric k×k
    m = k * (k - 1) // 2
    E = torch.zeros((k * k, m), dtype=dtype)
    c = 0
    for i in range(k):
        for j in range(i + 1, k):
            e = torch.zeros((k, k), dtype=dtype)
            e[i, j] = 1.0
            e[j, i] = -1.0
            E[:, c] = e.reshape(-1)
            c += 1

    term1 = kron(torch.linalg.inv(R).T, I_n - Q @ Q.T)
    term2 = (
        kron(I_k, Q)
        @ E
        @ torch.linalg.inv(E.T @ M_L @ kron(R.T, I_k) @ E)
        @ E.T
        @ M_L
        @ kron(I_k, Q.T)
    )

    return term1 + term2  # shape (n*k, n*k)


def complicated_loss(evals, evecs, k):
    w_evals = torch.linspace(1.0, 2.0, steps=k, dtype=evals.dtype, device=evals.device)
    loss_evals = (
        w_evals * evals + 0.1 * (w_evals**2) * (evals**2)
    ).sum()  # linear + quadratic, per-index wehts

    # Distinct wehts for each entry of evecs (n x k)
    i = torch.arange(evecs.size(0), dtype=evecs.dtype, device=evecs.device).unsqueeze(
        1
    )  # row indices
    j = torch.arange(evecs.size(1), dtype=evecs.dtype, device=evecs.device).unsqueeze(
        0
    )  # col indices
    W = (i + 1) + 0.37 * (j + 1) + 0.01 * (i + 1) * (j + 1)  # unique weht per (i,j)
    loss_evecs = (
        (W * torch.abs(evecs)) ** 2
    ).sum()  # quadratic in evecs; sign-invariant

    return loss_evecs + loss_evals


def test_adjoint(
    lap: torch.Tensor, mass: torch.Tensor, k: int, sigma: float | None, adjoint: str
):
    print(f"\n\n-----------{adjoint}-----------")
    if adjoint == "dense":
        lap = lap.to_dense()
        mass = mass.to_dense()

    lap = lap.requires_grad_(True)
    lap.grad = torch.zeros_like(lap)

    if adjoint == "dense":
        if sigma is not None:
            raise ValueError("Dense PyTorch does not work with manual sigma shift.")
        evals, evecs = torch.linalg.eig(torch.linalg.solve(mass, lap))
        evals, evecs = evals.real, evecs.real
        sort_idx = torch.argsort(-torch.abs(evals))
        evals = evals[sort_idx][:k]
        evecs = evecs[:, sort_idx][:, :k]
    else:
        evals, evecs = eigsh(lap, M=mass, k=k, sigma=sigma, adjoint=adjoint)
    print(f"eigenvalues: {evals}")
    print(f"eigenvectors: {evecs}")

    loss = ((evecs.abs() - 0.5) ** 2).sum()
    # loss = complicated_loss(evals, evecs, k)
    loss.backward()
    grad_a = lap.grad
    if adjoint == "dense":
        grad_a = 0.5 * (grad_a + grad_a.mT)
    print(f"grad A: {grad_a.to_dense()}")
    return evals, evecs, grad_a


if __name__ == "__main__":
    dtype = torch.double
    device = "cpu"
    torch.set_printoptions(sci_mode=False, linewidth=80 * 2)

    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 3, weight=2)
    G.add_edge(0, 2, weight=3.0)
    G.add_edge(2, 3, weight=4)
    G.add_edge(2, 4, weight=4)
    G.add_nodes_from(range(5))

    n = G.number_of_nodes()
    k = 5
    sigma = None  # -1e-12

    lap = nx.laplacian_matrix(G)
    lap = sp.scipy_to_torch(lap).to_sparse_coo().to(dtype=dtype)  # + 1e-2 * sp.eye(n)
    mass = sp.eye(n, dtype=dtype)

    # lap = torch.sparse_coo_tensor(
    #     torch.stack(2 * [torch.arange(5)]),
    #     torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
    #     device=device,
    #     dtype=dtype,
    # )

    # test_adjoint(lap, mass, k, sigma, "unroll")
    test_adjoint(lap, mass, k, sigma, "dense")
    # test_adjoint(lap, mass, k, sigma, "dodik-fixedpoint")
    # test_adjoint(lap, mass, k, sigma, "dodik-invert")
    test_adjoint(lap, mass, k, sigma, "truncate")
    # test_adjoint(lap, mass, k, sigma, "individual")

    # print("\n\n-----------DENSE GROUND TRUTH-----------")
    # lap.grad.zero_()
    # evals, evecs = torch.linalg.eigh(lap.to_dense())
    # evals, evecs = evals.real, evecs.real
    # sort_idx = torch.argsort(-torch.abs(evals))
    # evals = evals[sort_idx][:k]
    # evecs = evecs[:, sort_idx][:, :k]
    # loss = ((evecs.abs() - 0.5) ** 2).sum()
    # print(evals)
    # print(evecs)
    # loss.backward()
    # print(lap.grad.to_dense())

    time.sleep(0.5)
    quit(0)

    # unit_vectors = torch.eye(n * k, dtype=dtype)
    # lap_p = lap.requires_grad_()
    # jacobian_rows = [
    #     torch.autograd.grad(
    #         power_iterations(lap_p, M=mass, k=k, sigma=-1e-12)[1].flatten(),
    #         lap_p,
    #         vec,
    #     )[0]
    #     for vec in unit_vectors
    # ]
    # jacobian = torch.stack(jacobian_rows).to_dense().reshape(n, k, n, n)
    # print(jacobian)

    def b(a, x):
        return a @ x

    def qr_q(b):
        return torch.linalg.qr(b)[0]

    def implicit_func(a, x):
        return torch.linalg.qr(b(a, x))[0] - x

    # Test full formula:
    df_da = torch.func.jacrev(implicit_func, 0)
    df_dx = torch.func.jacrev(implicit_func, 1)
    dq_db = torch.func.jacrev(qr_q, 0)
    db_da = torch.func.jacrev(b, 0)
    db_dx = torch.func.jacrev(b, 1)

    df_da_mat = df_da(lap, evecs).reshape(n * k, n * n)
    df_dx_mat = df_dx(lap, evecs).reshape(n * k, n * k)
    dx_da_mat = -torch.linalg.solve(df_dx_mat, df_da_mat)
    dq_db_mat = dq_db(lap @ evecs).reshape(n * k, n * k)
    db_da_mat = db_da(lap, evecs).reshape(n * k, n * n)
    db_dx_mat = db_dx(lap, evecs).reshape(n * k, n * k)

    q, r = torch.linalg.qr(lap @ evecs)
    dl_dx = torch.linspace(0, 1, n * k, device=device, dtype=dtype).reshape([1, n * k])

    print(dl_dx @ dx_da_mat)
    print(-torch.linalg.solve(df_dx_mat.mT, dl_dx.mT).mT @ df_da_mat)

    # print(
    #     (
    #         dl_dx
    #         + torch.linalg.solve(
    #             (torch.linalg.inv(dq_db_mat) - db_dx_mat).mT, dl_dx.mT
    #         ).mT
    #         @ db_dx_mat
    #     )
    #     @ dq_db_mat
    #     @ db_da_mat
    # )

    # J_Q = qr_jacobian_Q(q, r, torch.tril(torch.ones_like(r), -1))
    # print(J_Q)
    # print(dq_db_mat)

    # print(torch.func.jvp(qr_q, (lap @ evecs,), (v,))[1])

    # dg_dq = torch.func.vjp(qr_q, evecs)[1](v)[0]
    # print(dg_dq)
    # print(dq_db)

    # s = q @ v.mT
    # m = torch.triu(s) + torch.triu(s, 1).mT
    # b = solve_triu_transposed(r, (v.mT + q.mT @ m).mT)
    # print(b)
    v = dl_dx
    J = qr_jacobian_system(q, r, torch.tril(torch.ones_like(r), -1))
    Jv = torch.zeros([J.shape[1], 1], device=device, dtype=dtype)
    Jv[: evecs.nelement(), 0] = v.flatten()
    # print(J @ Jv.flatten())
    # print(torch.linalg.solve(J, Jv)[: v.nelement(), :].reshape(*v.shape))

    vJ = torch.zeros([1, J.shape[0]], device=device, dtype=dtype)
    vJ[:, : v.nelement()] = v.flatten()
    # print(v.nelement())
    # print((vJ @ J)[:, : v.nelement()].reshape(*evecs.shape))
    # print(q.mT @ dq)
    # print(dq_1.mT @ q)
    # # print(dq_db_mat.reshape(n * k, n * k) @ v.flatten())
    # tril = torch.tril(
    #     torch.linalg.solve_triangular(r.mT, (q.mT @ v).mT, upper=False).mT, -1
    # )
    # qt_dq = -(tril + tril.mT)
    # print(torch.solve(q.mT, qt_dq))

    # tril_eye = torch.tril(
    #     torch.linalg.solve_triangular(r.mT, q.mT @ q, upper=False).mT, -1
    # )
    # # tril = torch.tril(q.mT @ v, -1)
    # print(dqt_q, r.shape, v.shape, q.shape, dq_1.shape)
    # print(tril_eye)
    # # print(q.shape, r.shape)
    # # krn = kron(eye_n, q.mT) + kron(q.mT, eye_n)
    # # print(krn.shape)

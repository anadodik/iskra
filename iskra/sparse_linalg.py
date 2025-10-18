# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from typing import Callable, Literal

import numpy as np
import scipy.sparse
import torch
from cholespy import CholeskySolverD, CholeskySolverF, MatrixType
from numpy import indices

import iskra.sparse as sp


def make_solver(mat: torch.Tensor) -> CholeskySolverF | CholeskySolverD:
    mat = mat.coalesce()
    if mat.dtype == torch.float32:
        solver = CholeskySolverF(
            mat.shape[0],
            mat.indices()[0],
            mat.indices()[1],
            mat.values(),
            MatrixType.COO,
        )
    elif mat.dtype == torch.float64:
        solver = CholeskySolverD(
            mat.shape[0],
            mat.indices()[0],
            mat.indices()[1],
            mat.values(),
            MatrixType.COO,
        )
    else:
        raise TypeError(
            f"CholeskySolver only supports f32 and f64 matrices, found {mat.dtype}."
        )
    return solver


class CholespySolve(torch.autograd.Function):
    """Differentiable linear system."""

    @staticmethod
    def setup_context(ctx, inputs, output):
        (solver, b, _) = inputs
        x = output
        ctx.save_for_backward(b, x)
        ctx.solver = solver

    @staticmethod
    def forward(
        solver: CholeskySolverF | CholeskySolverD,
        b: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = b.contiguous()
        if x is None:
            x = torch.zeros_like(b)
        solver.solve(b, x)
        return x

    @staticmethod
    def backward(
        ctx, forward_grad: torch.Tensor
    ) -> tuple[None, torch.Tensor | None, None]:
        forward_grad = forward_grad.contiguous()
        b_grad: torch.Tensor | None = None

        if ctx.needs_input_grad[1]:
            b_grad = torch.zeros_like(forward_grad)
            ctx.solver.solve(forward_grad, b_grad)
        return None, b_grad, None


def cholespy_solve(
    solver: CholeskySolverF | CholeskySolverD,
    b: torch.Tensor,
    x: torch.Tensor | None = None,
) -> torch.Tensor:
    return CholespySolve.apply(solver, b, x)  # pyright: ignore[reportReturnType]


class CholespyFactorAndSolve(torch.autograd.Function):
    """Differentiable linear system."""

    @staticmethod
    def setup_context(ctx, inputs, output):
        (mat, b, _) = inputs
        (solver, x) = output
        ctx.save_for_backward(mat, b, x)
        ctx.solver = solver

    @staticmethod
    def forward(
        mat: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> tuple[CholeskySolverF | CholeskySolverD, torch.Tensor]:
        if not mat.is_coalesced():
            raise ValueError("Matrix not coalesced, please call .coalesce() first.")
        solver: CholeskySolverF | CholeskySolverD = make_solver(mat)
        b = b.contiguous()
        if x is None:
            x = torch.zeros_like(b)
        solver.solve(b, x)
        return solver, x

    @staticmethod
    def backward(
        ctx, solver_grad: None, forward_grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        forward_grad = forward_grad.contiguous()
        n = forward_grad.shape[0]
        mat, _, x = ctx.saved_tensors
        b_grad: torch.Tensor | None = None
        mat_grad: torch.Tensor | None = None

        dg_dx = torch.zeros_like(forward_grad)
        ctx.solver.solve(forward_grad, dg_dx)
        # print("forward_grad", forward_grad)
        # print("dg_dx", dg_dx)
        if ctx.needs_input_grad[0]:
            # Compute masked outer product:
            # (dg/dx) @ x^T
            rows, cols = mat.indices()
            grad_vals = -dg_dx[rows, ...] * x[cols, ...]
            if grad_vals.ndim == 2:
                grad_vals = grad_vals.sum(-1)
            mat_grad = torch.sparse_coo_tensor(mat.indices(), grad_vals, [n, n])

        if ctx.needs_input_grad[1]:
            b_grad = dg_dx.clone()
        return mat_grad, b_grad, None


def cholespy_factor_and_solve(
    mat: torch.Tensor, b: torch.Tensor, x: torch.Tensor | None = None
) -> torch.Tensor:
    return CholespyFactorAndSolve.apply(mat, b, x)[1]  # pyright: ignore[reportAssignmentType, reportOptionalSubscript]


class CholeskySolver(torch.nn.Module):
    """Cholesky solver.

    Precomputes the Cholesky decomposition of the system matrix and solves the
    system by back-substitution.
    """

    def __init__(self, mat: torch.Tensor):
        super().__init__()
        self.solver = make_solver(mat)

    def forward(self, b: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        return cholespy_solve(self.solver, b, x)


def min_quadratic_energy(
    system: torch.Tensor,
    rhs: torch.Tensor,
    known_idx: torch.Tensor,
    known_values: torch.Tensor,
) -> torch.Tensor:
    if rhs.ndim != known_values.ndim or (
        rhs.ndim > 1 and rhs.shape[-1] != known_values.shape[-1]
    ):
        raise ValueError(
            "rhs must have the same number of dim and same last dim as known_values. "
            f"rhs.shape = {rhs.shape}, known_values.shape = {known_values.shape}."
        )
    if not system.is_coalesced():
        system = system.coalesce()
    n_rows = system.shape[0]
    unknown_mask = torch.ones([n_rows], dtype=torch.bool, device=known_idx.device)
    unknown_mask[known_idx] = False
    unknown_idx = torch.nonzero(unknown_mask).flatten()
    system_uu = sp.get_slice(system, unknown_idx, unknown_idx)
    if known_idx.nelement() > 0:
        system_uk = sp.get_slice(system, unknown_idx, known_idx)
        known_part = system_uk @ known_values
    else:
        known_part = 0
    rhs_u = rhs[unknown_idx, ...]
    new_rhs = rhs_u - known_part

    if system_uu.dtype == torch.cfloat:
        # TODO: make complex numbers work on the GPU and with gradients
        system_uu_sp = sp.torch_to_scipy(system_uu.detach().cpu())
        solver = scipy.sparse.linalg.splu(system_uu_sp)
        unknown = torch.tensor(
            solver.solve(new_rhs.detach().cpu().numpy()), device=rhs.device
        )
    else:
        solver = CholeskySolver(system_uu)
        unknown = solver(new_rhs)

    result = torch.zeros_like(rhs)
    result[unknown_idx] = unknown
    result[known_idx] = known_values
    return result


def power_iteration(
    a: torch.Tensor, x: torch.Tensor, m: torch.Tensor, sigma: float | None
) -> torch.Tensor:
    if sigma is not None:
        evecs = cholespy_factor_and_solve(a - sigma * m, torch.sparse.mm(m, x))
    else:
        evecs = torch.sparse.mm(a, x)
    evecs = torch.linalg.qr(evecs)[0]
    m_evecs = torch.sparse.mm(a, evecs)
    # evecs = evecs / torch.sqrt((evecs.conj() * m_evecs).sum(dim=0, keepdim=True).real)
    dot = (x.conj() * m_evecs).sum(dim=0, keepdim=True).real
    signs = 2 * (dot > 0).int() - 1
    evecs = signs * evecs

    # Shrink spectral radius:
    lr = 0.1
    return (1 - lr) * x + lr * evecs


def power_iterations(
    A: torch.Tensor,  # noqa: N803
    M: torch.Tensor | None = None,  # noqa: N803
    k: int = 1,
    sigma: float | None = None,
    maxiter: int = 10,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = A.device
    dtype = A.dtype
    n = A.shape[0]

    evals = torch.randn([k], device=device, dtype=dtype)
    evecs = torch.randn([n, k], device=device, dtype=dtype)
    if M is None:
        M = sp.eye(n, dtype=dtype, device=device)  # noqa: N806

    # The following serves to make the gradients symmetric in the backward pass:
    A = 0.5 * (A + A.mT).coalesce()  # noqa: N806

    # TODO: cholesky on M for non-diagonal M
    m_idcs = M.to_sparse_coo().indices()
    if (m_idcs[0] != m_idcs[1]).any():
        raise ValueError("M must be diagonal.")
    M_sqrt_inv = M.clone()  # noqa: N806
    M_sqrt_inv._values().copy_(1 / M._values().sqrt())
    M_sqrt_inv = M_sqrt_inv.coalesce()  # noqa: N806

    for it in range(maxiter):
        if sigma is not None:
            z = cholespy_factor_and_solve(A - sigma * M, torch.sparse.mm(M, evecs))
        else:
            z = torch.sparse.mm(A, evecs)
        z, _ = torch.linalg.qr(z)
        # z = M_sqrt_inv @ z
        z = z / torch.sqrt(
            (z.conj() * (torch.sparse.mm(M, z))).sum(dim=0, keepdim=True).real
        )

        if torch.linalg.vector_norm(evecs - z).all() < tol:
            evecs = z
            print(f"Converged after {it} power iterations.")
            break
        evecs = z
    # Rayleigh quotient for generalized eigenvalue
    evals = (evecs.conj() * torch.sparse.mm(A, evecs)).sum(-2)

    sort_idx = torch.argsort(-torch.abs(evals - (sigma if sigma is not None else 0)))
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    return evals, evecs


def make_power_iteration_vjp(
    a: torch.Tensor, evecs: torch.Tensor, m: torch.Tensor, sigma: float | None
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    with torch.enable_grad():
        a_g = a.clone().requires_grad_(True)
        evecs_g = evecs.clone().requires_grad_(True)
        evecs_out = power_iteration(a_g, evecs_g, m, sigma)

        def vjp_evecs(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (evecs_out,),
                (evecs_g),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

        def vjp_a(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (evecs_out,),
                (a_g),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

    return vjp_evecs, vjp_a


start = time.perf_counter()
make_power_iteration_vjp = torch.compile(
    make_power_iteration_vjp,
    backend="inductor",
)
make_power_iteration_vjp(
    torch.randn([20, 20]), torch.randn([20, 3]), torch.randn([20, 20]), None
)
print("Compiling VJP took:", time.perf_counter() - start)


def fixed_point_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    z_init: torch.Tensor,
    maxiter: int,
    tol: float | None = None,
) -> torch.Tensor:
    if tol is None:
        tol = 2 * torch.finfo(z_init.dtype).eps
    z_prev, z = z_init, f(z_init)
    i = 0
    while torch.linalg.norm(z_prev - z) > tol and i < maxiter:
        z_prev, z = z, f(z)
        i += 1
        # if i % 10 == 0:
        #     print(torch.linalg.norm(z_prev - z))
    print(f"Converged after {i} iterations.")
    return z


def gmres_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    init: torch.Tensor,
    maxiter: int,
    tol: float,
) -> torch.Tensor:
    """Matrix-free GMRES solver for (I - J^T) u = b.

    Arguments:
        f (Callable[[torch.Tensor], torch.Tensor]): Computes J^T @ z  (vector-Jacobian product)
        b (torch.Tensor): Right-hand side of linear system, same shape as u.
        init (torch.Tensor): Initial guess for u.
        maxiter (int): Maximum number of iterations for the solver.
        tol (float): Minimum tolerance the solver needs to reach before exiting.

    Returns:
        (torch.Tensor): Solution to the linear system (I - J^T) u = b.
    """
    shape = b.shape
    device, dtype = b.device, b.dtype

    # Flatten all for GMRES math
    res = b.flatten() - (init.flatten() - f(init).flatten())
    res_norm = torch.norm(res)
    if res_norm < tol:
        return init

    # Krylov basis and Hessenberg matrix
    krylov_basis = [res / res_norm]
    hessenberg = torch.zeros((maxiter + 1, maxiter), dtype=dtype, device=device)
    g = torch.zeros((maxiter + 1,), dtype=dtype, device=device)
    g[0] = res_norm

    for k in range(maxiter):
        q_k = krylov_basis[k].reshape(shape)
        # Compute (I - J^T) z:
        w = (q_k - f(q_k)).flatten()

        # Modified Gram-Schmidt orthogonalization:
        for i in range(k + 1):
            hessenberg[i, k] = torch.dot(krylov_basis[i], w)
            w -= hessenberg[i, k] * krylov_basis[i]

        hessenberg[k + 1, k] = torch.norm(w)
        if hessenberg[k + 1, k] > 0:
            krylov_basis.append(w / hessenberg[k + 1, k])
        else:
            break

        # Least-squares solve min ||beta e1 - H y||
        hessenberg_k = hessenberg[: k + 2, : k + 1]
        g_k = g[: k + 2]
        y, *_ = torch.linalg.lstsq(hessenberg_k, g_k.unsqueeze(1))
        y = y.squeeze(1)

        # Residual norm estimate
        res = torch.norm(g_k - hessenberg_k @ y)
        if res < tol:
            break

    basis_mat = torch.stack(krylov_basis[: k + 1], dim=1)  # [dim, k + 1]
    y, *_ = torch.linalg.lstsq(hessenberg[: k + 2, : k + 1], g[: k + 2].unsqueeze(1))
    x_flat = init.flatten() + basis_mat @ y.squeeze(1)
    return x_flat.reshape(shape)


def estimate_spectral_radius(
    f: Callable[[torch.Tensor], torch.Tensor], init: torch.Tensor, maxiter: int
) -> torch.Tensor:
    """Estimate spectral radius (max |evals|) of matrix J (given as callable f).

    Uses the power method J^T J, via J^T(J v).

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): Function that computes J @ v.
        init (torch.Tensor): Initial guess for largest eigenvector.
        maxiter (int): Maximum number of iterations for the solver.

    Returns:
        torch.Tensor: Spectral radius, i.e., the maximum absolute eigenvalue.
    """
    v = init / init.norm()
    for _ in range(maxiter):
        w = f(f(v))
        v = w / (w.norm() + 1e-30)
    w = f(f(v))
    rho_est = torch.sqrt((v * w).sum().abs())
    return rho_est


class Eigsh(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        (a, m, _, sigma, maxiter, tol, adjoint) = inputs
        evals, evecs = output
        ctx.save_for_backward(a, m, evals, evecs)
        ctx.adjoint = adjoint
        ctx.sigma = sigma
        ctx.maxiter = maxiter
        ctx.tol = tol

    @staticmethod
    def forward(
        A: torch.Tensor,  # noqa: N803
        M: torch.Tensor | None = None,  # noqa: N803
        k: int = 1,
        sigma: float = 0.0,
        maxiter: int = 10,
        tol: float = 1e-10,
        adjoint: Literal["unroll"] = "unroll",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(A, torch.Tensor)
        assert isinstance(M, torch.Tensor)

        device = A.device
        dtype = M.dtype
        if device == torch.device("cpu"):
            A_sp = sp.torch_to_scipy(A)  # noqa: N806
            M_sp = sp.torch_to_scipy(M)  # noqa: N806
            if k < A.shape[0] - 1:
                evals, evecs = scipy.sparse.linalg.eigsh(
                    A_sp, k=k, M=M_sp, sigma=sigma, maxiter=maxiter, tol=tol
                )
            else:
                A_sp = A_sp.toarray()  # noqa: N806
                M_sp = M_sp.toarray()  # noqa: N806
                evals, evecs = scipy.linalg.eigh(A_sp, b=M_sp)
            sort_idx = np.argsort(-np.abs(evals))
            evals = evals[sort_idx][:k]
            evecs = evecs[:, sort_idx][:, :k]
            return torch.tensor(evals.real, dtype=dtype, device=device), torch.tensor(
                evecs.real, dtype=dtype, device=device
            )
        else:
            raise NotImplementedError(
                f"CUDA sparse eigensolver with adjoint {adjoint} not implemented."
            )

    @staticmethod
    def backward(
        ctx, grad_evals: torch.Tensor | None, grad_evecs: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        a, m, evals, evecs = ctx.saved_tensors
        n: int = a.shape[0]
        k: int = evecs.shape[1]
        device = a.device
        dtype = a.dtype

        grad_a = None
        if ctx.needs_input_grad[0]:
            if ctx.adjoint == "truncate":
                if grad_evals is None:
                    grad_evals = torch.zeros(k, dtype=dtype, device=device)

                if grad_evecs is None:
                    grad_evecs = torch.zeros(n, k, dtype=dtype, device=device)
                tilde_g = evecs.mT @ grad_evecs  # (k, k)

                diff = evals[None, :] - evals[:, None]  # (k, k)
                one_over_diff = torch.zeros_like(diff)
                eps = torch.finfo(diff.dtype).eps
                mask = ~torch.eye(k, dtype=torch.bool, device=diff.device)
                one_over_diff[mask] = 1.0 / (
                    diff[mask] + (diff[mask] == 0).to(diff.dtype) * eps
                )

                s = 0.5 * (tilde_g - tilde_g.mT) * one_over_diff
                h = s + torch.diag(grad_evals)

                # Sparse outer product evecs @ h @ evecs.mT
                rows, cols = a.indices()
                grad_vals = -(evecs @ h)[rows, ...] * evecs[cols, ...]
                if grad_vals.ndim == 2:
                    grad_vals = grad_vals.sum(-1)
                grad_a = torch.sparse_coo_tensor(a.indices(), grad_vals, [n, n])
            if ctx.adjoint == "individual":
                dg_dx = torch.zeros_like(evecs)
                eye = sp.eye(a.shape[0], device=a.device, dtype=dtype)
                lhs_i = torch.arange(n, device=device, dtype=torch.int64)
                lhs_j = torch.full([n], n, device=device, dtype=torch.int64)
                for i in range(evals.shape[0]):
                    lhs_aa = a - evals[i] * eye
                    lhs = torch.sparse_coo_tensor(
                        lhs_aa.indices(), lhs_aa.values(), size=[n + 1, n + 1]
                    )
                    lhs = sp.append(lhs, torch.stack([lhs_i, lhs_j]), -evecs[:, i])
                    lhs = sp.append(lhs, torch.stack([lhs_j, lhs_i]), -evecs[:, i])
                    if grad_evals is None:
                        rhs = torch.nn.functional.pad(grad_evecs[:, i], (0, 1))
                    else:
                        rhs = torch.cat((grad_evecs[:, i], grad_evals[i : i + 1]), -1)
                    lhs_scipy = sp.torch_to_scipy(lhs).tocsr()
                    rhs_numpy = rhs.cpu().detach().numpy()
                    result = scipy.sparse.linalg.spsolve(lhs_scipy, rhs_numpy)[:n]
                    dg_dx[:, i] = torch.tensor(result, device=device)

                # Compute masked outer product:
                # (dg/dx) @ x^T
                rows, cols = a.indices()
                grad_vals = -dg_dx[rows, ...] * evecs[cols, ...]
                if grad_vals.ndim == 2:
                    grad_vals = grad_vals.sum(-1)
                grad_a = torch.sparse_coo_tensor(a.indices(), grad_vals, [n, n])
            elif ctx.adjoint in ("dodik-invert", "dodik-fixedpoint"):
                if ctx.adjoint == "dodik-invert":

                    def implicit_func(a, x):
                        return power_iteration(a, x, m, ctx.sigma) - x

                    # Test full formula:
                    df_da = torch.func.jacrev(implicit_func, 0)
                    df_dx = torch.func.jacrev(implicit_func, 1)

                    df_da_mat = df_da(a, evecs).reshape(n * k, n * n)
                    df_dx_mat = df_dx(a, evecs).reshape(n * k, n * k)
                    dx_da_mat = -torch.linalg.solve(df_dx_mat, df_da_mat)
                    grad_a = (grad_evecs.flatten() @ dx_da_mat).reshape(n, n)

                elif ctx.adjoint == "dodik-fixedpoint":
                    # TODO: eigenvalue loss

                    init = torch.randn_like(evecs)

                    # rho = estimate_spectral_radius(lambda z: vjp(z)[1], init, 10)
                    # print("RHO", rho)

                    # print(init, power_iteration(a, init), vjp(init)[1])
                    # print(vjp(init)[1])
                    # u = fixed_point_solver(
                    #     lambda z: grad_evecs + vjp(z)[1], init, ctx.maxiter, ctx.tol
                    # )

                    start = time.perf_counter()
                    vjp_evecs, vjp_a = make_power_iteration_vjp(a, evecs, m, ctx.sigma)
                    print("Making VJP took:", time.perf_counter() - start)
                    u = gmres_solve(
                        lambda z: vjp_evecs(z)[0],
                        grad_evecs,
                        init,
                        maxiter=ctx.maxiter,
                        tol=ctx.tol,
                    )

                grad_a = vjp_a(u)[0]

        if grad_a is not None:
            # Ensure symmetry of gradients:
            grad_a = 0.5 * (grad_a + grad_a.mT)
        return grad_a, None, None, None, None, None, None


def eigsh(
    A: torch.Tensor,  # noqa: N803
    M: torch.Tensor | None = None,  # noqa: N803
    k: int = 1,
    sigma: float | None = None,
    maxiter: int = 100,
    tol: float = 0,
    adjoint: Literal[
        "unroll", "individual", "truncate", "dodik-fixedpoint", "dodik-invert"
    ] = "individual",
) -> tuple[torch.Tensor, torch.Tensor]:
    if adjoint == "unroll":
        return power_iterations(A, M, k=k, sigma=sigma, maxiter=maxiter, tol=tol)
    else:
        return Eigsh.apply(A, M, k, sigma, maxiter, tol, adjoint)

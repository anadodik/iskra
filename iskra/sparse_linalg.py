# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from typing import Callable, Literal, TypeAlias

import numpy as np
import scipy.sparse
import torch
from cholespy import CholeskySolverD, CholeskySolverF, MatrixType

try:
    from sksparse import cholmod

    _cholmod_available = True
except ImportError:
    _cholmod_available = False

try:
    import cuda.core.experimental as cudax
    import nvmath.sparse.advanced as nvsparse

    _nvmath_available = True
except ImportError:
    _nvmath_available = False

import iskra.sparse as sp
from iskra.logging.logging import getLogger
from iskra.profiling import profile_block, profile_fn

SolverT: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

LOGGER = getLogger(__name__)


class CholmodSolver:
    def __init__(
        self,
        mat: torch.Tensor,
        analyze_only: bool = False,
        mode="supernodal",
        ordering_method="amd",
    ):
        with profile_block("solver_init"):
            if analyze_only:
                self.solver = cholmod.analyze(
                    sp.torch_to_scipy(mat), mode=mode, ordering_method=ordering_method
                )
            else:
                self.solver = cholmod.cholesky(
                    sp.torch_to_scipy(mat), mode=mode, ordering_method=ordering_method
                )

    @profile_fn(name="refactor_numeric")
    def refactor_numeric(self, mat: torch.Tensor):
        mat_sp = sp.torch_to_scipy(mat).tocsc()
        self.solver.cholesky_inplace(mat_sp)

    def __call__(self, b: torch.Tensor):
        b_np = b.detach().cpu().numpy()
        x_np = np.ascontiguousarray(self.solver(b_np))
        return torch.tensor(x_np, dtype=b.dtype, device=b.device)


class CUDSSSolver:
    def __init__(self, mat: torch.Tensor, analyze_only: bool = False):
        with profile_block("solver_init"):
            device = mat.device
            dtype = mat.dtype

            self.dummy_b = torch.zeros((1, mat.shape[0]), dtype=dtype, device=device).mT
            self.mat_csr = mat.to_sparse_csr()

            self.options = self.new_default_options()

            self.solver = nvsparse.DirectSolver(
                self.mat_csr, self.dummy_b, options=self.options
            )
            self.solver.plan()

            if not analyze_only:
                self.solver.factorize()

    def new_default_options(self):
        self.options = nvsparse.DirectSolverOptions()
        self.options.multithreading_lib = "openmp"
        self.options.blocking = True

    @profile_fn(name="refactor_numeric")
    def refactor_numeric(self, mat: torch.Tensor):
        new_values = mat.to_sparse_csr().values()
        self.mat_csr.values().data.copy_(new_values)
        self.solver.factorize()

    def __call__(self, b: torch.Tensor):
        device = b.device
        device_id = device.index if device.type == "cuda" else 0
        cudax.Device(device_id).set_current()
        if b.ndim > 1 and b.shape[-1] != 1:
            b_reshaped = b.mT.contiguous().mT.clone()
        elif b.ndim > 1 and b.shape[-1] == 1:
            b_reshaped = b[..., 0].clone()
        else:
            b_reshaped = b.clone()
        if b_reshaped.shape != self.dummy_b.shape:
            self.dummy_b = b_reshaped
            self.options = self.new_default_options()
            self.mat_csr = self.mat_csr.clone()
            self.solver.free()
            self.solver = nvsparse.DirectSolver(
                self.mat_csr, self.dummy_b, options=self.options
            )
            self.solver.plan()
            self.solver.factorize()
        else:
            self.solver.reset_operands(None, b_reshaped)
        x: torch.Tensor = self.solver.solve()
        torch.cuda.default_stream().synchronize()
        if b.ndim > 1 and b.shape[-1] == 1:
            return x[..., None]
        else:
            return x.clone()

    def __del__(self):
        if hasattr(self, "solver"):
            try:
                self.solver.free()
                self.solver = None
                torch.cuda.default_stream().synchronize()
            except Exception as e:
                print(f"Error while freeing solver: {e}")


def default_solver(
    mat: torch.Tensor, analyze_only: bool = False, is_psd: bool = True
) -> SolverT:
    # TODO: better support for LU
    mat = mat.coalesce()
    if is_psd:
        if mat.is_cpu and _cholmod_available:
            _solve_fn = CholmodSolver(mat, analyze_only=analyze_only)
        elif mat.is_cuda and _nvmath_available:
            _solve_fn = CUDSSSolver(mat, analyze_only=analyze_only)
        elif mat.device != torch.device("cpu") or not _cholmod_available:
            if mat.dtype == torch.float32:
                solver_t = CholeskySolverF
            elif mat.dtype == torch.float64:
                solver_t = CholeskySolverD
            else:
                raise TypeError(
                    f"Cholespy only supports f32 and f64 matrices, found {mat.dtype}."
                )
            solver = solver_t(
                mat.shape[0],
                mat.indices()[0],
                mat.indices()[1],
                mat.values(),
                MatrixType.COO,
            )

            def _solve_fn(b: torch.Tensor) -> torch.Tensor:
                x = torch.zeros_like(b)
                solver.solve(b, x)
                return x
    else:
        solver = scipy.sparse.linalg.splu(sp.torch_to_scipy(mat.detach().cpu()))

        def _solve_fn(b: torch.Tensor) -> torch.Tensor:
            b_np = b.detach().cpu().numpy()
            x_np = solver.solve(b_np)
            x = torch.tensor(x_np, dtype=b.dtype, device=b.device)
            return x

    return _solve_fn


@profile_fn(name="_call_solver")
def _call_solver(solver: SolverT, b: torch.Tensor) -> torch.Tensor:
    b = b.contiguous()
    return solver(b)


class LinearFactorAndSolve(torch.autograd.Function):
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
        solver_fn: SolverT | None = None,
    ) -> tuple[SolverT, torch.Tensor]:
        if not mat.is_coalesced():
            raise ValueError("Matrix not coalesced, please call .coalesce() first.")
        if solver_fn is None:
            solver_fn = default_solver(mat)
            assert solver_fn is not None
        x = _call_solver(solver_fn, b)
        return solver_fn, x

    @staticmethod
    def backward(
        ctx, _: None, forward_grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        forward_grad = forward_grad.contiguous()
        n = forward_grad.shape[0]
        mat, _, x = ctx.saved_tensors
        b_grad: torch.Tensor | None = None
        mat_grad: torch.Tensor | None = None

        dg_dx = _call_solver(ctx.solver, forward_grad)
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


@profile_fn(name="linear_solve")
def linear_solve(
    mat: torch.Tensor,
    b: torch.Tensor,
    solver_fn: SolverT | None = None,
) -> tuple[SolverT, torch.Tensor]:
    result: tuple[SolverT, torch.Tensor] = LinearFactorAndSolve.apply(mat, b, solver_fn)  # pyright: ignore[reportAssignmentType, reportReturnType]
    return result


def quad_energy_mat(mat: torch.Tensor, unknown_idx: torch.Tensor) -> torch.Tensor:
    return sp.get_slice(mat, unknown_idx, unknown_idx)


@profile_fn(name="quad_energy_rhs")
def quad_energy_rhs(
    mat: torch.Tensor,
    rhs: torch.Tensor,
    known_idx: torch.Tensor,
    known_values: torch.Tensor,
    unknown_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if unknown_idx is None:
        unknown_idx = sp.index_complement(rhs.shape[0], known_idx)
    if known_idx.nelement() > 0:
        mat_uk = sp.get_slice(mat, unknown_idx, known_idx)
        return rhs[unknown_idx, ...] - mat_uk @ known_values
    else:
        return rhs


def min_quadratic_energy(
    mat: torch.Tensor,
    rhs: torch.Tensor,
    known_idx: torch.Tensor,
    known_values: torch.Tensor,
    solver: SolverT | None = None,
) -> tuple[SolverT, torch.Tensor]:
    if rhs.ndim != known_values.ndim or (
        rhs.ndim > 1 and rhs.shape[-1] != known_values.shape[-1]
    ):
        raise ValueError(
            "rhs must have the same number of dim and same last dim as known_values. "
            f"rhs.shape = {rhs.shape}, known_values.shape = {known_values.shape}."
        )
    if not mat.is_coalesced():
        mat = mat.coalesce()
    unknown_idx = sp.index_complement(rhs.shape[0], known_idx)
    new_rhs = quad_energy_rhs(mat, rhs, known_idx, known_values, unknown_idx)

    mat_uu = sp.get_slice(mat, unknown_idx, unknown_idx)
    solver, unknown = linear_solve(mat_uu, new_rhs, solver_fn=solver)

    result = torch.zeros_like(rhs)
    result[unknown_idx] = unknown
    result[known_idx] = known_values
    return solver, result


def power_iteration(
    a: torch.Tensor,
    x: torch.Tensor,
    m: torch.Tensor,
    sigma: float | None,
    solver: SolverT | None = None,
) -> torch.Tensor:
    if sigma is not None:
        evecs = linear_solve(a - sigma * m, torch.sparse.mm(m, x), solver)[1]
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

    if sigma is not None:
        system = A - sigma * M
        solver = default_solver(system)

    A = A.to_sparse_csr()
    M = M.to_sparse_csr()
    for it in range(maxiter):
        if sigma is not None:
            solver, z = linear_solve(system, sp.matmul(M, evecs), solver)
        else:
            z = sp.matmul(A, evecs)
        z, _ = torch.linalg.qr(z)
        # z = M_sqrt_inv @ z
        z = z / torch.sqrt((z.conj() * (sp.matmul(M, z))).sum(dim=0, keepdim=True).real)

        if torch.linalg.vector_norm(evecs - z).all() < tol:
            evecs = z
            print(f"Converged after {it} power iterations.")
            break
        evecs = z
    # Rayleigh quotient for generalized eigenvalue
    evals = (evecs.conj() * sp.matmul(A, evecs)).sum(-2)

    sort_idx = torch.argsort(-torch.abs(evals - (sigma if sigma is not None else 0)))
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    return evals, evecs


def make_power_iteration_vjp(
    a: torch.Tensor,
    evecs: torch.Tensor,
    m: torch.Tensor,
    sigma: float | None,
    solver: SolverT | None = None,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    with torch.enable_grad():
        a_g = a.clone().requires_grad_(True)
        evecs_g = evecs.clone().requires_grad_(True)
        if sigma is not None and solver is None:
            solver = default_solver(a - sigma * m)
        evecs_out = power_iteration(a_g, evecs_g, m, sigma, solver=solver)
        evecs_out = evecs_out - evecs_g

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


# start = time.perf_counter()
# make_power_iteration_vjp = torch.compile(
#     make_power_iteration_vjp, backend="inductor", dynamic=True
# )
# print("Compiling VJP took:", time.perf_counter() - start)


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
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Matrix-free GMRES solver for J^T u = b.

    Arguments:
        f (Callable[[Tensor], Tensor]): Function which computes
            J^T @ z (e.g., a vector-Jacobian product).
        b (Tensor): Right-hand side of linear system, same shape as u.
        init (Tensor): Initial guess for u.
        max_iter (int): Maximum number of iterations for the solver.
        abs_tol (float): Absolute tolerance the solver needs to reach before exiting.
        rel_tol (float): Relative tolerance the solver needs to reach before exiting.
        preconditioner (Optional[Callable]): Function representing M^-1.
            If provided, the solver solves the left-preconditioned system
            M^-1 A u = M^-1 b.
        verbose (bool): Whether to print logging information.

    Returns:
        (torch.Tensor): Solution to the linear system, u.
    """
    if verbose:
        LOGGER.setLevel("INFO")
    shape = b.shape
    device, dtype = b.device, b.dtype
    dim = b.numel()

    b_flat = b.flatten()
    init_flat = init.flatten()

    res = b_flat - f(init).flatten()
    if preconditioner is not None:
        res = preconditioner(res.reshape(shape)).flatten()

    initial_res_norm = torch.linalg.norm(res)
    if initial_res_norm < abs_tol:
        LOGGER.warning("Early exiting GMRES, initial guess is below norm.")
        return init

    krylov_basis = torch.zeros((dim, max_iter + 1), dtype=dtype, device=device)
    krylov_basis[:, 0] = res / initial_res_norm

    hessenberg = torch.zeros((max_iter + 1, max_iter), dtype=dtype, device=device)

    cos_vals = torch.zeros(max_iter, dtype=dtype, device=device)
    sin_vals = torch.zeros(max_iter, dtype=dtype, device=device)

    g = torch.zeros(max_iter + 1, dtype=dtype, device=device)
    g[0] = initial_res_norm

    k = 0
    for k in range(max_iter):
        q_k = krylov_basis[:, k].reshape(shape)
        w = f(q_k).flatten()

        if preconditioner is not None:
            w = preconditioner(w.reshape(shape)).flatten()

        hessenberg[: k + 1, k] = krylov_basis[:, : k + 1].T @ w
        w = w - krylov_basis[:, : k + 1] @ hessenberg[: k + 1, k]

        hessenberg[k + 1, k] = torch.linalg.norm(w)
        if hessenberg[k + 1, k] > 1e-12:
            krylov_basis[:, k + 1] = w / hessenberg[k + 1, k]
        else:
            break

        for i in range(k):
            temp = cos_vals[i] * hessenberg[i, k] + sin_vals[i] * hessenberg[i + 1, k]
            hessenberg[i + 1, k] = (
                -sin_vals[i] * hessenberg[i, k] + cos_vals[i] * hessenberg[i + 1, k]
            )
            hessenberg[i, k] = temp

        h_k = hessenberg[k, k]
        h_k1 = hessenberg[k + 1, k]
        rho = torch.sqrt(h_k**2 + h_k1**2)
        cos_vals[k] = h_k / rho
        sin_vals[k] = h_k1 / rho

        hessenberg[k, k] = rho
        hessenberg[k + 1, k] = 0.0

        g[k + 1] = -sin_vals[k] * g[k]
        g[k] = cos_vals[k] * g[k]

        res = abs(g[k + 1])

        if verbose:
            abs_val = res.cpu().detach().item()
            rel_val = (
                (res / initial_res_norm).cpu().detach().item()
                if initial_res_norm > 0
                else 0.0
            )
            LOGGER.debug(
                f"GMRES residuals: res={abs_val:.3e}, "
                f"(tol={abs_tol + rel_tol * initial_res_norm:.3e}), "
                f"res_rel={rel_val:.3e} (rel_tol={rel_tol:.3e})."
            )
        if res <= abs_tol + rel_tol * initial_res_norm:
            break
    if verbose:
        LOGGER.info(f"GMRES exiting after {k + 1} iterations, residual: {res:.3e}")
    if res > abs_tol + rel_tol * initial_res_norm:
        abs_val = res.cpu().detach().item()
        rel_val = (
            (res / initial_res_norm).cpu().detach().item()
            if initial_res_norm > 0
            else 0.0
        )
        LOGGER.warning(
            f"GMRES did not converge within {max_iter} iterations.\n"
            f"Residuals: abs={abs_val:.3e} "
            f"(tol={abs_tol + rel_tol * initial_res_norm:.3e}), "
            f"rel={rel_val:.3e} (rel_tol={rel_tol:.3e})."
        )

    y = torch.zeros(k + 1, dtype=dtype, device=device)
    for i in range(k, -1, -1):
        y[i] = (g[i] - hessenberg[i, i + 1 : k + 1] @ y[i + 1 : k + 1]) / hessenberg[
            i, i
        ]

    correction = krylov_basis[:, : k + 1] @ y

    x_flat = init_flat + correction
    return x_flat.reshape(shape)


def build_diagonal_preconditioner(f, shape, device, dtype):
    diag = torch.zeros(shape.numel(), device=device, dtype=dtype)
    e_i = torch.zeros(shape.numel(), device=device, dtype=dtype)
    for i in range(shape.numel()):
        e_i[i] = 1.0
        e_i_shaped = e_i.reshape(shape)
        diag[i] = 1 - f(e_i_shaped).flatten()[i]
        e_i[i] = 0.0

    diag_inv = 1.0 / (diag + 1e-8)

    def preconditioner(v):
        return (v.flatten() * diag_inv).reshape(shape)

    return preconditioner


def build_sampled_diagonal_preconditioner(f, shape, device, dtype, n_samples=500):
    dim = shape.numel()
    indices = torch.randperm(dim, device=device)[:n_samples]

    diag_samples = torch.zeros(n_samples, device=device, dtype=dtype)
    e_i = torch.zeros(dim, device=device, dtype=dtype)
    for idx, i in enumerate(indices):
        e_i[i] = 1.0
        diag_samples[idx] = f(e_i.reshape(shape)).flatten()[i]
        e_i[i] = 0.0

    scale = 1.0 / (diag_samples.mean() + 1e-8)

    def preconditioner(v):
        return v * scale

    return preconditioner


# def gmres_solve(
#     f: Callable[[torch.Tensor], torch.Tensor],
#     b: torch.Tensor,
#     init: torch.Tensor,
#     maxiter: int,
#     tol: float,
# ) -> torch.Tensor:
#     """Matrix-free GMRES solver for (I - J^T) u = b.

#     Arguments:
#         f (Callable[[torch.Tensor], torch.Tensor]): Computes J^T @ z  (vector-Jacobian product)
#         b (torch.Tensor): Right-hand side of linear system, same shape as u.
#         init (torch.Tensor): Initial guess for u.
#         maxiter (int): Maximum number of iterations for the solver.
#         tol (float): Minimum tolerance the solver needs to reach before exiting.

#     Returns:
#         (torch.Tensor): Solution to the linear system (I - J^T) u = b.
#     """
#     shape = b.shape
#     device, dtype = b.device, b.dtype

#     res = b.flatten() - f(init).flatten()
#     res_norm = torch.norm(res)
#     if res_norm < tol:
#         return init

#     # Krylov basis and Hessenberg matrix
#     krylov_basis = [res / res_norm]
#     hessenberg = torch.zeros((maxiter + 1, maxiter), dtype=dtype, device=device)
#     g = torch.zeros((maxiter + 1,), dtype=dtype, device=device)
#     g[0] = res_norm

#     for k in range(maxiter):
#         q_k = krylov_basis[k].reshape(shape)
#         # Compute (I - J^T) z:
#         w = f(q_k).flatten()

#         # Modified Gram-Schmidt orthogonalization:
#         for i in range(k + 1):
#             hessenberg[i, k] = torch.dot(krylov_basis[i], w)
#             w -= hessenberg[i, k] * krylov_basis[i]

#         hessenberg[k + 1, k] = torch.norm(w)
#         if hessenberg[k + 1, k] > 0:
#             krylov_basis.append(w / hessenberg[k + 1, k])
#         else:
#             break

#         # Least-squares solve min ||beta e1 - H y||
#         hessenberg_k = hessenberg[: k + 2, : k + 1]
#         g_k = g[: k + 2]
#         with profile_block("lstsq"):
#             y, *_ = torch.linalg.lstsq(hessenberg_k, g_k.unsqueeze(1))
#         y = y.squeeze(1)

#         res = torch.norm(g_k - hessenberg_k @ y)
#         if res < tol:
#             break
#     print(f"GMRES exiting after {k} iterations, residual: {res}.")


#     basis_mat = torch.stack(krylov_basis[: k + 1], dim=1)  # [dim, k + 1]
#     y, *_ = torch.linalg.lstsq(hessenberg[: k + 2, : k + 1], g[: k + 2].unsqueeze(1))
#     x_flat = init.flatten() + basis_mat @ y.squeeze(1)
#     # res = torch.norm(b.flatten() - f(x_flat.reshape(shape)).flatten())
#     return x_flat.reshape(shape)


def cg_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    init: torch.Tensor,
    maxiter: int,
    tol: float,
) -> torch.Tensor:
    """Matrix-free Conjugate Gradient solver for (I - J^T) u = b.

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

    x = init.clone()
    r = b - f(x)
    r_flat = r.flatten()

    rsold = torch.dot(r_flat, r_flat)
    if torch.sqrt(rsold) < tol:
        return x

    p = r.clone()

    for k in range(maxiter):
        Ap = f(p)
        Ap_flat = Ap.flatten()
        p_flat = p.flatten()

        alpha = rsold / torch.dot(p_flat, Ap_flat)
        x = x + alpha * p
        r = r - alpha * Ap
        r_flat = r.flatten()

        rsnew = torch.dot(r_flat, r_flat)
        res = torch.sqrt(rsnew)

        if res < tol:
            print(f"CG exiting after {k + 1} iterations, residual: {res:.6e}")
            return x

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    res = torch.sqrt(rsold)
    print(f"CG exiting after {maxiter} iterations, residual: {res:.6e}")
    return x


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
        (a, m, _, sigma, maxiter, tol, bwd_method, bwd_max_iter, bwd_eps) = inputs
        evals, evecs = output
        ctx.save_for_backward(a, m, evals, evecs)
        ctx.adjoint = bwd_method
        ctx.sigma = sigma
        ctx.maxiter = maxiter
        ctx.tol = tol
        ctx.bwd_max_iter = bwd_max_iter
        ctx.bwd_eps = bwd_eps

    @staticmethod
    def forward(
        A: torch.Tensor,  # noqa: N803
        M: torch.Tensor | None = None,  # noqa: N803
        k: int = 1,
        sigma: float = 0.0,
        maxiter: int = 10,
        tol: float = 1e-10,
        bwd_method: Literal["unroll"] = "unroll",
        bwd_max_iter: int = 200,
        bwd_eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(A, torch.Tensor)
        assert isinstance(M, torch.Tensor)

        device = A.device
        dtype = M.dtype
        if A.is_cpu:
            A_sp = sp.torch_to_scipy(A)  # noqa: N806
            M_sp = sp.torch_to_scipy(M)  # noqa: N806
            if k < A.shape[0] - 1:
                evals, evecs = scipy.sparse.linalg.eigsh(
                    A_sp, k=k, sigma=sigma, maxiter=maxiter, tol=tol
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
                f"CUDA sparse eigensolver with adjoint {bwd_method} not implemented."
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
            elif ctx.adjoint == "individual":
                dg_dx = torch.zeros_like(evecs)
                eye = sp.eye(a.shape[0], device=a.device, dtype=dtype)
                lhs_i = torch.arange(n, device=device, dtype=torch.int64)
                lhs_j = torch.full([n], n, device=device, dtype=torch.int64)
                for i in range(evals.shape[0]):
                    lhs_aa = (a - evals[i] * eye).coalesce()
                    lhs = torch.sparse_coo_tensor(
                        lhs_aa.indices(), lhs_aa.values(), size=[n + 1, n + 1]
                    ).coalesce()
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
            elif ctx.adjoint == "dodik-invert":

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

                init = torch.zeros_like(evecs)

                # rho = estimate_spectral_radius(lambda z: vjp(z)[1], init, 10)
                # print("RHO", rho)

                # print(init, power_iteration(a, init), vjp(init)[1])
                # print(vjp(init)[1])
                # u = fixed_point_solver(
                #     lambda z: grad_evecs + vjp(z)[1], init, ctx.maxiter, ctx.tol
                # )

                # start = time.perf_counter()
                vjp_evecs, vjp_a = make_power_iteration_vjp(a, evecs, m, ctx.sigma)
                # print("Making VJP took:", time.perf_counter() - start)
                u = -gmres_solve(
                    lambda z: vjp_evecs(z)[0],
                    grad_evecs,
                    init,
                    max_iter=ctx.bwd_max_iter,
                    abs_tol=ctx.bwd_eps,
                    rel_tol=0,
                )

                grad_a = vjp_a(u)[0]

        if grad_a is not None:
            # Ensure symmetry of gradients:
            grad_a = 0.5 * (grad_a + grad_a.mT)
        return grad_a, None, None, None, None, None, None, None, None


def eigsh(
    A: torch.Tensor,  # noqa: N803
    M: torch.Tensor | None = None,  # noqa: N803
    k: int = 1,
    sigma: float | None = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    bwd_method: Literal[
        "unroll", "individual", "truncate", "dodik-fixedpoint", "dodik-invert"
    ] = "individual",
    bwd_max_iter: int = 100,
    bwd_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bwd_method == "unroll":
        return power_iterations(A, M, k=k, sigma=sigma, maxiter=maxiter, tol=tol)
    else:
        return Eigsh.apply(
            A, M, k, sigma, maxiter, tol, bwd_method, bwd_max_iter, bwd_eps
        )

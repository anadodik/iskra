# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from typing import Callable, Literal, TypeAlias

import numpy as np
import scipy.sparse
import torch
from cholespy import CholeskySolverD, CholeskySolverF, MatrixType

import iskra.sparse as sp
from iskra.logging.logging import getLogger
from iskra.profiling import profile_block, profile_fn

LOGGER = getLogger(__name__)

try:
    import sksparse
    from sksparse import cholmod

    if int(sksparse.__version__.split(".")[1]) > 4:
        LOGGER.warning(
            "Wrong scikit-sparse version installed. "
            f"Got {sksparse.__version__}, but need 0.4."
        )
        raise ImportError

    _cholmod_available = True
except ImportError:
    _cholmod_available = False

try:
    import cuda.core.experimental as cudax
    import nvmath.sparse.advanced as nvsparse

    _nvmath_available = True
except ImportError:
    _nvmath_available = False

SolverT: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


class CholmodSolver:
    def __init__(
        self,
        mat: sp.SparseTensor,
        analyze_only: bool = False,
        mode: Literal["auto", "simplicial", "supernodal"] = "supernodal",
        ordering_method: Literal[
            "default",
            "best",
            "natural",
            "metis",
            "nesdis",
            "amd",
            "colamd",
            "postordered",
        ] = "amd",
    ):
        """A wrapper around scikit-sparse's Cholmod sparse linear system solver.

        The library is not installed by default and Cholmod's licencing requires you to
        install version 0.4.6 separately on your machine.

        See the `scikit-sparse docs <https://scikit-sparse.readthedocs.io/en/latest/index.html>`_
        for more information on how to install the solvers and what the different solver
        options mean.

        Args:
            mat (SparseTensor[Float, [N, N]]): Matrix to invert.
            analyze_only (bool, optional): Only analyze the system or also factorize it.
                Defaults to False.
            mode (Literal["auto", "simplicial", "supernodal"]): Type of factorization
                to use.
            ordering_method (Literal["default", "best", "natural", "metis", "nesdis", "amd", "colamd", "postordered"]):
                Matrix permutation for Cholesky.
        """  # noqa: E501
        if not _cholmod_available:
            raise ValueError(
                "scikit-sparse and Cholmod need to be installed "
                "and available to use CholmodSolver."
            )
        with profile_block("solver_init"):
            if analyze_only:
                self.solver = cholmod.analyze(
                    sp.to_scipy(mat), mode=mode, ordering_method=ordering_method
                )
            else:
                self.solver = cholmod.cholesky(
                    sp.to_scipy(mat), mode=mode, ordering_method=ordering_method
                )

    @profile_fn(name="refactor_numeric")
    def refactor_numeric(self, mat: sp.SparseTensor):
        """Refactor the matrix numerically, reusing the existing symbolic factorization.

        Args:
            mat (sp.SparseTensor): Matrix to factor. Must have the same nonzero indices
                as the one used for the symbolic factorization.
        """
        mat_sp = sp.to_scipy(mat).tocsc()
        self.solver.cholesky_inplace(mat_sp)

    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        """Solve the linear system using the prefactored matrix.

        Args:
            b (Tensor[Float, [N]]): Right-hand side.

        Returns:
            Tensor[Float, [N]]: Linear system solution.
        """
        b_np = b.detach().cpu().numpy()
        x_np = np.ascontiguousarray(self.solver(b_np))
        return torch.tensor(x_np, dtype=b.dtype, device=b.device)


class CUDSSSolver:
    def __init__(self, mat: sp.SparseTensor, analyze_only: bool = False):
        """A wrapper around nvmath's CUDSS sparse linear system solver.

        Args:
            mat (SparseTensor[Float, [N, N]]): Matrix to invert.
            analyze_only (bool, optional): Only analyze the system or also factorize it.
                Defaults to False.
        """
        if not _nvmath_available:
            raise ValueError(
                "nvmath needs to be installed and available to use CUDSSSolver."
            )
        with profile_block("solver_init"):
            device = mat.device
            dtype = mat.dtype

            self.dummy_b = torch.zeros((1, mat.shape[0]), dtype=dtype, device=device).mT
            self.mat_csr = mat.to_sparse_csr()

            self.options = self.new_default_options()
            self.solver = nvsparse.DirectSolver(
                self.mat_csr.torch_tensor(),
                self.dummy_b,
                options=self.options,
            )
            self.solver.plan()

            if not analyze_only:
                self.solver.factorize()

    def new_default_options(self):
        self.options = nvsparse.DirectSolverOptions()
        self.options.multithreading_lib = "openmp"
        self.options.blocking = True

    @profile_fn(name="refactor_numeric")
    def refactor_numeric(self, mat: sp.SparseTensor):
        """Refactor the matrix numerically, reusing the existing symbolic factorization.

        Args:
            mat (sp.SparseTensor): Matrix to factor. Must have the same nonzero indices
                as the one used for the symbolic factorization.
        """
        new_values = mat.to_sparse_csr().values()
        self.mat_csr.values().data.copy_(new_values)
        self.solver.factorize()

    def __call__(self, b: torch.Tensor):
        """Solve the linear system using the prefactored matrix.

        Args:
            b (Tensor[Float, [Bs, N]]): Right-hand side.

        Returns:
            Tensor[Float, [Bs, N]]: Linear system solution.
        """
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
                self.mat_csr.torch_tensor(),
                self.dummy_b,
                options=self.options,
            )
            self.solver.plan()
            self.solver.factorize()
        else:
            self.solver.reset_operands(b=b_reshaped)
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
    mat: sp.SparseTensor, analyze_only: bool = False, is_psd: bool = True
) -> SolverT:
    """Reasonably guesses a good solver based on your matrix and local environment.

    For PSD matrices, it defaults to Cholmod on the CPU and CUDSS on the GPU,
    *provided these are installed and available in your environment*. See
    `CholmodSolver` and `CUDSSSolver` for more information. If these are not available,
    it falls back to cholespy's solver.

    For non-psd matrices, it defaults to scipy's `scipy.sparse.linalg.splu`, regardless
    of whether we are on the CPU or GPU. GPU tensors will incur a roundtrip performance
    penalty.

    Args:
        mat (SparseTensor[Float, [N, N]]): Matrix representing the linear system.
        analyze_only (bool, optional): Whether to perform symbolic factorization only,
            for solvers which support manualy separating stages. Defaults to False.
        is_psd (bool, optional): Whether the matrix is symmetric positive semidefinite.
            Defaults to True.

    Raises:
        TypeError: Raises if a PSD matrix with is passed but its datatype is
            not floating point.

    Returns:
        SolverT: Solver with prefactored system.
    """
    # TODO: better support for LU
    mat = mat.coalesce()
    is_float = mat.dtype == torch.float32 or mat.dtype == torch.float64
    if is_psd and is_float:
        if mat.is_cuda and not _nvmath_available:
            LOGGER.warning(
                "Requesting a CUDA sparse linear solver but nvmath is not installed. "
                "Performance will be degraded. "
                "Install it via `pip install nvmath-python[cuXX-dx]`."
            )
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
        solver = scipy.sparse.linalg.splu(sp.to_scipy(mat.detach().cpu()))

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


class _LinearFactorAndSolve(torch.autograd.Function):
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
            mat_grad = sp.coo_tensor(mat.indices(), grad_vals, [n, n])

        if ctx.needs_input_grad[1]:
            b_grad = dg_dx.clone()
        return mat_grad, b_grad, None


@profile_fn(name="linear_solve")
def linear_solve(
    mat: torch.Tensor,
    b: torch.Tensor,
    solver_fn: SolverT | None = None,
) -> tuple[SolverT, torch.Tensor]:
    """Perform a sparse linear solve for Ax = b.

    Example:
        For example, you can solve a linear system repeatedly in a loop as follows:
        ```
        x = x_init
        solver = None
        for step in range(n_steps):
            solver, x = linear_solve(mat, x, solver)
        ```


    Args:
        mat (Tensor[Float, [N, N]]): Matrix $A$ to invert.
        b (Tensor[Float, [N]]): Right-hand side $b$.
        solver_fn (SolverT | None, optional): Solver to use for the forward and
            backward pass. If None, one will be constructed via `default_solver()`.

    Returns:
        SolverT: The solver that was used to solve the system. If `solver_fn` was None,
            the function constructs it via `default_solver()`. Otherwise, it passes
            `solver_fn` back to the user.
        Tensor[Float, [N]]: Result of the solve $x$.
    """
    result: tuple[SolverT, torch.Tensor] = _LinearFactorAndSolve.apply(
        mat, b, solver_fn
    )  # pyright: ignore[reportAssignmentType, reportReturnType]
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
    a: sp.SparseTensor,
    x: torch.Tensor,
    m: sp.SparseTensor,
    sigma: float | None,
    solver: SolverT | None = None,
    shrink_factor: float = 0.0,
) -> torch.Tensor:
    """Single power iteration, intended for fixed-point backward pass.

    This function is primarily intended as the fixed-point relation to be used when
    differentiating the eigensolver. It does a couple of things differently from
    standard power iteration---users should almost always reach for `power_iterations()`
    instead.

    Args:
        a (SparseTensor[Float, [N, N]]): Matrix to eigendecompose.
        x (Tensor[Float, [k, N]]): Current iterate.
        m (SparseTensor[Float, [N, N]]): Diagonal "mass" matrix.
        sigma (float | None): Sigma used in the shift-invert mode.
        solver (SolverT | None, optional): Solver for (a - sigma * m) to be used
            in shift-invert mode. Defaults to None.
        shrink_factor (float): How much to shrink the outputs to reduce the
            spectral radius.

    Returns:
        Tensor[Float, [k, N]]: Iteration output.
    """
    if sigma is not None:
        evecs = linear_solve(a - sigma * m, sp.matmul(m, x), solver)[1]
    else:
        # Assume M is not singular, because largest eigenvalue of
        # m^{-1} a is infinite anyway:
        evecs = sp.inv_diag(m) @ sp.matmul(a, x)
    # Instead of taking M^{-1/2} Q(M^{1/2} U), we do the following to avoid inverting a
    # singular M, if M is singular:
    evecs = torch.linalg.solve_triangular(
        torch.linalg.qr(m.sqrt() @ evecs).R, evecs, upper=True, left=False
    )
    m_evecs = sp.matmul(m, evecs)

    dot = (x.conj() * m_evecs).sum(dim=0, keepdim=True).real
    signs = 2 * (dot > 0).int() - 1
    evecs = signs * evecs

    # Shrink spectral radius:
    return shrink_factor * x + (1 - shrink_factor) * evecs


def power_iterations(
    a: sp.SparseTensor,
    m: sp.SparseTensor | None = None,
    k: int = 1,
    sigma: float | None = None,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-level power iterations eigensolver; users should prefer eigsh in most cases.

    This function is called from eigsh in the ``unroll'' mode. It offers no other ways
    of differentiating through the eigensolver, unlike eigsh.

    Args:
        a (SparseTensor[Float, [N, N]]): Matrix to eigendecompose.
        m (SparseTensor[Float, [N, N]]): Diagonal "mass" matrix.
        k: int: How many eigenvalue-eigenvector pairs to compute.
        sigma (float | None): Sigma used in the shift-invert mode.
        max_iter (int): Maximum number of iterations.
        tol (float): Absolute convergence tolerance.

    Returns:
        Tensor[Float, [k]]: Eigenvalues.
        Tensor[Float, [k, N]]: Eigenvectors..

    Raises:
        ValueError: if a non-diagonal m is passed.
    """
    # TODO: cholesky on M for non-diagonal M
    if m is not None and not sp.is_diag(m):
        raise ValueError("power_iterations only works with diagonal m.")
    device = a.device
    dtype = a.dtype
    n = a.shape[0]

    evals = torch.randn([k], device=device, dtype=dtype)
    evecs = torch.randn([n, k], device=device, dtype=dtype)
    if m is None:
        m = sp.eye(n, dtype=dtype, device=device)

    # The following serves to make the gradients symmetric in the backward pass:
    a = 0.5 * (a + a.mT).coalesce()

    if sigma is not None:
        system = a - sigma * m
        solver = default_solver(system)

    a = a.to_sparse_csr()
    for it in range(max_iter):
        if sigma is not None:
            solver, z = linear_solve(system, sp.matmul(m, evecs), solver)
        else:
            z = sp.inv_diag(m) @ sp.matmul(a, evecs)
        z = torch.linalg.solve_triangular(
            torch.linalg.qr(m.sqrt() @ z).R, z, upper=True, left=False
        )
        if torch.linalg.vector_norm(evecs - z).all() < tol:
            evecs = z
            LOGGER.info(f"Power iterations converged after {it} power iterations.")
            break
        evecs = z
    # Rayleigh quotient for generalized eigenvalue
    evals = (evecs.conj() * sp.matmul(a, evecs)).sum(-2)

    sort_idx = torch.argsort(-torch.abs(evals - (sigma if sigma is not None else 0)))
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    return evals, evecs


def _make_power_iteration_vjp(
    a: torch.Tensor,
    evecs: torch.Tensor,
    m: torch.Tensor,
    sigma: float | None,
    solver: SolverT | None = None,
    shrink_factor: float = 0.9,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    with torch.enable_grad():
        a_g = a.clone().requires_grad_(True)
        evecs_g = evecs.clone().requires_grad_(True)
        if sigma is not None and solver is None:
            solver = default_solver(a - sigma * m)
        evecs_out = power_iteration(
            a_g, evecs_g, m, sigma, solver=solver, shrink_factor=shrink_factor
        )
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


def _fixed_point_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    z_init: torch.Tensor,
    max_iter: int,
    tol: float | None = None,
) -> torch.Tensor:
    if tol is None:
        tol = 2 * torch.finfo(z_init.dtype).eps
    z_prev, z = z_init, f(z_init)
    i = 0
    while torch.linalg.norm(z_prev - z) > tol and i < max_iter:
        z_prev, z = z, f(z)
        i += 1
        # if i % 10 == 0:
        #     print(torch.linalg.norm(z_prev - z))
    LOGGER.info(f"Fixed point solver converged after {i} iterations.")
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
    """Matrix-free GMRES solver for J^T x = b.

    Note that the function computes a right-hand side product as its primary use case is
    in the adjoint method.

    Warning:
        This function is largely LLM generated and needs to be verified by a human.

    Args:
        f (Callable[[Tensor[Float, [N]]], Tensor[Float, [N]]]): Function which
            computes $J^T x$ (e.g., a vector-Jacobian product), i.e., matrix-free
            application of an `N`-by-`N` matrix.
        b (Tensor[Float, [N]]): Right-hand side of linear system, same shape as $x$.
        init (Tensor[Float, [N]]): Initial guess for $x$.
        max_iter (int): Maximum number of iterations for the solver.
        abs_tol (float): Absolute tolerance on the residual norm.
        rel_tol (float): Tolerance on the residual norm relative to the initial
            residual norm. The solver exits once the residual norm drops below
            `abs_tol + rel_tol * initial_res_norm`, or after `max_iter` iterations.
        preconditioner (Callable[[Tensor[Float, [Ds]]], Tensor[Float, [Ds]]] | None):
            Function representing $M^{-1}$. If provided, the solver solves the
            left-preconditioned system $M^{-1} J^T x = M^{-1} b$.
        verbose (bool): Whether to print logging information.

    Returns:
        Tensor[Float, [N]]: Solution to the linear system, $x$. If the system failed
            to converge, returns the last iterate.
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
        breakdown = bool(hessenberg[k + 1, k] <= 1e-12)
        if not breakdown:
            krylov_basis[:, k + 1] = w / hessenberg[k + 1, k]

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
        if breakdown or res <= abs_tol + rel_tol * initial_res_norm:
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


def _build_diagonal_preconditioner(f, shape, device, dtype):
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


def _build_sampled_diagonal_preconditioner(f, shape, device, dtype, n_samples=500):
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


def cg_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    init: torch.Tensor,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Matrix-free conjugate gradient solver for Jx = b, for a PSD J.

    Warning:
        This function is largely LLM generated and needs to be verified by a human.

    Args:
        f (Callable[[Tensor[Float, [N]]], Tensor[Float, [N]]]): Function which
            computes $J^T x$ (e.g., a vector-Jacobian product), i.e., matrix-free
            application of an `N`-by-`N` matrix.
        b (Tensor[Float, [N]]): Right-hand side of linear system, same shape as $x$.
        init (Tensor[Float, [N]]): Initial guess for $x$.
        max_iter (int): Maximum number of iterations for the solver.
        tol (float): Absolute tolerance on the residual norm.

    Returns:
        Tensor[Float, [N]]: Solution to the linear system, $x$. If the system failed
            to converge, returns the last iterate.
    """
    x = init.clone()
    r = b - f(x)
    r_flat = r.flatten()

    rsold = torch.dot(r_flat, r_flat)
    if torch.sqrt(rsold) < tol:
        return x

    p = r.clone()

    for k in range(max_iter):
        ap = f(p)
        ap_flat = ap.flatten()
        p_flat = p.flatten()

        alpha = rsold / torch.dot(p_flat, ap_flat)
        x = x + alpha * p
        r = r - alpha * ap
        r_flat = r.flatten()

        rsnew = torch.dot(r_flat, r_flat)
        res = torch.sqrt(rsnew)

        if res < tol:
            LOGGER.info(f"CG exiting after {k + 1} iterations, residual: {res:.6e}")
            return x

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    res = torch.sqrt(rsold)
    LOGGER.info(f"CG exiting after {max_iter} iterations, residual: {res:.6e}")
    return x


def estimate_spectral_radius(
    f: Callable[[torch.Tensor], torch.Tensor], init: torch.Tensor, max_iter: int
) -> torch.Tensor:
    """Estimate spectral radius (max |evals|) of matrix J^T (given as callable f).

    Uses the power method to compute the spectral radius of J^T. Can be useful for
    debugging the inverse of vector-Jacobian products.

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): Function that computes v @ J.
        init (torch.Tensor): Initial guess for largest eigenvector.
        max_iter (int): Maximum number of iterations for the solver.

    Returns:
        torch.Tensor: Spectral radius, i.e., the maximum absolute eigenvalue.
    """
    v = init / (init.norm() + 1e-12)

    for _ in range(max_iter):
        w = f(v)
        v = w / (w.norm() + 1e-12)

    w = f(v)
    rho_est = (v * w).sum().abs()
    return rho_est


class _Eigsh(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        (a, m, _, sigma, max_iter, tol, bwd_method, bwd_max_iter, bwd_eps) = inputs
        evals, evecs = output
        ctx.save_for_backward(a, m, evals, evecs)
        ctx.bwd_method = bwd_method
        ctx.sigma = sigma
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.bwd_max_iter = bwd_max_iter
        ctx.bwd_eps = bwd_eps

    @staticmethod
    def forward(
        a: sp.SparseTensor,
        m: sp.SparseTensor,
        k: int = 1,
        sigma: float = 0.0,
        max_iter: int = 10,
        tol: float = 1e-10,
        bwd_method: Literal[
            "individual", "truncate", "fixed-point", "step-backward"
        ] = "individual",
        bwd_max_iter: int = 200,
        bwd_eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: make M optional
        assert isinstance(a, torch.Tensor)
        assert isinstance(m, torch.Tensor)

        device = a.device
        dtype = a.dtype
        if a.is_cpu:
            a_sp = sp.to_scipy(a)
            m_sp = sp.to_scipy(m)
            if k < a.shape[0] - 1:
                evals, evecs = scipy.sparse.linalg.eigsh(
                    a_sp, k=k, M=m_sp, sigma=sigma, maxiter=max_iter, tol=tol
                )
            else:
                a_sp = a_sp.toarray()
                m_sp = m_sp.toarray()
                evals, evecs = scipy.linalg.eigh(a_sp, b=m_sp)
            sort_idx = np.argsort(-np.abs(evals))
            evals = evals[sort_idx][:k]
            evecs = evecs[:, sort_idx][:, :k]
            return torch.tensor(evals.real, dtype=dtype, device=device), torch.tensor(
                evecs.real, dtype=dtype, device=device
            )
        else:
            return power_iterations(a, m, k=k, sigma=sigma, max_iter=max_iter, tol=tol)

    @staticmethod
    def backward(
        ctx, grad_evals: torch.Tensor | None, grad_evecs: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        a, m, evals, evecs = ctx.saved_tensors
        n: int = a.shape[0]
        k: int = evecs.shape[1]
        device = a.device
        dtype = a.dtype
        if m is None:
            m = sp.eye(a.shape[0], device=a.device, dtype=a.dtype)

        grad_a = None
        if ctx.needs_input_grad[0]:
            if grad_evals is None:
                grad_evals: torch.Tensor = torch.zeros(k, dtype=dtype, device=device)
            if grad_evecs is None:
                grad_evecs: torch.Tensor = torch.zeros(n, k, dtype=dtype, device=device)

            if ctx.bwd_method == "truncate":
                diff = evals[None, :] - evals[:, None]  # [k, k]
                one_over_diff = torch.zeros_like(diff)
                eps = torch.finfo(diff.dtype).eps
                mask = ~torch.eye(k, dtype=torch.bool, device=diff.device)
                one_over_diff[mask] = 1.0 / (
                    diff[mask] + (diff[mask] == 0).to(diff.dtype) * eps
                )

                tilde_g = evecs.mT @ grad_evecs  # [k, k]
                s = 0.5 * (tilde_g - tilde_g.mT) * one_over_diff
                h = s + torch.diag(grad_evals)

                # Sparse outer product evecs @ h @ evecs.mT
                rows, cols = a.indices()
                grad_vals = (evecs @ h)[rows, ...] * evecs[cols, ...]
                if grad_vals.ndim == 2:
                    grad_vals = grad_vals.sum(-1)
                grad_a = sp.coo_tensor(a.indices(), grad_vals, [n, n])
            elif ctx.bwd_method == "individual":
                dg_dx = torch.zeros_like(evecs)
                lhs_i = torch.arange(n, device=device, dtype=torch.int64)
                lhs_j = torch.full([n], n, device=device, dtype=torch.int64)
                for i in range(evals.shape[0]):
                    lhs_aa = (a - evals[i] * m).coalesce()
                    lhs = sp.coo_tensor(
                        lhs_aa.indices(), lhs_aa.values(), size=[n + 1, n + 1]
                    ).coalesce()
                    lhs_ba = -sp.matmul(m, evecs[:, i])
                    lhs = sp.append(lhs, torch.stack([lhs_i, lhs_j]), lhs_ba)
                    lhs = sp.append(lhs, torch.stack([lhs_j, lhs_i]), lhs_ba)
                    if grad_evals is None:
                        rhs = torch.nn.functional.pad(grad_evecs[:, i], (0, 1))
                    else:
                        rhs = torch.cat((grad_evecs[:, i], grad_evals[i : i + 1]), -1)
                    lhs_scipy = sp.to_scipy(lhs).tocsr()
                    rhs_numpy = rhs.cpu().detach().numpy()
                    result = scipy.sparse.linalg.spsolve(lhs_scipy, rhs_numpy)[:n]
                    dg_dx[:, i] = torch.tensor(result, device=device)

                # Compute masked outer product:
                # (dg/dx) @ x^T
                rows, cols = a.indices()
                grad_vals = -dg_dx[rows, ...] * evecs[cols, ...]
                if grad_vals.ndim == 2:
                    grad_vals = grad_vals.sum(-1)
                grad_a = sp.coo_tensor(a.indices(), grad_vals, [n, n])
            elif ctx.bwd_method == "step-backward":

                def implicit_func(a, x):
                    return power_iteration(a, x, m, ctx.sigma) - x

                # Test full formula:
                df_da = torch.func.jacrev(implicit_func, 0)
                df_dx = torch.func.jacrev(implicit_func, 1)

                df_da_mat = df_da(a, evecs).reshape(n * k, n * n)
                df_dx_mat = df_dx(a, evecs).reshape(n * k, n * k)
                dx_da_mat = -torch.linalg.solve(df_dx_mat, df_da_mat)
                grad_a = (grad_evecs.flatten() @ dx_da_mat).reshape(n, n)

            elif ctx.bwd_method == "fixed-point":
                # TODO: eigenvalue loss

                init = torch.zeros_like(evecs)

                # rho = estimate_spectral_radius(lambda z: vjp(z)[1], init, 10)
                # print("RHO", rho)

                # print(init, power_iteration(a, init), vjp(init)[1])
                # print(vjp(init)[1])
                # u = fixed_point_solver(
                #     lambda z: grad_evecs + vjp(z)[1], init, ctx.max_iter, ctx.tol
                # )

                # start = time.perf_counter()
                vjp_evecs, vjp_a = _make_power_iteration_vjp(a, evecs, m, ctx.sigma)
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
            else:
                raise ValueError(f"Unrecognized eigsh bwd_method={ctx.bwd_method}.")

        if grad_a is not None:
            # Ensure symmetry of gradients:
            grad_a = 0.5 * (grad_a + grad_a.mT)
        # TODO: differentiate w.r.t. M (fine to assume diagonal M for now)
        return grad_a, None, None, None, None, None, None, None, None


def eigsh(
    a: sp.SparseTensor,
    m: sp.SparseTensor | None = None,
    k: int = 1,
    sigma: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    bwd_method: Literal[
        "unroll", "individual", "truncate", "fixed-point"
    ] = "individual",
    bwd_max_iter: int = 100,
    bwd_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse hermitian matrix eigensolver.

    For the forward solve, the function delegates to scipy if the tensor is on the CPU,
    otherwise, it relies on `power_iterations()`. Independently, the user can specify
    multiple different approaches to computing gradients of this function.

    Args:
        a (SparseTensor[Float, [N, N]]): Matrix to eigendecompose.
        m (SparseTensor[Float, [N, N]]): Diagonal "mass" matrix.
        k: int: How many eigenvalue-eigenvector pairs to compute.
        sigma (float | None): Sigma used in the shift-invert mode.
        max_iter (int): Maximum number of iterations.
        tol (float): Absolute convergence tolerance.
        bwd_method (Literal["unroll", "individual", "truncate", "fixed-point"]): Which
            method to use for computing adjoints. Selecting `unroll` changes the
            forward solver too. See `the original paper <https://anadodik.github.io/publication/iskra/iskra.pdf>`_
            for an explanation of the different modes.
        bwd_max_iter (int): Maximum number of iterations for the backward GMRES solver
            if bwd_method="fixed-point".
        bwd_eps (float): Absolute convergence tolerance for the backward GMRES solver
            if bwd_method="fixed-point".

    Returns:
        Tensor[Float, [k]]: Eigenvalues.
        Tensor[Float, [k, N]]: Eigenvectors..

    Raises:
        ValueError: if a non-diagonal m is passed.
    """
    if not sp.is_diag(m):
        raise ValueError("Only works with diagonal m.")
    if bwd_method == "unroll":
        return power_iterations(a, m, k=k, sigma=sigma, max_iter=max_iter, tol=tol)
    else:
        return _Eigsh.apply(
            a, m, k, sigma, max_iter, tol, bwd_method, bwd_max_iter, bwd_eps
        )

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from typing import Any, Callable, Literal, Sequence, cast

import torch

from iskra.logging import getLogger
from iskra.profiling import profile_block, profile_fn
from iskra.sparse_linalg import (
    build_diagonal_preconditioner,
    build_sampled_diagonal_preconditioner,
    cg_solve,
    estimate_spectral_radius,
    gmres_solve,
)

LOGGER = getLogger(__name__)


def make_adjoint_vjps[T, **P](
    fn: Callable[P, T],
    inputs: Sequence[int] | int = 0,
    outputs: Sequence[int] | int = 0,
    params: tuple[int, ...] = (1,),
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
]:
    """Constructs the VJP functions needed for adjoint computation.

    Args:
        fn (Callable[P, T]): The adjoint relation f(x, y) = 0.
        iterates:

    Returns:
        _type_: _description_
    """
    if not isinstance(inputs, Sequence):
        inputs = [inputs]
    if not isinstance(outputs, Sequence):
        outputs = [outputs]

    args_list: list[Any] = list(args)
    with torch.enable_grad():
        for input_idx in inputs:
            arg = args_list[input_idx]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Requested gradient w.r.t. iterate at position {input_idx}, "
                    f"but passed value is not a Tensor. Got {type(arg)} instead."
                )
            args_list[input_idx] = arg.clone().requires_grad_(True)
        param_args = []
        for arg_i in params:
            arg = args_list[arg_i]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Requested gradient w.r.t. parameter at position {arg_i}, "
                    f"but passed value is not a Tensor. Got {type(arg)} instead."
                )
            arg = arg.clone().requires_grad_(True)
            args_list[arg_i] = arg
            param_args.append(arg)

        fn_outputs = fn(*args_list, **kwargs)  # type: ignore

        if not isinstance(fn_outputs, Sequence):
            fn_outputs = (fn_outputs,)

        befores: tuple[torch.Tensor, ...] = tuple(args_list[idx] for idx in inputs)
        afters: tuple[torch.Tensor, ...] = tuple(fn_outputs[idx] for idx in outputs)  # type: ignore
        afters_tensor = torch.cat([after.flatten() for after in afters])

        def vjp_inputs(z_grad: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(
                (afters_tensor,),
                befores,
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            return torch.cat([grad.flatten() for grad in grads])

        def vjp_params(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (afters_tensor,),
                param_args,
                (z_grad.flatten(),),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

    return vjp_inputs, vjp_params


def make_fixed_point_layer[T, **P](
    fn: Callable[P, T],
    iterates: Sequence[tuple[int, int]] | tuple[int, int] = (0, 0),
    argnums: int | tuple[int, ...] = 1,
    fwd_method: Literal["fixed-point"] | Callable[P, T] = "fixed-point",
    fwd_max_iter: int | None = 100,
    fwd_error_metric: Literal["delta"] | int | None = "delta",
    fwd_error_ord: int | float | Literal["fro", "nuc"] = 2,
    fwd_abs_tol: float = 1e-6,
    fwd_rel_tol: float = 1e-3,
    bwd_method: Literal["gmres"] = "gmres",
    gmres_init: torch.Tensor | None = None,
    callback_gmres_sol: Callable | None = None,
    bwd_max_iter: int = 500,
    bwd_abs_tol: float = 1e-6,
    bwd_rel_tol: float = 1e-3,
    verbose: bool = False,
) -> Callable[P, T]:
    if verbose:
        LOGGER.setLevel("INFO")
    if not isinstance(iterates[0], Sequence):
        iterates = [cast(tuple[int, int], iterates)]
    iterates = cast(Sequence[tuple[int, int]], iterates)

    if not isinstance(argnums, Sequence):
        argnums = (argnums,)
    if fwd_max_iter is not None and fwd_max_iter <= 1:
        raise ValueError(f"fwd_max_iter must be >= 1, is {fwd_max_iter}.")

    # TODO: look into accelerations: https://docs.sciml.ai/NonlinearSolve/stable/solvers/fixed_point_solvers/
    class SolverFn(torch.autograd.Function):
        @staticmethod
        def setup_context(ctx, inputs, outputs):
            for arg_i, arg in enumerate(inputs):
                if not isinstance(arg, torch.Tensor):
                    raise ValueError(
                        f"Iterate fn parameter at position {arg_i} is not a Tensor. "
                        f"Got {type(arg)} instead."
                    )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for out_i, out in enumerate(outputs):
                if not isinstance(out, torch.Tensor):
                    raise ValueError(
                        f"Iterate fn output at position {out_i} is not a Tensor. "
                        f"Got {type(out)} instead."
                    )
            ctx.n_inputs = len(inputs)
            ctx.n_outputs = len(outputs)
            ctx.save_for_backward(*inputs, *outputs)

        @staticmethod
        @profile_fn(name="SolverFn.forward")
        def forward(*args: P.args, **kwargs: P.kwargs) -> T:
            rel_ref = float("inf")
            args_list: list[Any] = list(args)
            outputs: T | None = None
            step_i = 0
            while True:
                step_i += 1
                if fwd_max_iter is not None and step_i >= fwd_max_iter:
                    break
                with profile_block("fwd_iter"):
                    outputs = fn(*args_list, **kwargs)  # type: ignore
                    outputs_tuple = outputs
                    if not isinstance(outputs_tuple, tuple):
                        outputs_tuple = (outputs_tuple,)
                    outputs_tuple = cast(tuple[Any, ...], outputs_tuple)
                before = torch.cat(
                    [args_list[iterate[0]].flatten() for iterate in iterates]
                )
                after = torch.cat(
                    [outputs_tuple[iterate[1]].flatten() for iterate in iterates]
                )

                for iterate in iterates:
                    args_list[iterate[0]] = outputs_tuple[iterate[1]]
                if fwd_error_metric is None:
                    continue

                if fwd_error_metric == "delta":
                    error = before - after
                    rel_ref = torch.linalg.norm(after.flatten(), ord=fwd_error_ord)
                elif isinstance(fwd_error_metric, int):
                    error = outputs_tuple[fwd_error_metric]
                    if rel_ref == float("inf"):
                        # If user provides residual, we compare tolerances relative to
                        # the initial residual, as the residual and output
                        # don't have to have comparable units.
                        rel_ref = torch.linalg.norm(error.flatten(), ord=fwd_error_ord)
                else:
                    raise ValueError(f"Invalid fwd_error_metric={fwd_error_metric}.")

                error_norm = torch.linalg.norm(error.flatten(), ord=fwd_error_ord)
                if error_norm <= fwd_abs_tol + fwd_rel_tol * rel_ref:
                    break

                if verbose and fwd_error_metric is not None:
                    if fwd_error_metric == "delta":
                        error_str = "delta"
                    else:
                        error_str = "residual"
                    abs_val = error_norm.cpu().detach().item()
                    rel_val = (
                        (error_norm / rel_ref).cpu().detach().item()
                        if rel_ref > 0
                        else 0.0
                    )
                    LOGGER.debug(
                        f"Forward {error_str}: abs={abs_val:.3e}, rel={rel_val:.3e}"
                    )
            if verbose:
                if fwd_error_metric == "delta":
                    error_str = "delta"
                else:
                    error_str = "residual"
                abs_val = error_norm.cpu().detach().item()
                rel_val = (
                    (error_norm / rel_ref).cpu().detach().item() if rel_ref > 0 else 0.0
                )
                LOGGER.info(
                    f"Forward {error_str}: abs={abs_val:.3e}, rel={rel_val:.3e}"
                )
            assert outputs is not None
            return outputs

        @staticmethod
        @profile_fn(name="SolverFn.backward")
        def backward(
            ctx, *grads_out: torch.Tensor | None
        ) -> tuple[torch.Tensor | None, ...]:
            result = ctx.n_inputs * [None]

            in_out = list(ctx.saved_tensors)
            inputs = in_out[: ctx.n_inputs]
            outputs = in_out[-ctx.n_outputs :]
            grad_iterate = torch.cat(
                [grads_out[iterate[1]].flatten() for iterate in iterates]
            )
            if grad_iterate is None or len(argnums) == 0:
                return result
            for iterate in iterates:
                inputs[iterate[0]] = outputs[iterate[1]]
            vjp_iterate, vjp_params = make_adjoint_vjps(
                fn,
                [iterate[0] for iterate in iterates],
                [iterate[1] for iterate in iterates],
                argnums,
                *inputs,
            )
            init = gmres_init
            if init is None:
                init = grad_iterate + vjp_iterate(grad_iterate)

            with profile_block("bwd_optim"):
                system_fn = lambda z: vjp_iterate(z) - z  # noqa: E731
                if verbose:
                    spec_init = torch.randn_like(grad_iterate)
                    spectral = estimate_spectral_radius(
                        lambda z: vjp_iterate(z), spec_init, 10_000
                    )
                    LOGGER.info(f"Spectral radius of df/dy: {spectral}")
                preconditioner = None
                # preconditioner = build_sampled_diagonal_preconditioner(
                #     system_fn, init.shape, init.device, init.dtype
                # )
                # def preconditioner(v):
                #     return (v.flatten() * torch.ones_like(init)).reshape(init.shape)

                if bwd_method == "gmres":
                    dl_df = -gmres_solve(
                        system_fn,
                        grad_iterate,
                        init,
                        max_iter=bwd_max_iter,
                        abs_tol=bwd_abs_tol,
                        rel_tol=bwd_rel_tol,
                        preconditioner=preconditioner,
                        verbose=verbose,
                    )
                    if callback_gmres_sol is not None:
                        callback_gmres_sol(-dl_df)
                else:
                    raise ValueError(f"Unrecognized backwards solver {bwd_method}")

            if verbose:
                gmres_error = torch.norm(
                    (grad_iterate.flatten() - system_fn(-dl_df)).flatten()
                )
                LOGGER.info(f"Iterate difference (backward): {gmres_error.item()}")

            with profile_block("bwd_inputs"):
                grad_param = vjp_params(dl_df)
                for i, argnum in enumerate(argnums):
                    result[argnum] = grad_param[i]
            return (*result,)

    return SolverFn.apply


@torch.no_grad
def compute_numerical_jacobian[T, **P](
    solver_fn: Callable[P, T],
    out_idx: int = 0,
    argnum: int = 1,
    eps: float = 1e-8,
    *args: P.args,
    **kwargs: P.kwargs,
) -> torch.Tensor:
    args_list = list(args)
    arg = args[argnum]
    if not isinstance(arg, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {argnum}, "
            f"but passed value is not a Tensor. Got {type(arg)} instead."
        )
    dtype = arg.dtype
    device = arg.device
    arg = arg.clone()
    offset = torch.zeros_like(arg.flatten())
    num_jac = None
    for i in range(offset.shape[0]):
        offset[i] = eps
        args_list[argnum] = arg + offset.reshape(*arg.shape)
        plus = solver_fn(*args_list, **kwargs)
        if isinstance(plus, tuple):
            plus = plus[out_idx]
        args_list[argnum] = arg - offset.reshape(*arg.shape)
        minus = solver_fn(*args_list, **kwargs)
        if isinstance(minus, tuple):
            minus = minus[out_idx]
        offset[i] = 0
        if num_jac is None:
            num_jac = torch.zeros(
                [plus.nelement(), arg.nelement()], dtype=dtype, device=device
            )
        num_jac[:, i] = (plus - minus).flatten() / (2 * eps)
    if num_jac is None:
        num_jac = torch.tensor([], dtype=dtype, device=device)
    return num_jac


@torch.no_grad
def compute_jacobian[T, **P](
    fn: Callable[P, T],
    in_idx: int = 0,
    out_idx: int = 0,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    args_list: list[Any] = list(args)
    arg: torch.Tensor = args[in_idx]
    if not isinstance(arg, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {argnum}, "
            f"but passed value is not a Tensor. Got {type(arg)} instead."
        )
    with torch.enable_grad():
        arg = arg.clone().requires_grad_(True)
        args_list[in_idx] = arg
        outputs = fn(*args_list, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        output = outputs[out_idx].flatten()

        def vjp(z_grad: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(
                (output,),
                (arg,),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            return torch.cat([grad.flatten() for grad in grads])

    device = arg.device
    dtype = arg.dtype
    basis = torch.eye(arg.nelement(), device=device, dtype=dtype)
    jac_rows = []
    for i in range(arg.nelement()):
        jac_rows.append(vjp(basis[i]).flatten())
    jac = torch.stack(jac_rows, 0)
    return jac


@torch.no_grad
def compute_jacobians[T, **P](
    fn: Callable[P, T],
    iterate_in: int = 0,
    iterate_out: int = 0,
    argnum: int = 1,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    arg: torch.Tensor = args[argnum]
    if not isinstance(arg, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {argnum}, "
            f"but passed value is not a Tensor. Got {type(arg)} instead."
        )
    iterate: torch.Tensor = args[iterate_in]
    if not isinstance(iterate, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {iterate_in}, "
            f"but passed value is not a Tensor. Got {type(iterate)} instead."
        )
    device = arg.device
    dtype = arg.dtype
    vjp_iterate, vjp_params = make_adjoint_vjps(
        fn, iterate_in, iterate_out, (argnum,), *args, **kwargs
    )
    basis = torch.eye(iterate.nelement(), device=device, dtype=dtype)
    jac_rows = []
    for i in range(iterate.nelement()):
        jac_rows.append(vjp_iterate(basis[i].flatten()).flatten())
    jac_iterate = torch.stack(jac_rows, 0)

    jac_bc_rows = []
    for i in range(iterate.nelement()):
        jac_bc_rows.append(vjp_params(basis[i].flatten())[0].flatten())
    jac_params = torch.stack(jac_bc_rows, 0)
    return jac_iterate, jac_params

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import time
from typing import Any, Callable, Literal, Sequence, cast

import torch

from iskra.profiling import profile_block, profile_fn
from iskra.sparse_linalg import (
    build_diagonal_preconditioner,
    build_sampled_diagonal_preconditioner,
    cg_solve,
    estimate_spectral_radius,
    gmres_solve,
)


def make_vjp[T, **P](
    fn: Callable[P, T],
    iterates: Sequence[tuple[int, int]] | tuple[int, int] = (0, 0),
    argnums: tuple[int, ...] = (1,),
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    if not isinstance(iterates[0], Sequence):
        iterates = [cast(tuple[int, int], iterates)]
    iterates = cast(Sequence[tuple[int, int]], iterates)

    args_list: list[Any] = list(args)
    with torch.enable_grad():
        for iterate_in, _ in iterates:
            arg = args_list[iterate_in]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Requested gradient w.r.t. iterate at position {iterate_in}, "
                    f"but passed value is not a Tensor. Got {type(arg)} instead."
                )
            args_list[iterate_in] = arg.clone().requires_grad_(True)
        params = []
        for arg_i in argnums:
            arg = args_list[arg_i]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Requested gradient w.r.t. parameter at position {arg_i}, "
                    f"but passed value is not a Tensor. Got {type(arg)} instead."
                )
            arg = arg.clone().requires_grad_(True)
            args_list[arg_i] = arg
            params.append(arg)

        outputs = fn(*args_list, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        befores = tuple(args_list[iterate[0]] for iterate in iterates)
        afters = tuple(outputs[iterate[1]] for iterate in iterates)
        iterate_diff = torch.cat(
            [(after - before).flatten() for before, after in zip(befores, afters)]
        )

        def vjp_iterate(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (iterate_diff,),
                befores,
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

        def vjp_params(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (iterate_diff,),
                params,
                (z_grad.flatten(),),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

    return vjp_iterate, vjp_params


def make_solver_layer[T, **P](
    fn: Callable[P, T],
    iterates: Sequence[tuple[int, int]] | tuple[int, int] = (0, 0),
    argnums: int | tuple[int, ...] = 1,
    fwd_method: Literal["fixed-point"] | Callable[P, T] = "fixed-point",
    fwd_max_iter: int = 90,
    fwd_eps: float = 1e-10,
    fwd_error_arg: int | None = None,
    fwd_error_tol: float | None = None,
    bwd_method: Literal["gmres"] = "gmres",
    gmres_init: torch.Tensor | None = None,
    callback_gmres_sol: Callable | None = None,
    bwd_max_iter: int = 200,
    bwd_eps: float = 1e-12,
    verbose: bool = False,
) -> Callable[P, T]:
    if not isinstance(iterates[0], Sequence):
        iterates = [cast(tuple[int, int], iterates)]
    iterates = cast(Sequence[tuple[int, int]], iterates)

    if not isinstance(argnums, Sequence):
        argnums = (argnums,)

    if fwd_error_tol is not None and fwd_error_arg is None:
        raise ValueError(
            "Must specify which function output is the error, please set fwd_error_arg."
        )

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
            args_list: list[Any] = list(args)
            for _ in range(fwd_max_iter):
                with profile_block("fwd_iter"):
                    outputs = fn(*args_list, **kwargs)
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
                with torch.no_grad():
                    # TODO: figure out criterion
                    difference = torch.norm((before - after).flatten())
                    for iterate in iterates:
                        args_list[iterate[0]] = outputs_tuple[iterate[1]]
                    if difference < fwd_eps:
                        break
                    if fwd_error_arg is not None:
                        if (outputs_tuple[fwd_error_arg] < fwd_error_tol).all():
                            break
                    if verbose:
                        print(
                            "Iterate difference (forward): ",
                            difference.cpu().detach().item(),
                        )
            if verbose:
                print(
                    "Iterate difference (forward): ", difference.cpu().detach().item()
                )
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
            vjp_iterate, vjp_params = make_vjp(fn, iterates, argnums, *inputs)
            init = gmres_init
            if init is None:
                init = torch.zeros_like(grad_iterate)

            with profile_block("bwd_optim"):
                system_fn = lambda z: torch.cat(  # noqa: E731
                    [grad.flatten() for grad in vjp_iterate(z)]
                )
                # spectral = estimate_spectral_radius(system_fn, init, 1000)
                # print(f"Spectral radius: {spectral}")
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
                        maxiter=bwd_max_iter,
                        tol=bwd_eps,
                        preconditioner=preconditioner,
                    )
                    if callback_gmres_sol is not None:
                        callback_gmres_sol(-dl_df)
                    # dl_df = -cg_solve(
                    #     lambda z: torch.cat(
                    #         [grad.flatten() for grad in vjp_iterate(z)]
                    #     ),
                    #     grad_iterate,
                    #     init,
                    #     maxiter=bwd_max_iter,
                    #     tol=bwd_eps,
                    # )
                else:
                    raise ValueError(f"Unrecognized backwards solver {bwd_method}")

            # TODO: add verbose to gmres:
            gmres_error = torch.norm(
                (
                    grad_iterate.flatten()
                    - torch.cat([grad.flatten() for grad in vjp_iterate(-dl_df)])
                ).flatten()
            )
            if verbose:
                print(f"Iterate difference (backward): {gmres_error.item()}")

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
    print(arg)
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
def compute_jacobians[T, **P](
    fn: Callable[P, T],
    iterate_in: int = 0,
    iterate_out: int = 0,
    argnum: int = 1,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    arg = args[argnum]
    if not isinstance(arg, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {argnum}, "
            f"but passed value is not a Tensor. Got {type(arg)} instead."
        )
    iterate = args[iterate_in]
    if not isinstance(iterate, torch.Tensor):
        raise ValueError(
            f"Requested numerical gradient w.r.t. parameter at position {iterate_in}, "
            f"but passed value is not a Tensor. Got {type(iterate)} instead."
        )
    device = arg.device
    dtype = arg.dtype
    vjp_iterate, vjp_params = make_vjp(
        fn, iterate_in, iterate_out, (argnum,), *args, **kwargs
    )
    basis = torch.eye(iterate.nelement(), device=device, dtype=dtype)
    jac_rows = []
    for i in range(iterate.nelement()):
        jac_rows.append(vjp_iterate(basis[i].reshape(*iterate.shape))[0].flatten())
    jac_iterate = torch.stack(jac_rows, 0)

    jac_bc_rows = []
    for i in range(iterate.nelement()):
        jac_bc_rows.append(vjp_params(basis[i].reshape(*iterate.shape))[0].flatten())
    jac_params = torch.stack(jac_bc_rows, 0)
    return jac_iterate, jac_params

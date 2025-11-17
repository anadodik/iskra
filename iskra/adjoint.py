# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from typing import Any, Callable, Iterable, Literal, cast

import torch

from iskra.sparse_linalg import gmres_solve


def make_vjp[T, **P](
    fn: Callable[P, T],
    iterate_in: int = 0,
    iterate_out: int = 0,
    argnums: tuple[int, ...] = (1,),
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[Callable[[torch.Tensor], tuple[torch.Tensor, ...]], ...]:
    args_list: list[Any] = list(args)
    with torch.enable_grad():
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
        # deformed_out = deformed_out - deformed_g
        before = args_list[iterate_in]
        after = cast(tuple[Any, ...], outputs)[iterate_out]
        iterate_diff = after - before

        def vjp_iterate(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (iterate_diff,),
                (before,),
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

        def vjp_params(z_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                (iterate_diff,),
                params,
                (z_grad,),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

    return vjp_iterate, vjp_params


def make_solver_layer[T, **P](
    fn: Callable[P, T],
    iterate_in: int = 0,
    iterate_out: int = 0,
    argnums: int | tuple[int, ...] = 1,
    fwd_method: Literal["fixed-point"] | Callable[P, T] = "fixed-point",
    fwd_max_iter: int = 90,
    fwd_eps: float = 1e-10,
    bwd_method: Literal["gmres"] = "gmres",
    bwd_max_iter: int = 200,
    bwd_eps: float = 1e-12,
) -> Callable[P, T]:
    if not isinstance(argnums, Iterable):
        argnums = (argnums,)

    class SolverFn(torch.autograd.Function):
        @staticmethod
        def setup_context(ctx, inputs, outputs):
            for arg_i, arg in enumerate(inputs):
                if not isinstance(arg, torch.Tensor):
                    raise ValueError(
                        f"Iterate fn parameter at position {arg_i} is not a Tensor. "
                        f"Got {type(arg)} instead."
                    )
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
        def forward(*args: P.args, **kwargs: P.kwargs) -> T:
            args_list: list[Any] = list(args)
            for _ in range(fwd_max_iter):
                outputs = fn(*args_list, **kwargs)
                before = args_list[iterate_in]
                after = cast(tuple[Any, ...], outputs)[iterate_out]
                with torch.no_grad():
                    difference = torch.linalg.vector_norm(before - after, axis=-1).max()
                    args_list[iterate_in] = after
                    if difference < fwd_eps:
                        break
            print("Final error: ", difference.cpu().detach().item())
            return outputs

        @staticmethod
        def backward(
            ctx, *grads_out: torch.Tensor | None
        ) -> tuple[torch.Tensor | None, ...]:
            result = ctx.n_inputs * [None]

            in_out = list(ctx.saved_tensors)
            inputs = in_out[: ctx.n_inputs]
            outputs = in_out[-ctx.n_outputs :]
            grad_iterate = grads_out[iterate_out]
            if grad_iterate is None or len(argnums) == 0:
                return result
            inputs[iterate_in] = outputs[iterate_out]
            vjp_iterate, vjp_params = make_vjp(
                fn, iterate_in, iterate_out, argnums, *inputs
            )
            init = torch.randn_like(grad_iterate)
            if bwd_method == "gmres":
                dl_df = -gmres_solve(
                    lambda z: vjp_iterate(z)[0],
                    grad_iterate,
                    init,
                    maxiter=bwd_max_iter,
                    tol=bwd_eps,
                )
            else:
                raise ValueError(f"Unrecognized backwards solver {bwd_method}")
            gmres_error = torch.norm((grad_iterate - vjp_iterate(-dl_df)[0]).flatten())
            print(f"GMRES Error: {gmres_error.item()}")
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
        plus = solver_fn(*args_list, **kwargs)[out_idx]
        args_list[argnum] = arg - offset.reshape(*arg.shape)
        minus = solver_fn(*args_list, **kwargs)[out_idx]
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

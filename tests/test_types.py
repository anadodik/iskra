import pytest
import torch

import iskra.types as it
from iskra.types import (
    C64,
    D1,
    D2,
    D3,
    F32,
    I16,
    I32,
    U8,
    Batches,
    Bool,
    Edges,
    Tensor,
    Tets,
    Tris,
    Verts,
    _type_repr,
    typed,
)

test_func_template = """
import iskra

@typed
def _func(a: {type}):
    pass
"""

correct = [
    (
        test_func_template.format(type=Tensor[F32, [D1]]),
        torch.empty([1], dtype=torch.float32),
    ),
    (
        test_func_template.format(type=Tensor[I32, []]),
        torch.empty([], dtype=torch.int32),
    ),
    (
        test_func_template.format(type=Tensor[U8, [Batches, D1, D2, D3]]),
        torch.empty([1, 2, 3], dtype=torch.uint8),
    ),
    (
        test_func_template.format(type=Tensor[F32, [Batches, D1, D2, D3]]),
        torch.empty([100, 5, 1, 2, 3], dtype=torch.float32),
    ),
    (
        test_func_template.format(type=Tensor[C64, [Verts, Batches, D2 | D3]]),
        torch.empty([100, 5, 3], dtype=torch.cfloat),
    ),
    (
        test_func_template.format(type=Tensor[Bool, [Batches, D2]]),
        torch.empty([100, 2], dtype=torch.bool),
    ),
    (
        test_func_template.format(type=Tensor[I16, [Verts, Edges, D2]]),
        torch.empty([100, 20, 2], dtype=torch.int16),
    ),
]


incorrect_type = [
    (
        test_func_template.format(type=Tensor[F32, [D1]]),
        torch.empty([1], dtype=torch.int32),
    ),
    (
        test_func_template.format(type=Tensor[I32, []]),
        torch.empty([], dtype=torch.float32),
    ),
    (
        test_func_template.format(type=Tensor[U8, [Batches, D1, D2, D3]]),
        torch.empty([1, 2, 3], dtype=torch.cfloat),
    ),
    (
        test_func_template.format(type=Tensor[F32, [Batches, D1, D2, D3]]),
        torch.empty([100, 5, 1, 2, 3], dtype=torch.int16),
    ),
    (
        test_func_template.format(type=Tensor[C64, [Verts, Batches, D2 | D3]]),
        torch.empty([100, 5, 3], dtype=torch.cdouble),
    ),
    (
        test_func_template.format(type=Tensor[Bool, [Batches, D2]]),
        torch.empty([100, 2], dtype=torch.int32),
    ),
    (
        test_func_template.format(type=Tensor[I16, [Verts, Edges, D2]]),
        torch.empty([100, 20, 2], dtype=torch.bool),
    ),
]


@pytest.mark.parametrize("func_text,arg", correct)
def test_correct(func_text: str, arg: torch.Tensor) -> None:
    locals_dict = locals()
    exec(func_text, globals(), locals_dict)
    _func = locals_dict["_func"]
    _func(arg)


@pytest.mark.parametrize("func_text,arg", incorrect_type)
def test_incorrect_type(func_text: str, arg: torch.Tensor) -> None:
    locals_dict = locals()
    exec(func_text, globals(), locals_dict)
    _func = locals_dict["_func"]
    with pytest.raises(TypeError):
        _func(arg)


def test_top_level_union() -> None:
    @typed
    def _func(arg: Tensor[I32, [D1, D2]] | int):
        pass

    arg = torch.zeros([1, 2], dtype=torch.int32)
    _func(arg)

    arg = 5
    _func(arg)

    arg = torch.zeros([1, 2], dtype=torch.float32)
    with pytest.raises(TypeError):
        _func(arg)


def test_union_with_tensor() -> None:
    @typed
    def _func(arg: Tensor[I32, [D1, D2]] | Tensor[F32, [D1, D2]]):
        pass

    arg = torch.zeros([1, 2], dtype=torch.int32)
    _func(arg)

    arg = torch.zeros([1, 2], dtype=torch.float32)
    _func(arg)

    @typed
    def _func_2(arg: Tensor[I32 | F32, [D1, D2]]):
        pass

    arg = torch.zeros([1, 2], dtype=torch.int32)
    _func_2(arg)

    arg = torch.zeros([1, 2], dtype=torch.float32)
    _func_2(arg)


def test_list() -> None:
    @typed
    def _func_union(arg: list[Tensor[I32, [D1, D2]] | int]):
        pass

    @typed
    def _func_list(arg: list[Tensor[I32, [D1, D2]]]):
        pass

    arg_t = torch.zeros([1, 2], dtype=torch.int32)
    arg_f = torch.zeros([1, 2], dtype=torch.float32)
    arg_i = 5

    _func_list([arg_t])
    _func_union([arg_t])

    # The following is always fine because we never type check
    # non-tensor objects:
    _func_list([arg_i])  # pyright: ignore

    # This is fine too:
    _func_union([arg_i])  # pyright: ignore

    with pytest.raises(TypeError):
        _func_union([arg_f])
    with pytest.raises(TypeError):
        _func_union([arg_f])


def test_tuple() -> None:
    @typed
    def _func_union(arg: tuple[Tensor[I32, [D1, D2]], Tensor[F32, [D1, D2]]] | int):
        pass

    @typed
    def _func_tuple(arg: tuple[Tensor[I32, [D1, D2]], Tensor[F32, [D1, D2]]]):
        pass

    @typed
    def _func_ellipses(arg: tuple[Tensor[I32, [D1, D2]], ...]):
        pass

    tensor_i32 = torch.zeros([1, 2], dtype=torch.int32)
    tensor_f32 = torch.zeros([1, 2], dtype=torch.float32)
    arg_int = 5

    _func_tuple((tensor_i32, tensor_f32))
    _func_union((tensor_i32, tensor_f32))
    _func_ellipses((tensor_i32, tensor_i32))

    # The following is always fine because we never type check
    # non-tensor objects:
    _func_tuple(arg_int)  # pyright: ignore

    # This is fine too:
    _func_union(arg_int)  # pyright: ignore

    with pytest.raises(TypeError):
        _func_union((tensor_f32, tensor_f32))

    with pytest.raises(TypeError):
        _func_union((tensor_f32, tensor_f32))

    with pytest.raises(TypeError):
        _func_ellipses((tensor_i32, tensor_f32))


def test_any() -> None:
    # Test: Tensor[Any, [2, 3]]
    # Test: tuple[Any | Tensor[F32, ...]]
    pass


def test_pytrees() -> None:
    @typed
    def _func_union(
        arg: tuple[Tensor[I32, [D1, D2]], list[Tensor[F32, [D1, D2]]]] | int,
    ):
        pass

    @typed
    def _func_tuple(
        arg: list[int | tuple[Tensor[I32, [D1, D2]], Tensor[F32, [D1, D2]]]],
    ):
        pass

    pass


# The following should show up red in the IDE,
# but no easy way to test for that.
# TODO: add to docs:
# def test_types() -> None:
#     _: Tensor[F32, [[Batches], D2]]  # should break!
#     _: Tensor[F32, D1, D2, D3]  # should break!
#     _: Tensor[F32, [1.32, D3]]  # should break!
#     _: Tensor[F32, [object()]]  # should break!

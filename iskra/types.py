# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import ast
import builtins
import ctypes
import dataclasses
import importlib
import inspect
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import typing
from abc import ABCMeta, abstractmethod
from collections import abc, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import EllipsisType, GenericAlias
from typing import (
    Any,
    AnyStr,
    ClassVar,
    Generic,
    Literal as L,
    NoReturn,
    TypeAlias,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    override,
)

import torch

import iskra.sparse as sp

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack


def _type_repr(obj):
    if isinstance(obj, type):
        if obj.__module__ in ["builtins", "iskra.types"]:
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    if isinstance(obj, tuple) or isinstance(obj, list):
        # Special case for `repr` of types with `ParamSpec`:
        return "[" + ", ".join(_type_repr(t) for t in obj) + "]"
    if isinstance(obj, types.GenericAlias):
        module = ""
        if obj.__module__ not in ["builtins", "iskra.types"]:
            module = f"{obj.__module__}."
        return f"{module}{_type_repr(obj.__origin__)}{_type_repr(obj.__args__)}"
    if isinstance(obj, typing.NewType) or isinstance(obj, typing.TypeAliasType):
        module = ""
        if obj.__module__ not in ["builtins", "iskra.types"]:
            module = f"{obj.__module__}."
        if (
            obj.__module__ == "iskra.types"
            and len(obj.__name__) == 2  # pyright: ignore
            and obj.__name__[0] == "D"  # pyright: ignore
            and obj.__name__[1].isnumeric()  # pyright: ignore
        ):
            return obj.__name__[1]  # pyright: ignore
        return f"{module}{obj.__name__}"  # pyright: ignore
    return repr(obj)


class _MetaAbstractDType(type):
    registered_dtypes: ClassVar[list[type]] = []

    def __init_subclass__(cls, **kwargs):
        cls.dtypes: Iterable[torch.dtype] = ()
        super().__init_subclass__(**kwargs)


class AbstractDType(metaclass=_MetaAbstractDType):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("AbstractDType cannot be instantiated.")

    def __init_subclass__(cls) -> None:
        if AbstractDType in cls.__bases__:
            cls.registered_dtypes.append(cls)


# fmt: off
class F16(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.float16, torch.half)
class BF16(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.bfloat16,)
class F32(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.float32, torch.float)
class F64(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.float64, torch.double)
class C32(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.complex32, torch.chalf)
class C64(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.complex64, torch.cfloat)
class C128(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.complex128, torch.cdouble)
class U8(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.uint8,)
class U16(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.uint16,)
class U32(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.uint32,)
class U64(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.uint64,)
class I8(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.int8,)
class I16(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.int16, torch.short)
class I32(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.int32,)
class I64(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.int64, torch.long)
class Bool(AbstractDType):
    dtypes: Iterable[torch.dtype] = (torch.bool,)
    
Batch = typing.NewType("Batch", int)
Batches = typing.NewType("Batches", list[int])
Verts = typing.NewType("Verts", int)
Edges = typing.NewType("Edges", int)
Tris = typing.NewType("Tris", int)
Tets = typing.NewType("Tets", int)
Dim = typing.NewType("Dim", int)

type D1 = L[1]
type D2 = L[2]
type D3 = L[3]
type D4 = L[4]
type D5 = L[5]
type D6 = L[6]
type D7 = L[7]
type D8 = L[8]
type D9 = L[9]
type D10 = L[10]
type D11 = L[11]
type D12 = L[12]
type D13 = L[13]
type D14 = L[14]
type D15 = L[15]
type D16 = L[16]
# fmt: on


def _is_shape_expr(obj) -> bool:
    if obj is Ellipsis or isinstance(obj, int) or isinstance(obj, list):
        return True
    obj = type(obj)
    names = "_ConcatenateGenericAlias", "ParamSpec"
    return obj.__module__ == "typing" and any(obj.__name__ == name for name in names)


class TensorGenericAlias(GenericAlias):
    def __new__(cls, origin, args):
        # print(f"{origin}: {args}")
        if not (isinstance(args, tuple) and len(args) == 2):
            raise TypeError("Tensor must be used as Tensor[DType, Shape].")
        t_dtype, t_shape = args
        if isinstance(t_shape, (tuple, list)):
            args = (t_dtype, *t_shape)
        elif not _is_shape_expr(t_shape):
            raise TypeError(
                f"Expected a list of types, an ellipsis, "
                f"ParamSpec, or Concatenate. Got {t_shape}"
            )
        return super().__new__(cls, origin, args)

    @override
    def __repr__(self):
        if len(self.__args__) == 2 and _is_shape_expr(self.__args__[1:]):
            return super().__repr__()
        shape_str: str = ", ".join([_type_repr(a) for a in self.__args__[1:]])
        return f"Tensor[{_type_repr(self.__args__[0])}, [{shape_str}]]"

    @override
    def __str__(self):
        if len(self.__args__) == 2 and _is_shape_expr(self.__args__[1:]):
            return super().__repr__()
        shape_str: str = ", ".join([_type_repr(a) for a in self.__args__[1:]])
        return f"Tensor[{_type_repr(self.__args__[0])}, [{shape_str}]]"

    # def __reduce__(self):
    #     args = self.__args__
    #     if not (len(args) == 2 and _is_shape_expr(args[0])):
    #         args = args[-1], list(args[:-1])
    #     return TensorGenericAlias, (TensorType, args)

    def __getitem__(self, item):
        print(f"{self}: {item}")
        # Called during TypeVar substitution, returns the custom subclass
        # rather than the default types.GenericAlias object.  Most of the
        # code is copied from typing's _GenericAlias and the builtin
        # types.GenericAlias.
        if not isinstance(item, tuple):
            item = (item,)

        new_args = super().__getitem__(item).__args__

        # args[0] occurs due to things like Z[[int, str, bool]] from PEP 612
        if not isinstance(new_args[0], (tuple, list)):
            t_result = new_args[-1]
            t_args = new_args[:-1]
            new_args = (t_args, t_result)
        return TensorGenericAlias(TensorT, tuple(new_args))

    # @classmethod
    # def __subclasshook__(cls, c):
    #     return c.__name__ == "Tensor"


class TensorT[DType: AbstractDType, **Shape](torch.Tensor):
    def __class_getitem__(cls, item) -> TensorGenericAlias:
        return TensorGenericAlias(cls, item)


type Tensor[DType: AbstractDType, **Shape] = TensorT[DType, Shape] | torch.Tensor


def _get_annotation_dtypes(name: str, annotation: Any) -> list[torch.dtype]:
    dtypes: list[torch.dtype] = []
    match annotation:
        case types.UnionType():
            for union_arg in typing.get_args(annotation):
                dtypes.extend(_get_annotation_dtypes(name, union_arg))
        case _MetaAbstractDType():
            dtypes.extend(annotation.dtypes)
        case _:
            raise TypeError(
                f"Tensor annotation of argument `{name}` has invalid DType: {annotation}. "
                f"Allowed DTypes are: {AbstractDType.registered_dtypes}."
            )
    return dtypes


def _check_annotation(name: str, annotation: Any, arg: Any) -> bool:
    # This function is only meant to run when torch.Tensor is passed into the function.
    if not (
        isinstance(arg, torch.Tensor) or isinstance(arg, list) or isinstance(arg, tuple)
    ):
        return True
    match typing.get_origin(annotation):
        case types.UnionType:
            success = False
            for union_arg in typing.get_args(annotation):
                success |= _check_annotation(name, union_arg, arg)
            return success
        case builtins.list:
            annotation_args = typing.get_args(annotation)
            if len(annotation_args) == 0:
                return True
            elif len(annotation_args) == 1:
                for elem in arg:
                    if not _check_annotation(name, annotation_args[0], elem):
                        return False
                return True
            else:
                raise TypeError(
                    f"Annotation '{name}: {_type_repr(annotation)}': wrong"
                    f"number of type args (={len(annotation_args)}), expected 0 or 1."
                )
        case builtins.tuple:
            annotation_args = typing.get_args(annotation)
            if len(annotation_args) == 0:
                return True
            elif len(annotation_args) == 2 and annotation_args[1] == builtins.Ellipsis:
                for elem in arg:
                    if not _check_annotation(name, annotation_args[0], elem):
                        return False
                return True
            if len(annotation_args) != len(arg):
                raise TypeError(
                    f"Annotation '{name}: {_type_repr(annotation)}': wrong"
                    f"number of type args (={len(annotation_args)}), tuple has length {len(arg)}."
                )
            else:
                for annotation_elem, elem in zip(annotation_args, arg):
                    if not _check_annotation(name, annotation_elem, elem):
                        return False
            return True
        case typing.TypeAliasType(__name__="Tensor"):
            annotation_args = typing.get_args(annotation)
            if len(annotation_args) == 0:
                return True
            elif len(annotation_args) == 2:
                allowed_dtypes = _get_annotation_dtypes(name, annotation_args[0])
                if arg.dtype not in allowed_dtypes:
                    return False
                shape_annotation = annotation_args[1:]
                return True
            else:
                raise TypeError(
                    f"Annotation '{name}: {_type_repr(annotation)}': wrong"
                    f"number of type args (={len(annotation_args)}), expected 0 or 2."
                )
        case _:
            # If torch.Tensor, but could not match above, must be wrong.
            print(f"Found unrecognized type {typing.get_origin(annotation)}??")
            return False
    return True


def typed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    signature: inspect.Signature = inspect.signature(
        func, locals=locals(), globals=globals()
    )
    parameters = list(signature.parameters.items())

    def check_param(name: str, param: inspect.Parameter, arg: Any) -> None:
        print(type(arg))
        # Need to handle PyTrees
        if (isinstance(arg, dict)) and not isinstance(arg, torch.Tensor):
            print("WARNING: PYTREES NOT SUPPORTED")
            return
        success = _check_annotation(name, param.annotation, arg)
        if not success:
            # DType must be returned from recursion bc of PyTrees:
            # f"Invalid dtype (={arg.dtype}) of argument "
            raise TypeError(
                # f"Invalid dtype (={arg.dtype}) of argument "
                f"`{name}: {_type_repr(param.annotation)}`"
            )

    def decorator(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            param_i: int = 0
            for param_i, arg in enumerate(args):
                name, param = parameters[param_i]
                if param.kind == param.VAR_POSITIONAL:
                    raise TypeError("Don't know how to handle variadics yet!")
                check_param(name, param, arg)

            for name, arg in kwargs.items():
                param = signature.parameters[name]
                if param.kind == param.VAR_KEYWORD:
                    raise TypeError("Don't know how to handle variadics yet!")
                check_param(name, param, arg)
            return func(*args, **kwargs)
        except TypeError as error:
            error.add_note(f"Type error in function {func.__name__}!")
            # TODO: add file and line of code to output
            raise error

    return decorator


mat_idcs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).T
mat_0_idcs = torch.cat([torch.full([mat_idcs.shape[1]], 0)[None, :], mat_idcs])
mat_1_idcs = torch.cat([torch.full([mat_idcs.shape[1]], 1)[None, :], mat_idcs])
tensor_idcs = torch.cat([mat_0_idcs, mat_1_idcs], -1)
mat_0_vals = torch.full([mat_idcs.shape[1]], 2.0)
mat_1_vals = torch.full([mat_idcs.shape[1]], 3.0)
tensor_vals = torch.cat([mat_0_vals, mat_1_vals])
print(tensor_idcs)
print(sp.coo_tensor(tensor_idcs, tensor_vals))

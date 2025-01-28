# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.


from typing import Any, Iterable

import torch
from typing_extensions import Self


class Quaternion(torch.Tensor):
    """Quaternion wrapper around a PyTorch Tensor.

    Quaternions are represented as 4-tuples:
    w_r + x_r i + y_r j + z_r k

    Helpful links:
    https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf
    https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    """

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor | Self,
        requires_grad: bool = False,
    ):
        t = tensor._tensor if isinstance(tensor, Quaternion) else tensor
        wrapper = torch.Tensor._make_subclass(
            cls,
            t,
            requires_grad,  # type: ignore
        )
        wrapper._tensor = t
        return wrapper

    @property
    def w(self) -> torch.Tensor:
        return self[..., 0]

    @property
    def x(self) -> torch.Tensor:
        return self[..., 1]

    @property
    def y(self) -> torch.Tensor:
        return self[..., 2]

    @property
    def z(self) -> torch.Tensor:
        return self[..., 3]

    def __matmul__(self, other: torch.Tensor | Self) -> Self:
        return matmul(self.real, other.real)  # type: ignore

    def conj(self) -> Self:
        return conj(self)  # type: ignore

    def inv(self) -> Self:
        return inv(self)  # type: ignore

    def norm_sq(self) -> torch.Tensor:
        return torch.Tensor(norm_sq(self))  # type: ignore

    def norm(self) -> torch.Tensor:  # type: ignore
        return torch.Tensor(norm(self))  # type: ignore

    def normalize(self) -> Self:
        return normalize(self)  # type: ignore

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return rotate(self, x)

    @property
    def matrix(self) -> torch.Tensor:
        return to_matrix(self)


def quaternion(
    data, *, dtype=None, device=None, requires_grad=False, pin_memory=False
) -> Quaternion:
    kwargs = {
        "dtype": dtype,
        "device": device,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    if isinstance(data, torch.Tensor) or isinstance(data, Quaternion):
        if dtype is None:
            kwargs["dtype"] = data.dtype
        if device is None:
            kwargs["device"] = data.device
        if requires_grad is None:
            kwargs["requires_grad"] = data.requires_grad
        if pin_memory is None:
            kwargs["pin_memory"] = data.pin_memory
        result = data.to(dtype=kwargs["dtype"], device=kwargs["device"])
        result = result.requires_grad_(kwargs["requires_grad"])
        if kwargs["pin_memory"]:
            result = result.pin_memory()
        return result.as_subclass(Quaternion)
    return torch.tensor(data, **kwargs).as_subclass(Quaternion)


def zeros(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> Quaternion:
    quat: torch.Tensor = torch.zeros(
        [*batch_shape, 4], dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion(quat)


def zeros_like(other: torch.Tensor | Quaternion) -> Quaternion:
    quat: torch.Tensor = torch.zeros(
        [*other.shape[:-1], 4],
        dtype=other.dtype,
        device=other.device,
        requires_grad=other.requires_grad,
    )
    return quaternion(quat)


def ones(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> Quaternion:
    quat: torch.Tensor = torch.ones(
        [*batch_shape, 4], dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion(quat)


def eye(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> Quaternion:
    quat = quaternion(
        [1, 0, 0, 0], dtype=dtype, device=device, requires_grad=requires_grad
    ).expand(*batch_shape, -1)
    return quat


def eye_like(other: torch.Tensor | Quaternion) -> Quaternion:
    return eye(
        other.shape[:-1],
        dtype=other.dtype,
        device=other.device,
        requires_grad=other.requires_grad,
    )


def look_at(
    origin: torch.Tensor, target: torch.Tensor, up: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    f = torch.nn.functional.normalize(target - origin, dim=-1, eps=eps)
    s = torch.nn.functional.normalize(
        torch.linalg.cross(f, up, dim=-1), dim=-1, eps=eps
    )
    u = torch.nn.functional.normalize(torch.linalg.cross(s, f, dim=-1), dim=-1, eps=eps)
    result = torch.zeros([*f.shape[:-1], 4, 4], dtype=f.dtype, device=f.device)
    result[..., 0, :3] = s
    result[..., 1, :3] = u
    result[..., 2, :3] = -f
    result[..., 0, 3] = -torch.sum(s * origin, dim=-1)
    result[..., 1, 3] = -torch.sum(u * origin, dim=-1)
    result[..., 2, 3] = torch.sum(f * origin, dim=-1)
    return result


def from_matrix(m: torch.Tensor) -> Quaternion:
    device = m.device
    # Converting a Rotation Matrix to a Quaternion - Mike Day, Insomniac Games
    t0 = 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
    q0 = quaternion(
        [
            m[..., 2, 1] - m[..., 1, 2],
            t0,
            m[..., 1, 0] + m[..., 0, 1],
            m[..., 0, 2] + m[..., 2, 0],
        ],
        device=device,
    )

    t1 = 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
    q1 = quaternion(
        [
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] + m[..., 0, 1],
            t1,
            m[..., 2, 1] + m[..., 1, 2],
        ],
        device=device,
    )
    t2 = 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]
    q2 = quaternion(
        [
            m[..., 1, 0] - m[..., 0, 1],
            m[..., 0, 2] + m[..., 2, 0],
            m[..., 2, 1] + m[..., 1, 2],
            t2,
        ],
        device=device,
    )

    t3 = 1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q3 = quaternion(
        [
            t3,
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1],
        ],
        device=device,
    )

    mask0 = m[..., 0, 0] > m[..., 1, 1]
    t01 = torch.where(mask0, t0, t1)
    q01 = torch.where(mask0, q0, q1)

    mask1 = m[..., 0, 0] < -m[..., 1, 1]
    t23 = torch.where(mask1, t2, t3)
    q23 = torch.where(mask1, q2, q3)

    mask2 = m[..., 2, 2] < 0.0
    t0123 = torch.where(mask2, t01, t23)
    q0123 = torch.where(mask2, q01, q23)

    return q0123 * 0.5 / torch.sqrt(t0123)  # type: ignore


def from_axis_angle(
    theta: torch.Tensor | float | int, axis: torch.Tensor | Iterable[float | int]
) -> Quaternion:
    half_theta = 0.5 * theta
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis)
    if isinstance(half_theta, float) or isinstance(half_theta, int):
        half_theta = torch.full_like(axis, half_theta)[..., :1]
    if half_theta.ndim == axis.ndim - 1:
        half_theta = half_theta[..., None]
    quat = torch.cat([torch.cos(half_theta), torch.sin(half_theta) * axis], -1)
    return quaternion(quat)


def from_point(x: torch.Tensor) -> Quaternion:
    return quaternion(torch.nn.functional.pad(x, (1, 0)))


def matmul(a: torch.Tensor | Quaternion, b: torch.Tensor | Quaternion) -> Quaternion:
    a, b = torch.broadcast_tensors(a, b)  # type: ignore
    w = a[..., 0] * b[..., 0] - torch.sum(a[..., 1:4] * b[..., 1:4], dim=-1)
    xyz = (
        a[..., 0:1] * b[..., 1:4]
        + b[..., 0:1] * a[..., 1:4]
        + torch.linalg.cross(a[..., 1:4], b[..., 1:4])
    )
    return quaternion(torch.cat([w[..., None], xyz], dim=-1))


def conj(q: torch.Tensor | Quaternion) -> Quaternion:
    return quaternion(torch.cat([q[..., :1], -q[..., 1:4]], -1))


def inv(p: torch.Tensor | Quaternion) -> Quaternion:
    return quaternion(conj(p) / norm_sq(p))


def norm_sq(q: torch.Tensor | Quaternion) -> Quaternion:
    return quaternion(torch.sum(q**2.0, -1, keepdim=True))


def norm(q: torch.Tensor | Quaternion) -> Quaternion:
    return quaternion(torch.sqrt(torch.sum(q**2.0, -1, keepdim=True)))


def normalize(q: torch.Tensor | Quaternion) -> Quaternion:
    return quaternion(q / norm(q))


def rotate(q: torch.Tensor | Quaternion, x: torch.Tensor) -> torch.Tensor:
    x_quat = from_point(x)
    r = q @ x_quat @ conj(q)
    return torch.tensor(r[..., 1:4])


def to_matrix(q: Quaternion) -> torch.Tensor:
    # Adapted from https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    result = torch.Tensor(q.new_zeros([*q.shape[:-1], 4, 4]))
    w = q.w
    x = q.x
    y = q.y
    z = q.z

    result[..., 0, 0] = w * w + x * x - y * y - z * z
    result[..., 0, 1] = 2 * x * y + 2 * w * z
    result[..., 0, 2] = 2 * x * z - 2 * w * y

    result[..., 1, 0] = 2 * x * y - 2 * w * z
    result[..., 1, 1] = w * w + y * y - x * x - z * z
    result[..., 1, 2] = 2 * y * z + 2 * w * x

    result[..., 2, 0] = 2 * x * z + 2 * w * y
    result[..., 2, 1] = 2 * y * z - 2 * w * x
    result[..., 2, 2] = w * w + z * z - x * x - y * y

    result[..., 3, 3] = 1.0
    return result

# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

from typing import Any, Iterable

import torch
from typing_extensions import Self

import iskra.geometry.quaternions as quat


class DualQuaternion(torch.Tensor):
    """Dual quaterntion implemented in PyTorch.

    Dual quaternions are represented as 8-tuples:
    (w_r + x_r i + y_r j + z_r k) + (w_d + x_d i + y_d j + z_d k) eps

    Helpful links:
    https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf
    https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf

    Args:
        torch (_type_): _description_
    """

    @property
    def real(self) -> quat.Quaternion:  # type: ignore
        return quat.quaternion(self[..., :4])

    @property
    def dual(self) -> quat.Quaternion:
        return quat.quaternion(self[..., -4:])

    def __matmul__(self, other: Self) -> Self:
        return matmul(self, other)  # type: ignore

    def qconj(self) -> Self:
        return qconj(self)  # type: ignore

    def dconj(self) -> Self:
        return dconj(self)  # type: ignore

    def normalize(self, eps: float = 1e-12) -> Self:
        return normalize(self, eps)  # type: ignore

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return transform(self, x)

    @property
    def rotation(self) -> quat.Quaternion:
        return to_rotation(self)

    @property
    def translation(self) -> torch.Tensor:
        return to_translation(self)

    @property
    def matrix(self) -> torch.Tensor:
        return to_matrix(self)


def dual_quaternion(*args: Any, **kwargs: Any) -> DualQuaternion:
    return DualQuaternion(*args, **kwargs)


def zeros(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> DualQuaternion:
    quat: torch.Tensor = torch.zeros(
        [*batch_shape, 8], dtype=dtype, device=device, requires_grad=requires_grad
    )
    return DualQuaternion(quat)


def zeros_like(other: torch.Tensor | DualQuaternion) -> DualQuaternion:
    quat: torch.Tensor = torch.zeros(
        [*other.shape[:-1], 8],
        dtype=other.dtype,
        device=other.device,
        requires_grad=other.requires_grad,
    )
    return DualQuaternion(quat)


def ones(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> DualQuaternion:
    quat: torch.Tensor = torch.ones(
        [*batch_shape, 8], dtype=dtype, device=device, requires_grad=requires_grad
    )
    return DualQuaternion(quat)


def eye(
    batch_shape: tuple[int, ...] | list[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> DualQuaternion:
    quat = torch.tensor(
        [1, 0, 0, 0, 0, 0, 0, 0],
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    quat = quat.expand(*batch_shape, -1)
    return DualQuaternion(quat.clone())


def eye_like(other: DualQuaternion) -> DualQuaternion:
    return eye(
        other.shape[:-1],
        dtype=other.dtype,
        device=other.device,
        requires_grad=other.requires_grad,
    )


def from_real_and_dual(
    real: torch.Tensor | quat.Quaternion, dual: torch.Tensor | quat.Quaternion
) -> DualQuaternion:
    return DualQuaternion(torch.cat([real, dual], dim=-1))


def from_point(x: torch.Tensor) -> DualQuaternion:
    point = eye(
        x.shape[:-1],
        dtype=x.dtype,
        device=x.device,
        requires_grad=x.requires_grad,
    )
    point.data[..., -3:] = x
    return point


def from_rigid(
    rotation: torch.Tensor | quat.Quaternion, translation: torch.Tensor
) -> DualQuaternion:
    rotation = quat.normalize(rotation)
    tq = torch.nn.functional.pad(translation, (1, 0))
    tq = 0.5 * quat.quaternion(tq) @ rotation
    return from_real_and_dual(rotation, tq)


def from_rotation(rotation: torch.Tensor | quat.Quaternion) -> DualQuaternion:
    rotation = quat.normalize(rotation)
    return from_real_and_dual(rotation, quat.zeros_like(rotation))


def from_axis_angle(
    theta: torch.Tensor | float | int, axis: torch.Tensor | Iterable[float | int]
) -> DualQuaternion:
    rotation = quat.normalize(quat.from_axis_angle(theta, axis))
    return from_real_and_dual(rotation, quat.zeros_like(rotation))


def from_translation(
    translation: torch.Tensor | Iterable[int | float],
) -> DualQuaternion:
    if not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation)
    tq = torch.nn.functional.pad(translation, (1, 0))
    tq = quat.quaternion(tq)
    rotation = quat.eye_like(tq)
    tq = 0.5 * tq @ rotation
    return from_real_and_dual(rotation, tq)


def look_at(
    origin: torch.Tensor | Iterable[int | float],
    target: torch.Tensor | Iterable[int | float],
    up: torch.Tensor | Iterable[int | float],
    eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up)
    camera_rotation = quat.look_at(origin, target, up, eps=eps)
    camera_rotation = quat.from_matrix(camera_rotation)
    translation = from_translation(target - origin)
    rotation = from_rotation(camera_rotation[None, ...])
    result = rotation @ translation
    return result


def matmul(a: DualQuaternion, b: DualQuaternion) -> DualQuaternion:
    real = a.real @ b.real
    dual1 = a.real @ b.dual
    dual2 = a.dual @ b.real
    return from_real_and_dual(real, dual1 + dual2)


def qconj(dq: DualQuaternion) -> DualQuaternion:
    return from_real_and_dual(dq.real.conj(), dq.dual.conj())


def dconj(dq: DualQuaternion) -> DualQuaternion:
    return from_real_and_dual(dq.real, -dq.dual)


def inv(dq: DualQuaternion) -> DualQuaternion:
    inv_rotation = quat.inv(dq.real)
    inv_translation = -2.0 * dq.dual @ dq.real.conj()
    inv_translation[..., -1] = 0
    inv_translation = 0.5 * quat.quaternion(inv_translation) @ inv_rotation

    return from_real_and_dual(inv_rotation, inv_translation)


def normalize(dq: DualQuaternion, eps: float = 1e-12) -> DualQuaternion:
    # return dq
    norm = dq.real.norm()
    # dot = torch.sum(dq.real * dq.dual, -1, keepdim=True)
    # real = dq.real / (real_norm + eps)  # type: ignore
    # dual = real_norm * dq.dual / dot #  / (real_norm + eps)
    # norm = torch.sqrt(qconj(dq) @ dq)
    return dq / (norm + eps)


# def sclerp(
#     start: DualQuaternion, end: DualQuaternion, t: float | torch.Tensor
# ) -> DualQuaternion:
#     dot = torch.sum(start.real * end.real, -1, keepdim=True)
#     end = end * torch.sign(dot)  # type: ignore
#     diff = start.conj() @ end
#     r = torch.sqrt(diff.real[..., 1:] * diff.dual[..., 1:])


def interpolate(transforms: DualQuaternion, weights: torch.Tensor) -> DualQuaternion:
    if weights.ndim != transforms.ndim - 1 or weights.shape[-1] != transforms.shape[-2]:
        raise ValueError(
            "Incompatible tensor shapes."
            "weights has to be [..., n_transforms] but is {weights.shape}, "
            f"transforms has to be [..., n_transforms, 8] but is {transforms.shape}."
        )
    interpolated = torch.sum(transforms * weights[..., None], -2)
    return normalize(interpolated)  # type: ignore


def to_rotation(dq: DualQuaternion) -> quat.Quaternion:
    return dq.real


def to_translation(dq: DualQuaternion) -> torch.Tensor:
    t = 2.0 * dq.dual @ dq.real.conj()
    return torch.Tensor(t[..., 1:4])


def to_matrix(dq: DualQuaternion) -> torch.Tensor:
    # Adapted from https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    result = torch.Tensor(dq.new_zeros([*dq.shape[:-1], 4, 4]))
    w = dq.real.w
    x = dq.real.x
    y = dq.real.y
    z = dq.real.z
    t = to_translation(dq)

    result[..., 0, 0] = w * w + x * x - y * y - z * z
    result[..., 0, 1] = 2 * x * y + 2 * w * z
    result[..., 0, 2] = 2 * x * z - 2 * w * y

    result[..., 1, 0] = 2 * x * y - 2 * w * z
    result[..., 1, 1] = w * w + y * y - x * x - z * z
    result[..., 1, 2] = 2 * y * z + 2 * w * x

    result[..., 2, 0] = 2 * x * z + 2 * w * y
    result[..., 2, 1] = 2 * y * z - 2 * w * x
    result[..., 2, 2] = w * w + z * z - x * x - y * y

    result[..., 3, 0] = t[..., 0]
    result[..., 3, 1] = t[..., 1]
    result[..., 3, 2] = t[..., 2]
    result[..., 3, 3] = 1.0
    return result


def transform(dq: DualQuaternion, x: torch.Tensor) -> torch.Tensor:
    # alternatively, we could use the following formulation,
    # but that might change the gradients during autodiff:
    # return DualQuat.quat_rotate(self.rotation, x) + self.translation
    xq = from_point(x)
    return torch.Tensor(dq @ xq @ dq.qconj())[..., -3:]

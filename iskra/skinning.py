# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

import dataclasses
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from easing_functions import CubicEaseInOut

from iskra.mesh import BBox, Mesh

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self

import torch

from iskra.geometry import barycentric_interpolate, dual_quat, edge_lengths, quat
from iskra.topology import scatter_vertex_values


@dataclass
class Transform:
    scale: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.tensor([1.0, 1.0, 1.0])
    )
    rigid: dual_quat.DualQuaternion = dataclasses.field(
        default_factory=lambda: dual_quat.eye([1])
    )

    def __matmul__(self, other: Self) -> Self:
        return Transform(scale=other.scale * self.scale, rigid=self.rigid @ other.rigid)

    @property
    def rotation(self) -> torch.Tensor:
        return self.rigid.rotation

    @property
    def translation(self) -> torch.Tensor:
        return self.rigid.translation

    @property
    def scale_matrix(self) -> torch.Tensor:
        result = self.scale.new_zeros([*self.scale.shape[:-1], 4, 4])
        result[..., 0, 0] = self.scale[..., 0]
        result[..., 1, 1] = self.scale[..., 1]
        result[..., 2, 2] = self.scale[..., 2]
        result[..., 3, 3] = 1.0
        return result

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return self.rigid.rotation.matrix

    @property
    def translation_matrix(self) -> torch.Tensor:
        translation = self.rigid.translation
        result = translation.new_zeros([*translation.shape[:-1], 4, 4])
        result[..., 0, 0] = 1.0
        result[..., 1, 1] = 1.0
        result[..., 2, 2] = 1.0
        result[..., 3, 3] = 1.0
        result[..., 3, :3] = translation
        return result

    @property
    def matrix(self) -> torch.Tensor:
        result = dual_quat.to_matrix(self.rigid)
        result[..., 0, :] = result[..., 0, :] * self.scale[..., 0:1]
        result[..., 1, :] = result[..., 1, :] * self.scale[..., 1:2]
        result[..., 2, :] = result[..., 2, :] * self.scale[..., 2:3]
        return result


def axis_angle_between_vectors(
    v1: torch.Tensor, v2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v1_unit = torch.nn.functional.normalize(v1, p=2, dim=-1)
    v2_unit = torch.nn.functional.normalize(v2, p=2, dim=-1)
    axis = torch.linalg.cross(v1_unit, v2_unit, dim=-1)
    sin_theta = torch.linalg.vector_norm(axis, dim=-1)
    cos_theta = torch.sum(v1_unit * v2_unit, dim=-1)
    axis = torch.nn.functional.normalize(axis, p=2, dim=-1)
    angle = torch.atan2(sin_theta, cos_theta)
    return axis, angle


def transform_from_points(rest_points, points) -> dual_quat.DualQuaternion:
    return dual_quat.from_translation(points - rest_points)


def transform_from_bones(
    rest_bones: torch.Tensor, bones: torch.Tensor
) -> tuple[Transform, Transform]:
    rest_bone_length = edge_lengths(rest_bones)
    bone_length = edge_lengths(bones)

    scale = bone_length / rest_bone_length

    rest_bone_vec = rest_bones[..., 1, :] - rest_bones[..., 0, :]
    bone_vec = bones[..., 1, :] - bones[..., 0, :]
    axis, angle = axis_angle_between_vectors(bone_vec, rest_bone_vec)
    unit_y = torch.zeros_like(rest_bone_vec)
    unit_y[..., 1] = 1.0
    rest_axis, rest_angle = axis_angle_between_vectors(rest_bone_vec, unit_y)

    rest_origin = rest_bones.mean(-2)
    origin = bones.mean(-2)

    inv_rest_rotation = dual_quat.from_rotation(
        quat.from_axis_angle(-rest_angle[..., None], rest_axis)
    )
    inv_rest_translation = dual_quat.from_translation(-rest_origin)
    inv_rest_transform = inv_rest_rotation @ inv_rest_translation

    rest_rotation = dual_quat.from_rotation(
        quat.from_axis_angle(rest_angle[..., None], rest_axis)
    )
    rotation = dual_quat.from_rotation(quat.from_axis_angle(-angle[..., None], axis))
    full_rotation = rotation @ rest_rotation
    full_translation = dual_quat.from_translation(origin)
    full_transform = full_translation @ full_rotation

    scale = torch.stack([torch.ones_like(scale), scale, torch.ones_like(scale)], -1)
    full_transform = Transform(scale=scale, rigid=full_transform)
    return inv_rest_transform, full_transform


def deform_bones(bones: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    if transforms.shape[-2:] == (4, 4):
        transforms = transforms.flatten(-2, -1)
    else:
        # dot = torch.sum(transforms * transforms[0:1, ...], -1)
        # transforms[dot < 0] = -transforms[dot < 0]
        transforms = dual_quat.normalize(transforms)
    if transforms.shape[-1] == 16:
        transforms = transforms.reshape(-1, 4, 4)
    else:
        transforms = dual_quat.normalize(transforms)
        transforms = dual_quat.to_matrix(transforms)
    bones_h = torch.nn.functional.pad(
        bones.to(device=transforms.device), pad=(0, 1), value=1.0
    )[..., None]
    deformed: torch.Tensor = (bones_h.mT @ transforms[..., None, :, :]).mT
    deformed = deformed.squeeze(-1)
    deformed = deformed[..., :3] / deformed[..., -1:]
    return deformed


class SkinningHandles(torch.nn.Module):
    def __init__(
        self,
        vertices: torch.Tensor,
        point_handle_idx: torch.Tensor,
        bone_handle_idx: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("rest_vertices", vertices.clone())
        self.register_buffer("vertices", vertices.clone())
        self.n_vertices = vertices.shape[0]
        self.register_buffer("point_handle_idx", point_handle_idx)
        self.n_points = point_handle_idx.shape[0]
        self.register_buffer("bone_handle_idx", bone_handle_idx)
        self.n_bones = bone_handle_idx.shape[0]

        self.n_handles = self.n_points + self.n_bones

        if TYPE_CHECKING:
            self.vertices: torch.Tensor
            self.rest_vertices: torch.Tensor
            self.point_handle_idx: torch.Tensor
            self.bone_handle_idx: torch.Tensor

    def normalize(self, bbox: BBox | None = None) -> None:
        if bbox is not None:
            bbox = bbox.to(self.vertices.device)
        else:
            bbox = self.bbox
        max_extent = torch.max(bbox.extent)
        self.vertices = (self.vertices - bbox.min) / max_extent
        self.rest_vertices = (self.rest_vertices - bbox.min) / max_extent
        # self.vertices = (self.vertices - bbox.min - bbox.extent / 2) / max_extent + 0.5

    @property
    def rest_points(self) -> torch.Tensor:
        return scatter_vertex_values(self.rest_vertices, self.point_handle_idx)

    @property
    def rest_bones(self) -> torch.Tensor:
        return scatter_vertex_values(self.rest_vertices, self.bone_handle_idx)

    @property
    def points(self) -> torch.Tensor:
        return scatter_vertex_values(self.vertices, self.point_handle_idx)

    @property
    def bones(self) -> torch.Tensor:
        return scatter_vertex_values(self.vertices, self.bone_handle_idx)

    def reset_positions(self) -> None:
        self.vertices = self.rest_vertices.clone()

    def bone_samples(self, n_samples: int, eps: float = 1e-3) -> torch.Tensor:
        bones = self.bones
        if bones.shape[0] == 0:
            return torch.zeros(
                [0, 2, self.vertices.shape[-1]],
                device=self.vertices.device,
                dtype=self.vertices.dtype,
            )
        samples_per_bone = n_samples // bones.shape[0]
        bary = torch.linspace(
            0 + eps, 1 - eps, samples_per_bone, device=self.vertices.device
        )
        bary = torch.stack([bary, 1.0 - bary], -1)
        samples_list = [
            barycentric_interpolate(bones[i : i + 1, ...], bary)
            for i in range(bones.shape[0])
        ]
        samples = torch.stack(samples_list, 0)
        return samples

    def bone_to_weight_idx(self, bone_idx: int) -> int:
        return bone_idx + self.n_points

    def weight_to_bone_idx(self, weight_idx: int) -> int:
        return weight_idx - self.n_points

    def weight_to_point_idx(self, weight_idx: int) -> int:
        return weight_idx

    def point_to_weight_idx(self, point_idx: int) -> int:
        return point_idx

    def vertex_to_weight_idx(self, vertex_idx: int) -> tuple[int, ...]:
        in_points = torch.nonzero(self.point_handle_idx.flatten() == vertex_idx)
        if in_points.shape[0] != 0:
            return in_points
        in_handles = torch.nonzero(self.bone_handle_idx == vertex_idx)
        if in_handles.shape[0] != 0:
            return in_handles[:, 0] + self.n_points
        return in_points

    def get_transforms(self):
        point_transform = transform_from_points(
            self.rest_points, self.points
        ).get_matrix()
        inv_rest_bones_transform, bones_transform = transform_from_bones(
            self.rest_bones, self.bones
        )
        bones_transform = inv_rest_bones_transform.compose(bones_transform).get_matrix()
        transforms = torch.cat([point_transform, bones_transform], 0)
        return transforms

    def deform(
        self,
        vertices: torch.Tensor,
        weights: torch.Tensor,
        transforms: torch.Tensor,
    ) -> torch.Tensor:
        if transforms.shape[-2:] == (4, 4):
            transforms = transforms.flatten(-2, -1)
        else:
            # dot = torch.sum(transforms * transforms[0:1, ...], -1)
            # transforms[dot < 0] = -transforms[dot < 0]
            transforms = dual_quat.normalize(transforms)
        interpolated_transform = (
            torch.tensor(weights, dtype=torch.float32, device=transforms.device)
            @ transforms
        )
        if interpolated_transform.shape[-1] == 16:
            interpolated_transform = interpolated_transform.reshape(-1, 4, 4)
        else:
            interpolated_transform = dual_quat.normalize(interpolated_transform)
            interpolated_transform = dual_quat.to_matrix(interpolated_transform)
        vertices_h = torch.nn.functional.pad(
            vertices.to(device=transforms.device), pad=(0, 1), value=1.0
        )[..., None]
        deformed: torch.Tensor = (vertices_h.mT @ interpolated_transform).mT
        deformed = deformed.squeeze(-1)
        deformed = deformed[:, :3] / deformed[:, -1:]
        return deformed

    @classmethod
    def from_path(
        cls, path: Path | str, device: str | torch.device = "cuda"
    ) -> Self | tuple[Self, torch.Tensor]:
        mesh = Mesh.from_path(path, device=device)
        return SkinningHandles(
            mesh.geometry.vertices,
            mesh.topology.isolated_vertices,
            mesh.topology.edges,
        )


def interpolate_transforms(
    weights: torch.Tensor, transforms: torch.Tensor
) -> torch.Tensor:
    if transforms.shape[-2:] == (4, 4):
        transforms = transforms.flatten(-2, -1)
    else:
        transforms = dual_quat.normalize(transforms)
    interpolated_transform = (
        torch.tensor(weights, dtype=torch.float32, device=transforms.device)
        @ transforms
    )
    if interpolated_transform.shape[-1] == 16:
        interpolated_transform = interpolated_transform.reshape(-1, 4, 4)
    else:
        interpolated_transform = dual_quat.normalize(interpolated_transform)
    return interpolated_transform


def deform_vertices(vertices: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    if transforms.shape[-1] == 8:
        transforms = dual_quat.to_matrix(transforms)
    vertices_homo = torch.nn.functional.pad(
        vertices.to(device=transforms.device), pad=(0, 1), value=1.0
    )[..., None]
    deformed: torch.Tensor = (vertices_homo.mT @ transforms).mT
    deformed = deformed.squeeze(-1)
    deformed = deformed[:, :3] / deformed[:, -1:]
    return deformed


def deform_normals(normals: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    if transforms.shape[-1] == 8:
        transforms = dual_quat.to_matrix(transforms)
    transforms = torch.linalg.inv(
        transforms[..., :3, :3] + 1e-3 * torch.eye(3, device=transforms.device)
    ).mT
    deformed: torch.Tensor = (normals[..., None].mT @ transforms).mT
    deformed = deformed.squeeze(-1)
    return deformed


def handles_to_transforms(
    handles: SkinningHandles, deformed: Iterable[SkinningHandles]
) -> list[Transform]:
    rest_points = handles.rest_points
    rest_bones = handles.rest_bones
    previous_transforms = None
    transforms = []
    for frame in deformed:
        # Blender only guarantees that the vertex order won't be changed when exporting
        # .obj files, but the line-segment order is free to change.
        # For this reason, we must use the index of the original handles,
        # not the deformed handles (i.e. frame.bones is not guaranteed to work).
        points = scatter_vertex_values(frame.vertices, handles.point_handle_idx)
        bones = scatter_vertex_values(frame.vertices, handles.bone_handle_idx)
        point_transform = transform_from_points(rest_points, points)
        inv_rest_bones_transform, bones_transform = transform_from_bones(
            rest_bones, bones
        )
        bones_transform = bones_transform.rigid @ inv_rest_bones_transform
        frame_transforms = torch.cat([point_transform, bones_transform], 0)
        if previous_transforms is not None:
            # dot = torch.sum(frame_transforms.real * previous_transforms.real, -1)
            # print(dot)
            # frame_transforms.real[dot < 0] = -frame_transforms.real[dot < 0]
            # frame_transforms.dual[dot < 0] = -frame_transforms.dual[dot < 0]
            frame_transforms = frame_transforms @ previous_transforms
        transforms.append(frame_transforms)
        previous_transforms = frame_transforms
        rest_bones = bones
        rest_points = points
    return transforms


@dataclass
class KeyFrame:
    t: float
    transforms: torch.Tensor


class KeyFramedAnimation:
    def __init__(self, fps: float, duration: float) -> None:
        self.fps = fps
        self.duration = duration
        self.keyframes = []
        self.easing = CubicEaseInOut(start=0, end=self.duration, duration=self.duration)

    def add_key_frame(self, keyframe: KeyFrame):
        idx = bisect_left(self.keyframes, keyframe.t, key=lambda x: x.t)
        self.keyframes.insert(idx, keyframe)

    def tween(self, t: float) -> torch.Tensor:
        t = self.easing(t)
        idx = bisect_left(self.keyframes, t, key=lambda x: x.t)
        if idx == 0:
            transforms = self.keyframes[0].transforms
            # transforms = dual_quat.to_matrix(self.keyframes[0].transforms)
        elif idx == len(self.keyframes):
            # transforms = dual_quat.to_matrix(self.keyframes[-1].transforms)
            transforms = self.keyframes[-1].transforms
        else:
            t1 = self.keyframes[idx - 1].t
            t2 = self.keyframes[idx].t
            t_scaled = (t - t1) / (t2 - t1)
            t1 = dual_quat.normalize(self.keyframes[idx - 1].transforms)
            t2 = dual_quat.normalize(self.keyframes[idx].transforms)
            # t1 = dual_quat.to_matrix(self.keyframes[idx - 1].transforms)
            # t2 = dual_quat.to_matrix(self.keyframes[idx].transforms)
            transforms = (1 - t_scaled) * t1 + t_scaled * t2
        transforms = dual_quat.normalize(transforms)
        return transforms

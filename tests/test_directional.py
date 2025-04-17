# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.


import numpy as np
import pytest
import torch

from iskra.directional import (
    face_tangent_bundle,
    to_extrinsic,
    to_extrinsic_n_rosy,
    to_intrinsic,
    to_intrinsic_n_rosy,
    transport_from_face,
)
from iskra.topology import edge_flaps


@pytest.fixture
def no_boundary() -> tuple[torch.Tensor, torch.Tensor]:
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    )
    return verts, faces


def test_vertex_tangent_bundle(no_boundary: tuple[torch.Tensor, torch.Tensor]) -> None:
    verts, faces = no_boundary
    flaps = edge_flaps(faces)
    tangents, binormals, connection = face_tangent_bundle(verts, faces, flaps)

    source = 0
    intrinsic = to_intrinsic(torch.tensor([1.0, 0.0, 0]), tangents[0], binormals[0])
    transported = transport_from_face(
        source, intrinsic, faces.shape[0], flaps, connection, 1
    )
    extrinsic = to_extrinsic(transported, tangents, binormals)
    expected_face_1 = torch.tensor([0.0, 0.0, -1.0])
    expected_face_2 = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(extrinsic[1], expected_face_1)
    torch.testing.assert_close(extrinsic[2], expected_face_2)


def test_n_rosy_transport(no_boundary: tuple[torch.Tensor, torch.Tensor]) -> None:
    verts, faces = no_boundary
    flaps = edge_flaps(faces)
    tangents, binormals, connection = face_tangent_bundle(verts, faces, flaps)

    source = 0
    n = 4
    intrinsic = to_intrinsic_n_rosy(
        torch.tensor([1.0, 0.0, 0]), tangents[0], binormals[0], n
    )
    transported = transport_from_face(
        source, intrinsic, faces.shape[0], flaps, connection, n
    )
    extrinsic = to_extrinsic_n_rosy(transported, tangents, binormals, n)

    expected_face_2 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(extrinsic[2, ...], expected_face_2)

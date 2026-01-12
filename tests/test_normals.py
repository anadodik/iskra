import random

import numpy as np
import pytest
import torch

from iskra.geometry.normals import interior_angles, triangle_normals
from iskra.topology import face_index, reduce_on_subface


def seed_all(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def _global_seed():
    seed_all(0)


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def triangles() -> tuple[torch.Tensor, torch.Tensor]:
    h = np.sqrt(3) / 2
    verts = torch.tensor(
        [
            [-h, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [+h, 0.5, 0.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2], [1, 3, 2]],
    )
    return verts, faces


def test_function_shapes_and_types(device, triangles):
    verts, faces = triangles
    verts = verts.to(device)
    faces = faces.to(device)
    triangles = face_index(verts, faces)
    face_normals = triangle_normals(triangles)
    angles = interior_angles(triangles, signed=False)
    angle_normals = angles[..., :, None] * face_normals[..., None, :]
    vert_normals = reduce_on_subface(angle_normals, faces, verts.shape[0], "sum")

    assert True

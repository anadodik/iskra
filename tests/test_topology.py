# Copyright (c) 2023 - present, Ana Dodik. All rights reserved.

import itertools
from collections.abc import Iterable
from functools import partial
from typing import Literal

import pytest
import torch
from numpy import isin

from iskra.topology import (
    connected_components,
    edge_flaps,
    get_subfaces,
    reduce_on_subface,
)

assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


@pytest.fixture
def tetrahedra() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 2, 3], [4, 1, 3, 2]],
        dtype=torch.int64,
    )


@pytest.fixture
def triangles() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 2], [1, 3, 2]],
        dtype=torch.int64,
    )


@pytest.fixture
def edges() -> torch.Tensor:
    return torch.tensor(
        [[0, 1], [1, 2], [2, 0]],
        dtype=torch.int64,
    )


@pytest.fixture
def disconnected() -> tuple[int, torch.Tensor]:
    n_vertices = 6
    faces = torch.tensor(
        [[1, 2, 3], [2, 4, 3]],
        dtype=torch.int64,
    )
    return n_vertices, faces


def test_tetrahedra_subfaces(tetrahedra: torch.Tensor) -> None:
    tris, tets_to_tris, tets_to_tris_sign = get_subfaces(tetrahedra)
    tris_expected = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    )

    torch.testing.assert_close(tris, tris_expected, rtol=0, atol=0)

    tets_to_tris_expected = torch.tensor([[0, 1, 2, 3], [5, 4, 6, 3]])
    torch.testing.assert_close(tets_to_tris, tets_to_tris_expected, rtol=0, atol=0)

    tets_to_tris_sign_expeceted = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, -1.0]]
    )
    torch.testing.assert_close(
        tets_to_tris_sign, tets_to_tris_sign_expeceted, rtol=0, atol=0
    )
    assert_equal(tets_to_tris_sign, tets_to_tris_sign_expeceted)


def test_triangles_subfaces(triangles: torch.Tensor) -> None:
    edges, faces_to_edges, face_to_edge_sign = get_subfaces(triangles)

    edges_expected = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    assert_equal(edges, edges_expected)

    faces_to_edges_expected = torch.tensor([[2, 1, 0], [4, 2, 3]])
    assert_equal(faces_to_edges, faces_to_edges_expected)

    face_to_edge_sign_expected = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    assert_equal(face_to_edge_sign, face_to_edge_sign_expected)


def test_edge_flaps(triangles: torch.Tensor) -> None:
    flaps = edge_flaps(triangles)
    flaps_expected = torch.tensor([[0, -1], [-1, 0], [0, 1], [1, -1], [-1, 1]])
    assert_equal(flaps, flaps_expected)


def test_edges_subfaces(edges: torch.Tensor) -> None:
    verts, edges_to_verts, edges_to_verts_sign = get_subfaces(edges, 0)
    verts_expected = torch.tensor([[0], [1], [2]])

    torch.testing.assert_close(verts, verts_expected, rtol=0, atol=0)
    edges_to_verts_expected = torch.tensor([[1, 0], [2, 1], [0, 2]])

    torch.testing.assert_close(edges_to_verts, edges_to_verts_expected, rtol=0, atol=0)
    edges_to_verts_sign_expected = torch.ones_like(
        edges_to_verts_expected, dtype=torch.float32
    )
    torch.testing.assert_close(
        edges_to_verts_sign, edges_to_verts_sign_expected, rtol=0, atol=0
    )


def test_connected_components(disconnected: tuple[int, torch.Tensor]):
    n_components, vertex_labels, face_labels = connected_components(*disconnected)
    assert n_components == 3
    assert_equal(vertex_labels, torch.tensor([0, 1, 1, 1, 1, 2]))
    assert_equal(face_labels, torch.tensor([1, 1]))


def test_scatter(triangles: torch.Tensor):
    n_verts = 6
    n_tris = triangles.shape[0]

    # Equivalent of averaging face scalars on vertices:
    data, data_ndim = torch.randn([n_tris]), 0
    expected = torch.zeros([n_verts])
    for f in range(n_tris):
        for c in range(3):  # corner
            expected[triangles[f, c]] += data[f]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of averaging face scalars on vertices:
    data, data_ndim = torch.randn([n_tris, 3]), 0
    expected = torch.zeros([n_verts])
    for f in range(n_tris):
        for c in range(3):  # corner
            expected[triangles[f, c]] += data[f, c]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of averaging corner scalars on vertices:
    data, data_ndim = torch.randn([n_tris, 3]), 0
    expected = torch.zeros([n_verts])
    for f in range(n_tris):
        for c in range(3):  # corner
            expected[triangles[f, c]] += data[f, c]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of averaging face normals on vertices:
    data, data_ndim = torch.randn([n_tris, 3]), 1
    expected = torch.zeros([n_verts, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(data.shape[-data_ndim]):
                expected[triangles[f, c], d] += data[f, d]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of weighted averaging of face normals on vertices:
    data, data_ndim = torch.randn([n_tris, 3, 3]), 1
    expected = torch.zeros([n_verts, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(data.shape[-data_ndim]):
                expected[triangles[f, c], d] += data[f, c, d]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of weighted averaging of covariances on vertices:
    data, data_ndim = torch.randn([n_tris, 3, 3]), 2
    expected = torch.zeros([n_verts, 3, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            data_iter = itertools.product(
                *(
                    range(data.shape[dim])
                    for dim in range(data.ndim - data_ndim, data.ndim)
                )
            )
            for ds in data_iter:
                expected[triangles[f, c], *ds] += data[f, *ds]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)


def test_batched_scatter(triangles: torch.Tensor):
    batch_size = 16
    n_verts = 6
    n_tris = triangles.shape[0]
    triangles = triangles[None, :, :].expand(batch_size, -1, -1)

    # Equivalent of averaging face scalars on vertices:
    data, data_ndim = torch.randn([batch_size, n_tris]), 0
    expected = torch.zeros([batch_size, n_verts])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                expected[b, triangles[b, f, c]] += data[b, f]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim, batch_ndim=1
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of averaging corner scalars on vertices:
    data, data_ndim = torch.randn([batch_size, n_tris, 3]), 0
    expected = torch.zeros([batch_size, n_verts])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                expected[b, triangles[b, f, c]] += data[b, f, c]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim, batch_ndim=1
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of averaging face normals on vertices:
    data, data_ndim = torch.randn([batch_size, n_tris, 3]), 1
    expected = torch.zeros([batch_size, n_verts, 3])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                for d in range(data.shape[-data_ndim]):
                    expected[b, triangles[b, f, c], d] += data[b, f, d]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim, batch_ndim=1
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of weighted averaging of face normals on vertices:
    data, data_ndim = torch.randn([batch_size, n_tris, 3, 3]), 1
    expected = torch.zeros([batch_size, n_verts, 3])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                for d in range(data.shape[-data_ndim]):
                    expected[b, triangles[b, f, c], d] += data[b, f, c, d]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim, batch_ndim=1
    )
    torch.testing.assert_close(result, expected)

    # Equivalent of weighted averaging of covariances on vertices:
    data, data_ndim = torch.randn([batch_size, n_tris, 3, 3]), 2
    expected = torch.zeros([batch_size, n_verts, 3, 3])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                data_iter = itertools.product(
                    *(
                        range(data.shape[dim])
                        for dim in range(data.ndim - data_ndim, data.ndim)
                    )
                )
                for ds in data_iter:
                    expected[b, triangles[b, f, c], *ds] += data[b, f, *ds]
    result = reduce_on_subface(
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim, batch_ndim=1
    )
    torch.testing.assert_close(result, expected)

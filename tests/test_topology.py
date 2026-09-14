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
    face_index,
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

    tets_to_tris_expected = torch.tensor([[3, 2, 1, 0], [3, 6, 4, 5]])
    torch.testing.assert_close(tets_to_tris, tets_to_tris_expected, rtol=0, atol=0)

    tets_to_tris_sign_expeceted = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1.0]]
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
    edges_to_verts_sign_expected = torch.zeros_like(
        edges_to_verts_expected, dtype=torch.float32
    )
    edges_to_verts_sign_expected[:, 0] = 1
    edges_to_verts_sign_expected[:, 1] = -1
    torch.testing.assert_close(
        edges_to_verts_sign, edges_to_verts_sign_expected, rtol=0, atol=0
    )


def test_connected_components(disconnected: tuple[int, torch.Tensor]):
    n_components, vertex_labels, face_labels = connected_components(*disconnected)
    assert n_components == 3
    assert_equal(vertex_labels, torch.tensor([0, 1, 1, 1, 1, 2]))
    assert_equal(face_labels, torch.tensor([1, 1]))


def test_gather_tets(tetrahedra: torch.Tensor):
    n_tets = tetrahedra.shape[0]

    tris, tet_tri, _ = get_subfaces(tetrahedra)
    edges, tri_edge, _ = get_subfaces(tris)
    tet_tri_edge = face_index(tri_edge, tet_tri)
    face_ndim = 3

    # Gathering scalars
    data = torch.randn([edges.shape[0]])
    expected = torch.zeros([n_tets, 4, 3])
    for f in range(n_tets):
        for t in range(4):  # triangle
            for e in range(3):  # edge
                expected[f, t, e] = data[tet_tri_edge[f, t, e]]
    result = face_index(data, tet_tri_edge, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Gathering vectors:
    data = torch.randn([edges.shape[0], 3])
    expected = torch.zeros([n_tets, 4, 3, 3])
    for f in range(n_tets):
        for t in range(4):  # triangle
            for e in range(3):  # edge
                for d in range(3):
                    expected[f, t, e, d] = data[tet_tri_edge[f, t, e], d]
    result = face_index(data, tet_tri_edge, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Gathering batched vectors:
    data = torch.randn([8, edges.shape[0], 3])
    expected = torch.zeros([8, n_tets, 4, 3, 3])
    for b in range(8):
        for f in range(n_tets):
            for t in range(4):  # triangle
                for e in range(3):  # edge
                    for d in range(3):
                        expected[b, f, t, e, d] = data[b, tet_tri_edge[f, t, e], d]
    result = face_index(data, tet_tri_edge.expand(8, -1, -1, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)


def test_gather_tris(triangles: torch.Tensor):
    n_verts = 6
    n_tris = triangles.shape[0]
    face_ndim = 2

    # Equivalent of gathering vertex scalars onto triangles:
    data = torch.randn([n_verts])
    expected = torch.zeros([n_tris, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(3):
                expected[f, c] = data[triangles[f, c]]
    result = face_index(data, triangles, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering batched vertex scalars onto triangles:
    data = torch.randn([8, n_verts])
    expected = torch.zeros([8, n_tris, 3])
    for b in range(8):
        for f in range(n_tris):
            for c in range(3):  # corner
                for d in range(3):
                    expected[b, f, c] = data[b, triangles[f, c]]
    result = face_index(data, triangles.expand(8, -1, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles.expand(8, -1, -1))
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex vectors onto triangles:
    data = torch.randn([n_verts, 3])
    expected = torch.zeros([n_tris, 3, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(3):
                expected[f, c, d] = data[triangles[f, c], d]
    result = face_index(data, triangles, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex matrices onto triangles:
    data = torch.randn([n_verts, 3, 3])
    expected = torch.zeros([n_tris, 3, 3, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(3):
                expected[f, c, d] = data[triangles[f, c], d]
    result = face_index(data, triangles, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex matrices onto triangles:
    data = torch.randn([n_verts, 3, 3])
    expected = torch.zeros([n_tris, 3, 3, 3])
    for f in range(n_tris):
        for c in range(3):  # corner
            for d in range(3):
                expected[f, c, d] = data[triangles[f, c], d]
    result = face_index(data, triangles, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering batched vertex matrices onto triangles:
    data = torch.randn([8, n_verts, 3, 3])
    expected = torch.zeros([8, n_tris, 3, 3, 3])
    for b in range(8):
        for f in range(n_tris):
            for c in range(3):  # corner
                for d in range(3):
                    expected[b, f, c, d] = data[b, triangles[f, c], d]
    result = face_index(data, triangles.expand(8, -1, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    result = face_index(data, triangles.expand(8, -1, -1))
    torch.testing.assert_close(result, expected)


def test_gather_verts():
    isolated_verts = torch.tensor([0, 2, 4])
    n_verts = 6
    n_isolated_verts = isolated_verts.shape[0]
    face_ndim = 1

    # Equivalent of gathering vertex scalars onto other vertices:
    data = torch.randn([n_verts])
    expected = torch.zeros([n_isolated_verts])
    for f in range(n_isolated_verts):
        expected[f] = data[isolated_verts[f]]
    result = face_index(data, isolated_verts, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    # result = face_index(data, isolated_verts)
    # torch.testing.assert_close(result, expected)

    # Equivalent of gathering batched vertex scalars onto other vertices:
    data = torch.randn([8, n_verts])
    expected = torch.zeros([8, n_isolated_verts])
    for b in range(8):
        for f in range(n_isolated_verts):
            expected[b, f] = data[b, isolated_verts[f]]
    result = face_index(data, isolated_verts.expand(8, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Following cannot work because we do not know if a function is 8 faces
    # with `n_verts` vertices each, or a batch of vertices with 8 batch elements.
    # result = face_index(data, isolated_verts.expand(8, -1))
    # torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex vectors onto other vertices:
    data = torch.randn([n_verts, 3])
    expected = torch.zeros([n_isolated_verts, 3])
    for f in range(n_isolated_verts):
        for d in range(3):
            expected[f, d] = data[isolated_verts[f], d]
    result = face_index(data, isolated_verts, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)
    # result = face_index(data, isolated_verts)
    # torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex vectors onto other vertices:
    data = torch.randn([8, n_verts, 3])
    expected = torch.zeros([8, n_isolated_verts, 3])
    for b in range(8):
        for f in range(n_isolated_verts):
            for d in range(3):
                expected[b, f, d] = data[b, isolated_verts[f], d]
    result = face_index(data, isolated_verts.expand(8, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Again, following cannot work because we do not know if a function is 8 faces
    # with `n_verts` vertices each, or a batch of vertices with 8 batch elements.
    # result = face_index(data, isolated_verts.expand(8, -1))
    # torch.testing.assert_close(result, expected)


def test_gather_single_vert():
    # A one-element index is the case where `F` itself has size 1. There is no
    # subface dimension to remove here, so `squeeze` must leave `F` alone.
    single_vert = torch.tensor([2])
    n_verts = 6
    face_ndim = 1

    # Equivalent of gathering vertex scalars onto a single vertex:
    data = torch.randn([n_verts])
    expected = torch.zeros([1])
    expected[0] = data[single_vert[0]]
    result = face_index(data, single_vert, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering batched vertex scalars onto a single vertex:
    data = torch.randn([8, n_verts])
    expected = torch.zeros([8, 1])
    for b in range(8):
        expected[b, 0] = data[b, single_vert[0]]
    result = face_index(data, single_vert.expand(8, -1), face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # Equivalent of gathering vertex vectors onto a single vertex:
    data = torch.randn([n_verts, 3])
    expected = torch.zeros([1, 3])
    for d in range(3):
        expected[0, d] = data[single_vert[0], d]
    result = face_index(data, single_vert, face_ndim=face_ndim)
    torch.testing.assert_close(result, expected)

    # A genuine `[F, FS]` index with `FS` == 1 does still get squeezed, though:
    result = face_index(data, single_vert[:, None])
    torch.testing.assert_close(result, expected)
    result = face_index(data, single_vert[:, None], squeeze=False)
    torch.testing.assert_close(result, expected[:, None, :])


def test_gather_face_ndim_errors():
    isolated_verts = torch.tensor([0, 2, 4])
    n_verts = 6
    data = torch.randn([n_verts])

    # A 1D list of vertices has to be declared as such, we cannot guess it:
    with pytest.raises(ValueError):
        face_index(data, isolated_verts)

    # `face_ndim` counts face dimensions, so it cannot exceed `faces.ndim`:
    with pytest.raises(ValueError):
        face_index(data, isolated_verts[:, None], face_ndim=3)

    # Batch dimensions that disagree are reported rather than broadcast:
    with pytest.raises(ValueError):
        face_index(torch.randn([4, n_verts]), isolated_verts.expand(8, -1), face_ndim=1)


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

    # Equivalent of averaging face vectors on vertices:
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
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
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
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
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
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
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
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
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
        data, triangles, n_verts, reduce="sum", data_ndim=data_ndim
    )
    torch.testing.assert_close(result, expected)


def test_scatter_greedy_batched(triangles: torch.Tensor):
    # With `batch_ndim` derived from `face_ndim`, the greedy `data_ndim` default
    # accounts for the batch.
    batch_size = 16
    n_verts = 6
    n_tris = triangles.shape[0]
    triangles = triangles[None, :, :].expand(batch_size, -1, -1)

    # Equivalent of averaging face normals on vertices, without saying data_ndim:
    data = torch.randn([batch_size, n_tris, 3])
    expected = torch.zeros([batch_size, n_verts, 3])
    for b in range(batch_size):
        for f in range(n_tris):
            for c in range(3):  # corner
                for d in range(3):
                    expected[b, triangles[b, f, c], d] += data[b, f, d]
    result = reduce_on_subface(data, triangles, n_verts, reduce="sum")
    torch.testing.assert_close(result, expected)


def test_scatter_face_ndim_errors(triangles: torch.Tensor):
    n_verts = 6
    n_tris = triangles.shape[0]

    # A 1D list of vertices has to be declared as such, we cannot guess it:
    with pytest.raises(ValueError):
        reduce_on_subface(torch.randn([n_tris]), triangles[:, 0], n_verts, "sum")

    # Batch dimensions that disagree are reported rather than broadcast:
    with pytest.raises(ValueError):
        reduce_on_subface(
            torch.randn([4, n_tris]),
            triangles.expand(8, -1, -1),
            n_verts,
            "sum",
            data_ndim=0,
        )

    # `data_ndim` has to leave at least one face dimension in the data:
    with pytest.raises(ValueError):
        reduce_on_subface(
            torch.randn([n_tris, 3]), triangles, n_verts, "sum", data_ndim=2
        )

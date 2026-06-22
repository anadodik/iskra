import pytest
import torch

import iskra.sparse as sp
from tests.template import assert_equal


def test_isect_indices_basic():
    a = torch.tensor([[0, 1, 2], [3, 4, 5]])
    b = torch.tensor([[1, 2, 3], [4, 5, 6]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert torch.equal(a_mask, torch.tensor([False, True, True]))
    assert torch.equal(b_mask, torch.tensor([True, True, False]))


def test_isect_indices_empty():
    a = torch.tensor([[0, 1], [2, 3]])
    b = torch.tensor([[4, 5], [6, 7]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert not a_mask.any()
    assert not b_mask.any()


def test_isect_indices_order_independent():
    a = torch.tensor([[2, 1, 0], [5, 4, 3]])
    b = torch.tensor([[1, 2], [4, 5]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert torch.equal(a_mask, torch.tensor([True, True, False]))
    assert torch.equal(b_mask, torch.tensor([True, True]))


def test_mul_sparse_sparse_basic():
    a_idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    a_val = torch.tensor([2.0, 3.0, 4.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 6))

    b_idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b_val = torch.tensor([5.0, 6.0, 7.0])
    b = sp.coo_tensor(b_idx, b_val, size=(4, 6))

    out = sp.mul_sparse_sparse(a, b)

    exp_idx = torch.tensor([[1, 2], [4, 5]])
    exp_val = torch.tensor([3.0 * 5.0, 4.0 * 6.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_mul_sparse_sparse_shape_mismatch():
    a = sp.coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(2, 2))
    b = sp.coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(3, 3))
    with pytest.raises(ValueError):
        sp.mul_sparse_sparse(a, b)


def test_mul_sparse_sparse_partial_intersection():
    a_idx = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    a_val = torch.tensor([2.0, 3.0, 4.0, 5.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4))

    b_idx = torch.tensor([[1, 3], [1, 3]])
    b_val = torch.tensor([10.0, 20.0])
    b = sp.coo_tensor(b_idx, b_val, size=(4, 4))

    out = sp.mul_sparse_sparse(a, b).coalesce()

    exp_idx = torch.tensor([[1, 3], [1, 3]])
    exp_val = torch.tensor([3.0 * 10.0, 5.0 * 20.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_cat_sparse():
    a = sp.coo_tensor(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([1.0, 2.0]), size=(3, 2)
    )
    b = sp.coo_tensor(
        torch.tensor([[0, 2], [0, 1]]), torch.tensor([3.0, 4.0]), size=(3, 2)
    )

    out = sp.cat([a, b], dim=1).coalesce()
    exp = sp.coo_tensor(
        torch.tensor([[0, 1, 0, 2], [0, 1, 2, 3]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(3, 4),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape

    out = sp.cat([a, b], dim=0).coalesce()
    exp = sp.coo_tensor(
        torch.tensor([[0, 1, 3, 5], [0, 1, 0, 1]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(6, 2),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape


def test_indexing():
    a_idx = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 0, 2, 3]])
    a_val = torch.tensor([2.0, 3.0, 6.0, 4.0, 5.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4))
    a_dense = a.to_dense()

    # We check for a few test that slicing matches what we'd expect manually,
    # to hopefully also catch if `SparseTensor.to_dense` goes awry.
    assert_equal(a[2, 2].to_dense(), torch.tensor(4.0))
    assert_equal(a[0, 1:2].to_dense(), torch.tensor([0.0]))

    assert_equal(a[2, 2].to_dense(), a_dense[2, 2])
    assert_equal(a[3].to_dense(), a_dense[3])
    assert_equal(a[0, 1:2].to_dense(), a_dense[0, 1:2])
    assert_equal(a[0, 1:3].to_dense(), a_dense[0, 1:3])
    assert_equal(a[0:2, 1:3].to_dense(), a_dense[0:2, 1:3])

    assert_equal(a[0, torch.tensor([0])].to_dense(), a_dense[0, torch.tensor([0])])
    assert_equal(a[0, torch.arange(1, 2)].to_dense(), a_dense[0, torch.arange(1, 2)])
    assert_equal(a[0, torch.arange(1, 3)].to_dense(), a_dense[0, torch.arange(1, 3)])
    assert_equal(
        a[0:2, torch.arange(1, 3)].to_dense(), a_dense[0:2, torch.arange(1, 3)]
    )

    mask = torch.tensor([True, False, True, False])
    assert_equal(a[1, mask].to_dense(), a_dense[1, mask])
    assert_equal(a[2:, mask].to_dense(), a_dense[2:, mask])

    assert_equal(a[0:2, None, 1:3].to_dense(), a_dense[0:2, None, 1:3])
    assert_equal(a[3, None, 2].to_dense(), a_dense[3, None, 2])
    assert_equal(a[None, 3, None].to_dense(), a_dense[None, 3, None])
    assert_equal(a[None, mask, :].to_dense(), a_dense[None, mask, :])

    # Following are different!
    print(a[(torch.arange(0, 2)), torch.arange(1, 3)].to_dense())
    print(a.to_dense()[torch.arange(0, 2), torch.arange(1, 3)])

    print(a[(0, 1), (1, 2)].to_dense())
    print(a.to_dense()[(0, 1), (1, 2)])

    print(a[mask, mask].to_dense())
    print(a_dense[mask, mask])

    print(a[(True, False, True, False), (True, False, True, False)].to_dense())
    print(a_dense[(True, False, True, False), (True, False, True, False)])

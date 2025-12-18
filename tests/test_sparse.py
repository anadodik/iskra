import pytest
import torch

import iskra.sparse as sp


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
    a = torch.sparse_coo_tensor(a_idx, a_val, size=(4, 6))

    b_idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b_val = torch.tensor([5.0, 6.0, 7.0])
    b = torch.sparse_coo_tensor(b_idx, b_val, size=(4, 6))

    out = sp.mul_sparse_sparse(a, b)

    exp_idx = torch.tensor([[1, 2], [4, 5]])
    exp_val = torch.tensor([3.0 * 5.0, 4.0 * 6.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_mul_sparse_sparse_shape_mismatch():
    a = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(2, 2)
    )
    b = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(3, 3)
    )
    with pytest.raises(ValueError):
        sp.mul_sparse_sparse(a, b)


def test_mul_sparse_sparse_partial_intersection():
    a_idx = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    a_val = torch.tensor([2.0, 3.0, 4.0, 5.0])
    a = torch.sparse_coo_tensor(a_idx, a_val, size=(4, 4))

    b_idx = torch.tensor([[1, 3], [1, 3]])
    b_val = torch.tensor([10.0, 20.0])
    b = torch.sparse_coo_tensor(b_idx, b_val, size=(4, 4))

    out = sp.mul_sparse_sparse(a, b).coalesce()

    exp_idx = torch.tensor([[1, 3], [1, 3]])
    exp_val = torch.tensor([3.0 * 10.0, 5.0 * 20.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_cat_sparse():
    a = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([1.0, 2.0]), size=(3, 2)
    )
    b = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [0, 1]]), torch.tensor([3.0, 4.0]), size=(3, 2)
    )

    out = sp.cat([a, b], dim=1).coalesce()
    exp = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 0, 2], [0, 1, 2, 3]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(3, 4),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape

    out = sp.cat([a, b], dim=0).coalesce()
    exp = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 3, 5], [0, 1, 0, 1]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(6, 2),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape

# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import numpy as np
import scipy.sparse
import torch


def eye(n: int, device: str | torch.device = "cpu"):
    idx = torch.arange(n, device=device)
    ij = torch.stack(2 * [idx])
    values = torch.ones([n], device=device)
    return torch.sparse_coo_tensor(ij, values, size=[n, n]).coalesce()


def diag(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    assert values.ndim == 1
    n = values.shape[0]
    idx = torch.arange(n, device=values.device)
    ii = torch.stack(2 * [idx])
    return torch.sparse_coo_tensor(ii, values, size=[n, n]).coalesce()


def scipy_to_torch(
    x: scipy.sparse.sparray, device: torch.device | str = "cpu"
) -> torch.Tensor:
    x_coo = x.tocoo()
    row = torch.tensor(x_coo.row, device=device)
    col = torch.tensor(x_coo.col, device=device)
    data = torch.tensor(x_coo.data, device=device)
    x_torch = torch.sparse_coo_tensor(torch.stack([row, col]), data, size=x.shape)
    x_torch = x_torch.to_sparse_csr()
    return x_torch


def torch_to_scipy(x: torch.Tensor) -> torch.Tensor:
    x = x.to_sparse_coo().coalesce()
    data = x.values().cpu().numpy()
    idcs = x.indices().cpu().numpy().astype(np.int64)
    x_scipy = scipy.sparse.coo_array((data, (idcs[0], idcs[1])), shape=x.shape)
    return x_scipy

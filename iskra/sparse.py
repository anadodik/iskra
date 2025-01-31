# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import torch


def eye(n: int, device: str | torch.device = "cpu"):
    idx = torch.arange(n, device=device)
    ij = torch.stack(2 * [idx])
    values = torch.ones([n], device=device)
    return torch.sparse_coo_tensor(ij, values, size=[n, n])

def diag(values: torch.Tensor) -> torch.Tensor:
    assert values.ndim == 1
    n = values.shape[0]
    idx = torch.arange(n, device=values.device)
    ii = torch.stack(2 * [idx])
    return torch.sparse_coo_tensor(ii, values, size=[n, n])
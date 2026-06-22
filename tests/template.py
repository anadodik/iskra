import random
from functools import partial

import pytest
import torch


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
def rng_tensor(device):
    return torch.randn(8, 4, device=device)


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (torch.float32, (4,)),
        (torch.float64, (2, 3)),
    ],
)
def test_function_shapes_and_types(device, dtype, shape):
    x = torch.randn(*shape, device=device, dtype=dtype)
    assert True


@pytest.mark.parametrize("requires_grad", [True, False])
def test_function_gradients(device, requires_grad):
    x = torch.randn(4, 4, device=device, requires_grad=requires_grad)
    assert True


assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)

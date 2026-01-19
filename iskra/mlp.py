# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.


from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Literal

# import tinycudann as tcnn
import torch

from iskra.logging import getLogger

LOGGER = getLogger(__name__)


class ScaleEmbedding(torch.nn.Module):
    def __init__(self, dim: int, scale: float = 2, offset: float = -1.0):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.dim = dim
        self.n_output_features = self.dim
        if TYPE_CHECKING:
            self.frequencies: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale) - self.offset


class Squareplus(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.sqrt(x**2 + 4)) / 2


_ACTIVATIONS: dict[str, Callable[[], torch.nn.Module]] = {
    "Softplus": lambda: torch.nn.Softplus(beta=4),
    "Squareplus": Squareplus,
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "SiLU": torch.nn.SiLU,
    "GELU": torch.nn.GELU,
}


class PyTorchMLP(torch.nn.Module):
    def __init__(  # type: ignore
        self,
        n_input_dims: int,
        n_output_dims: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "LeakyReLU",
        bias: bool = False,
        device: str | torch.Tensor = "cuda",
    ) -> None:
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.activation = activation
        activation_cls = _ACTIVATIONS[self.activation]

        assert n_hidden_layers >= 0
        hidden_layers: list[torch.nn.Module] = []
        for _ in range(n_hidden_layers):
            hidden_layers.append(torch.nn.Linear(n_neurons, n_neurons, bias=bias))
            hidden_layers.append(activation_cls())

        input_layer = torch.nn.Linear(self.n_input_dims, n_neurons, bias=bias)
        output_layer = torch.nn.Linear(n_neurons, self.n_output_dims, bias=bias)
        self.network = torch.nn.Sequential(
            input_layer,
            activation_cls(),
            *hidden_layers,
            output_layer,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # type: ignore


@dataclass(kw_only=True)
class NetworkConfig:
    otype: Literal["PyTorchMLP", "FullyFusedMLP", "CutlassMLP"]
    activation: Literal["Softplus", "Squareplus", "ReLU", "LeakyReLU", "SiLU", "GELU"]
    encoding_config: dict[str, str | int] | None
    n_neurons: int
    n_hidden_layers: int


class MLP(torch.nn.Module):
    def __init__(  # type: ignore
        self,
        n_input_dims: int,
        n_output_dims: int,
        network_config: NetworkConfig,
    ) -> None:
        super().__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        self.network: torch.nn.Module
        if network_config.otype == "PyTorchMLP":
            if network_config.encoding_config is not None:
                encoding = tcnn.Encoding(
                    n_input_dims=self.n_input_dims,
                    encoding_config=network_config.encoding_config,
                )
                mlp = PyTorchMLP(
                    encoding.n_output_dims,
                    self.n_output_dims,
                    n_neurons=network_config.n_neurons,
                    n_hidden_layers=network_config.n_hidden_layers,
                    activation=network_config.activation,
                )
                self.network = torch.nn.Sequential(encoding, mlp).to("cuda")
            else:
                self.network = PyTorchMLP(
                    self.n_input_dims,
                    self.n_output_dims,
                    n_neurons=network_config.n_neurons,
                    n_hidden_layers=network_config.n_hidden_layers,
                    activation=network_config.activation,
                ).to("cuda")
        else:
            network_config_dict = asdict(network_config)
            encoding_config_dict = network_config_dict.pop("encoding_config")
            self.network = tcnn.NetworkWithInputEncoding(
                n_input_dims,
                n_output_dims,
                encoding_config_dict,
                network_config_dict,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # type: ignore

# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from typing import TYPE_CHECKING, Iterator

import torch


class BBox(torch.nn.Module):
    def __init__(self, min: torch.Tensor, max: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("min", min)
        self.register_buffer("max", max)
        if TYPE_CHECKING:
            self.min: torch.Tensor
            self.max: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([self.min, self.max])

    def __repr__(self) -> str:
        return f"BBox(min={self.min}, max={self.max})"

    @property
    def extent(self) -> torch.Tensor:
        return self.max - self.min

    @property
    def center(self) -> torch.Tensor:
        return self.min + 0.5 * (self.max - self.min)

    @classmethod
    def compute(cls, vertices: torch.Tensor) -> "BBox":
        bbox_min = torch.min(vertices, dim=0).values
        bbox_max = torch.max(vertices, dim=0).values
        return cls(bbox_min, bbox_max)

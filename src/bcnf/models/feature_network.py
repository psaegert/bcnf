from typing import Type

import torch
from torch import nn


class FeatureNetwork(nn.Module):
    def __init__(self) -> None:
        super(FeatureNetwork, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FullyConnectedFeatureNetwork(FeatureNetwork):
    def __init__(self, sizes: list[int], activation: Type[nn.Module] = nn.GELU, dropout: float = 0.0, batch_norm: bool = False) -> None:
        super(FullyConnectedFeatureNetwork, self).__init__()

        self.nn = nn.Sequential()

        if len(sizes) < 2:
            # No transformations from one layer to another, use identity (0 layers)
            self.nn.append(nn.Identity())
        else:
            for i in range(len(sizes) - 2):
                self.nn.append(nn.Linear(sizes[i], sizes[i + 1]))
                if batch_norm:
                    self.nn.append(nn.BatchNorm1d(sizes[i + 1]))
                self.nn.append(activation())
                if dropout > 0.0:
                    self.nn.append(nn.Dropout(dropout))

            self.nn.append(nn.Linear(sizes[-2], sizes[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn.forward(x)

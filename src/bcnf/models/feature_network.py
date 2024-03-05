from typing import Type

import torch
from torch import nn


class FeatureNetwork(nn.Module):
    def __init__(self) -> None:
        super(FeatureNetwork, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FullyConnectedFeatureNetwork(FeatureNetwork):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, activation: Type[nn.Module] = nn.GELU, dropout: float = 0.0, batch_norm: bool = False) -> None:
        super(FullyConnectedFeatureNetwork, self).__init__()

        self.nn = nn.Sequential()
        layer_transforms = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_transforms) - 2):
            self.nn.append(nn.Linear(layer_transforms[i], layer_transforms[i + 1]))
            if batch_norm:
                self.nn.append(nn.BatchNorm1d(layer_transforms[i + 1]))
            self.nn.append(activation())
            if dropout > 0.0:
                self.nn.append(nn.Dropout(dropout))

        self.nn.append(nn.Linear(layer_transforms[-2], layer_transforms[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn.forward(x)

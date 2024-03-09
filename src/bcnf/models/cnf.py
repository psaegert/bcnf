from abc import abstractmethod
from typing import Type

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from bcnf.models.feature_network import FeatureNetwork


class InvertibleLayer(nn.Module):
    log_det_J: float | torch.Tensor | None
    n_conditions: int

    @abstractmethod
    def forward(self, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        pass


class ConditionalInvertibleLayer(nn.Module):
    log_det_J: float | torch.Tensor | None
    n_conditions: int

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class ConditionalNestedNeuralNetwork(nn.Module):
    def __init__(self, sizes: list[int], n_conditions: int, activation: Type[nn.Module] = nn.GELU, dropout: float = 0.0, device: str = "cpu") -> None:
        super(ConditionalNestedNeuralNetwork, self).__init__()

        self.n_conditions = n_conditions

        self.nn = nn.Sequential()

        if len(sizes) < 2:
            # No transformations from one layer to another, use identity (0 layers)
            self.nn.append(nn.Identity())
        else:
            # Add the conditions to the input
            sizes[0] += n_conditions

            # Account for splitting the output into t and s
            sizes[-1] *= 2

            for i in range(len(sizes) - 2):
                self.nn.append(nn.Linear(sizes[i], sizes[i + 1]))
                self.nn.append(activation())
                if dropout > 0.0:
                    self.nn.append(nn.Dropout(dropout))

            self.nn.append(nn.Linear(sizes[-2], sizes[-1]))

    def to(self, device: str) -> "ConditionalNestedNeuralNetwork":  # type: ignore
        super().to(device)
        self.device = device
        self.nn.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_conditions > 0:
            # Concatenate the input with the condition
            x = torch.cat([x, y], dim=1)

        # Get the translation coefficients t and the scale coefficients s from the neural network
        t, s = self.nn.forward(x).chunk(2, dim=1)

        # Return the coefficients
        return t, torch.tanh(s)


class ConditionalAffineCouplingLayer(ConditionalInvertibleLayer):
    def __init__(self, input_size: int, nested_sizes: list[int], n_conditions: int, dropout: float = 0.0, device: str = "cpu") -> None:
        super(ConditionalAffineCouplingLayer, self).__init__()

        self.n_conditions = n_conditions
        self.log_det_J: torch.Tensor = torch.zeros(1).to(device)

        # Create the nested neural network
        self.nn = ConditionalNestedNeuralNetwork(
            sizes=[int(np.ceil(input_size / 2))] + nested_sizes + [int(np.floor(input_size / 2))],
            n_conditions=n_conditions,
            dropout=dropout,
            device=device)

    def to(self, device: str) -> "ConditionalAffineCouplingLayer":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        self.nn.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Split the input into two halves
        x_a, x_b = x.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn.forward(x_a, y)

        # Apply the transformation
        z_a = x_a  # skip connection
        z_b = torch.exp(log_s) * x_b + t  # affine transformation

        # Calculate the log determinant of the Jacobian
        if log_det_J:
            self.log_det_J = log_s.sum(dim=1)

        # Return the output
        return torch.cat([z_a, z_b], dim=1)

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split the input into two halves
        z_a, z_b = z.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn.forward(z_a, y)

        # Apply the inverse transformation
        x_a = z_a
        x_b = (z_b - t) * torch.exp(- log_s)

        # Return the output
        return torch.cat([x_a, x_b], dim=1)


class OrthonormalTransformation(ConditionalInvertibleLayer):
    def __init__(self, input_size: int) -> None:
        super(OrthonormalTransformation, self).__init__()

        self.log_det_J: float = 0

        # Create the random orthonormal matrix via QR decomposition
        self.orthonormal_matrix: torch.Tensor = torch.linalg.qr(torch.randn(input_size, input_size))[0]
        self.orthonormal_matrix.requires_grad = False

    def to(self, device: str) -> "OrthonormalTransformation":  # type: ignore
        super().to(device)
        self.device = device
        self.orthonormal_matrix = self.orthonormal_matrix.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the transformation
        return x @ self.orthonormal_matrix

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply the inverse transformation
        return z @ self.orthonormal_matrix.T


class ActNorm(InvertibleLayer):
    def __init__(self, size: int) -> None:
        super(ActNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        z = self.scale * x + self.bias
        self.log_det_J = torch.sum(torch.log(torch.abs(self.scale)), dim=-1)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.bias) / self.scale


class CondRealNVP(ConditionalInvertibleLayer):
    def __init__(self, size: int, nested_sizes: list[int], n_blocks: int, n_conditions: int, feature_network: FeatureNetwork | None, dropout: float = 0.0, act_norm: bool = False, device: str = "cpu"):
        super(CondRealNVP, self).__init__()

        if n_conditions == 0 or feature_network is None:
            self.h = nn.Identity()
        else:
            self.h = feature_network

        self.size = size
        self.nested_sizes = nested_sizes
        self.n_blocks = n_blocks
        self.n_conditions = n_conditions
        self.device = device
        self.dropout = dropout
        self.log_det_J: torch.Tensor = torch.zeros(1).to(self.device)

        # Create the network
        self.layers = nn.ModuleList()
        for _ in range(self.n_blocks - 1):
            if act_norm:
                self.layers.append(ActNorm(self.size))
            self.layers.append(ConditionalAffineCouplingLayer(self.size, self.nested_sizes, self.n_conditions, dropout=self.dropout, device=self.device))
            self.layers.append(OrthonormalTransformation(self.size))

        # Add the final affine coupling layer
        self.layers.append(ConditionalAffineCouplingLayer(self.size, self.nested_sizes, self.n_conditions, dropout=self.dropout, device=self.device))

    def to(self, device: str) -> "CondRealNVP":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        for layer in self.layers:
            layer.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network

        if log_det_J:
            self.log_det_J = torch.zeros(x.shape[0]).to(self.device)

            for layer in self.layers:
                if isinstance(layer, ConditionalInvertibleLayer):
                    x = layer(x, y, log_det_J)
                elif isinstance(layer, InvertibleLayer):
                    x = layer(x, log_det_J)
                else:
                    raise ValueError(f"Layer must be an instance of ConditionalInvertibleLayer or InvertibleLayer, but got {type(layer)}")
                self.log_det_J += layer.log_det_J

            return x

        for layer in self.layers:
            x = layer(x, y, log_det_J)

        return x

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network in reverse
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalInvertibleLayer):
                z = layer.inverse(z, y)
            elif isinstance(layer, InvertibleLayer):
                z = layer.inverse(z)
            else:
                raise ValueError(f"Layer must be an instance of ConditionalInvertibleLayer or InvertibleLayer, but got {type(layer)}")

        return z

    def sample(self, n_samples: int, y: torch.Tensor, sigma: float = 1, outer: bool = False, batch_size: int = 100, output_device: str = "cpu", verbose: bool = False) -> torch.Tensor:
        m_batch_sizes = [batch_size] * (n_samples // batch_size) + [n_samples % batch_size]
        y_hat_list = []

        with torch.no_grad():
            for m in tqdm(m_batch_sizes, desc="Sampling", disable=not verbose):
                y_hat_list.append(self._sample(m, y=y, outer=True, sigma=sigma, verbose=verbose).to(output_device))

        y_hat = torch.cat(y_hat_list, dim=0)

        return y_hat

    def _sample(self, n_samples: int, y: torch.Tensor, sigma: float = 1, outer: bool = False, verbose: bool = False) -> torch.Tensor:
        """
        Sample from the model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        y : torch.Tensor
            The conditions used for sampling.
            If 1st order tensor and len(y) == n_conditions, the same conditions are used for all samples.
            If 2nd order tensor, y.shape must be (n_samples, n_conditions), and each row is used as the conditions for each sample.
        sigma : float
            The standard deviation of the normal distribution to sample from.
        outer : bool
            If True, the conditions are broadcasted to match the shape of the samples.
            If False, the conditions are matched to the shape of the samples.
        verbose : bool
            If True, print debug information.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """

        y = y.to(self.device)

        if y.ndim == 1:
            if verbose:
                print('Broadcasting')
            # if len(y) != n_input_conditions:
            #     raise ValueError(f"y must have length {n_input_conditions}, but got len(y) = {len(y)}")

            # Generate n_samples for each condition in y
            z = sigma * torch.randn(n_samples, self.size).to(self.device)
            y = y.repeat(n_samples, 1)

            # Apply the inverse network
            return self.inverse(z, y).view(n_samples, self.size)
        elif y.ndim == 2:
            if outer:
                if verbose:
                    print('Outer')
                # if y.shape[1] != n_input_conditions:
                #     raise ValueError(f"y must have shape (n_samples_per_condition, {n_input_conditions}), but got y.shape = {y.shape}")

                n_samples_per_condition = y.shape[0]

                # Generate n_samples for each condition in y
                z = sigma * torch.randn(n_samples * n_samples_per_condition, self.size).to(self.device)
                y = y.repeat(n_samples, 1)

                # Apply the inverse network
                return self.inverse(z, y).view(n_samples, n_samples_per_condition, self.size)
            else:
                if verbose:
                    print('Matching')
                # if y.shape[0] != n_samples or y.shape[1] != n_input_conditions:
                #     raise ValueError(f"y must have shape (n_samples, {n_input_conditions}), but got y.shape = {y.shape}")

                z = sigma * torch.randn(n_samples, self.size).to(self.device)

                return self.inverse(z, y).view(n_samples, self.size)
        else:
            raise ValueError(f"y must be a 1st or 2nd order tensor, but got y.shape = {y.shape}")

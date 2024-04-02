from abc import abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from bcnf.factories import FeatureNetworkFactory, LayerFactory
from bcnf.models.feature_network import FeatureNetwork, FeatureNetworkStack
from bcnf.utils import ParameterIndexMapping


class InvertibleLayer(nn.Module):
    log_det_J: float | torch.Tensor | None
    n_conditions: int

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def forward(self, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        pass


class ConditionalInvertibleLayer(nn.Module):
    log_det_J: float | torch.Tensor | None
    n_conditions: int
    device: str

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def forward(self, y: torch.Tensor, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass


class ConditionalNestedNeuralNetwork(nn.Module):
    def __init__(
            self,
            sizes: list[int],
            n_conditions: int,
            n_output_parameters: int,
            layer: str = "Linear",
            layer_kwargs: dict[str, Any] | None = None,
            activation: str = "GELU",
            activation_kwargs: dict[str, Any] | None = None,
            dropout: float = 0.0,
            device: str = "cpu") -> None:
        super(ConditionalNestedNeuralNetwork, self).__init__()

        self.n_conditions = n_conditions
        self.n_output_parameters = n_output_parameters
        self.device = device

        self.nn = nn.Sequential()

        if len(sizes) < 2:
            # No transformations from one layer to another, use identity (0 layers)
            self.nn.append(nn.Identity())
        else:
            # Add the conditions to the input
            sizes[0] += n_conditions

            # Account for splitting the output into t and s
            sizes[-1] *= self.n_output_parameters

            for i in range(len(sizes) - 2):
                self.nn.append(LayerFactory.get_layer(layer, sizes[i], sizes[i + 1], **layer_kwargs or {}))
                self.nn.append(LayerFactory.get_layer(activation, **activation_kwargs or {}))
                if dropout > 0.0:
                    self.nn.append(nn.Dropout(dropout))

            self.nn.append(LayerFactory.get_layer(layer, sizes[-2], sizes[-1], **layer_kwargs or {}))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def to(self, device: str) -> "ConditionalNestedNeuralNetwork":  # type: ignore
        super().to(device)
        self.device = device
        self.nn.to(device)

        return self

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_conditions > 0:
            # Concatenate the input with the condition
            y = torch.cat([y, h], dim=1)

        # Get the translation coefficients t and the scale coefficients s from the neural network
        t, s = self.nn.forward(y).chunk(2, dim=1)

        # Return the coefficients
        return t, torch.tanh(s)


class ConditionalAffineCouplingLayer(ConditionalInvertibleLayer):
    def __init__(
            self,
            input_size: int,
            nested_sizes: list[int],
            n_conditions: int,
            layer: str = "Linear",
            layer_kwargs: dict[str, Any] | None = None,
            activation: str = "GELU",
            activation_kwargs: dict[str, Any] | None = None,
            dropout: float = 0.0,
            device: str = "cpu",
            two_way: bool = False) -> None:
        super(ConditionalAffineCouplingLayer, self).__init__()

        self.n_conditions = n_conditions
        self.log_det_J: torch.Tensor = torch.zeros(1).to(device)
        self.device = device
        self.two_way = two_way

        # Create the nested neural network
        self.nn_a = ConditionalNestedNeuralNetwork(
            sizes=[int(np.ceil(input_size / 2))] + nested_sizes + [int(np.floor(input_size / 2))],
            n_conditions=n_conditions,
            n_output_parameters=2,  # scale and translation
            layer=layer,
            layer_kwargs=layer_kwargs,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=dropout,
            device=device)

        # Create the nested neural network
        if two_way:
            self.nn_b = ConditionalNestedNeuralNetwork(
                sizes=[int(np.floor(input_size / 2))] + nested_sizes + [int(np.ceil(input_size / 2))],
                n_conditions=n_conditions,
                n_output_parameters=2,  # scale and translation
                layer=layer,
                layer_kwargs=layer_kwargs,
                activation=activation,
                activation_kwargs=activation_kwargs,
                dropout=dropout,
                device=device)

    def to(self, device: str) -> "ConditionalAffineCouplingLayer":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        self.nn_a.to(device)
        if self.two_way:
            self.nn_b.to(device)

        return self

    def forward(self, y: torch.Tensor, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Check if x needs reshaping (i.e. if it is a 1D tensor)
        if y.dim() == 1:
            # Reshape x to have a batch dimension
            y = y.unsqueeze(0)
        if x.dim() == 1:
            # Reshape x to have a batch dimension
            x = x.unsqueeze(0)

        # Split the input into two halves
        y_a, y_b = y.chunk(2, dim=-1)

        # One way
        t_a, log_s_a = self.nn_a.forward(y_a, x)
        z_b = torch.exp(log_s_a) * y_b + t_a

        # Other way
        if self.two_way:
            t_b, log_s_b = self.nn_b.forward(z_b, x)  # The other way depends on the transformed z_b
            z_a = torch.exp(log_s_b) * y_a + t_b
        else:
            z_a = y_a

        # Calculate the log determinant of the Jacobian
        if log_det_J:
            self.log_det_J = log_s_a.sum(dim=-1)

            if self.two_way:
                self.log_det_J += log_s_b.sum(dim=-1)

        # Return the output
        return torch.cat([z_a, z_b], dim=-1)

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split the input into two halves
        z_a, z_b = z.chunk(2, dim=-1)

        # One way
        t_a, log_s_a = self.nn_a.forward(z_a, y)
        y_b = (z_b - t_a) * torch.exp(- log_s_a)

        if self.two_way:
            t_b, log_s_b = self.nn_b.forward(y_b, y)  # The other way depends on the transformed y_b
            y_a = (z_a - t_b) * torch.exp(- log_s_b)
        else:
            y_a = z_a

        # Return the output
        return torch.cat([y_a, y_b], dim=-1)


class ConditionalRQSplineCouplingLayer(ConditionalInvertibleLayer):
    def __init__(self, input_size: int, nested_sizes: list[int], n_conditions: int, dropout: float = 0.0, device: str = "cpu", two_way: bool = False) -> None:
        super(ConditionalRQSplineCouplingLayer, self).__init__()

        self.n_conditions = n_conditions
        self.log_det_J: torch.Tensor = torch.zeros(1).to(device)
        self.device = device
        self.two_way = two_way

        # Create the nested neural network
        self.nn_a = ConditionalNestedNeuralNetwork(
            sizes=[int(np.ceil(input_size / 2))] + nested_sizes + [int(np.floor(input_size / 2))],
            n_conditions=n_conditions,
            n_output_parameters=6,  # (az^2 + bz + c) / (dz^2 + ez + g)
            dropout=dropout,
            device=device)

        # Create the nested neural network
        if two_way:
            self.nn_b = ConditionalNestedNeuralNetwork(
                sizes=[int(np.floor(input_size / 2))] + nested_sizes + [int(np.ceil(input_size / 2))],
                n_conditions=n_conditions,
                n_output_parameters=6,  # (az^2 + bz + c) / (dz^2 + ez + g)
                dropout=dropout,
                device=device)

    def to(self, device: str) -> "ConditionalRQSplineCouplingLayer":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        self.nn_a.to(device)
        if self.two_way:
            self.nn_b.to(device)

        return self

    def forward(self, y: torch.Tensor, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Check if x needs reshaping (i.e. if it is a 1D tensor)
        if y.dim() == 1:
            # Reshape x to have a batch dimension
            y = y.unsqueeze(0)
        if x.dim() == 1:
            # Reshape x to have a batch dimension
            x = x.unsqueeze(0)

        # SPLIT
        y_a, y_b = y.chunk(2, dim=-1)

        # GET SCALE AND TRANSLATION
        t_a, log_s_a = self.nn_a.forward(y_a, x)

        if self.two_way:
            t_b, log_s_b = self.nn_b.forward(y_b, x)

        # TRANSFORM
        if self.two_way:
            # affine transformation
            z_a = torch.exp(log_s_b) * y_a + t_b
        else:
            # skip connection
            z_a = y_a

        z_b = torch.exp(log_s_a) * y_b + t_a

        # JACOBIAN
        if log_det_J:
            self.log_det_J = log_s_a.sum(dim=-1)

            if self.two_way:
                self.log_det_J += log_s_b.sum(dim=-1)

        # Return the output
        return torch.cat([z_a, z_b], dim=-1)

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split the input into two halves
        z_a, z_b = z.chunk(2, dim=-1)

        # Get the coefficients from the neural network
        t_a, log_s_a = self.nn_a.forward(z_a, y)
        if self.two_way:
            t_b, log_s_b = self.nn_b.forward(z_b, y)

        # Apply the inverse transformation
        if self.two_way:
            # affine transformation
            y_a = (z_a - t_b) * torch.exp(- log_s_b)
        else:
            # skip connection
            y_a = z_a
        y_b = (z_b - t_a) * torch.exp(- log_s_a)

        # Return the output
        return torch.cat([y_a, y_b], dim=-1)


class OrthonormalTransformation(ConditionalInvertibleLayer):
    def __init__(self, input_size: int, random_state: int | None = None) -> None:
        super(OrthonormalTransformation, self).__init__()

        self.log_det_J: float = 0
        self.device: str = "cpu"

        if random_state is not None:
            torch.manual_seed(random_state)

        # Create the random orthonormal matrix via QR decomposition
        self.orthonormal_matrix: torch.Tensor = nn.Parameter(torch.linalg.qr(torch.randn(input_size, input_size))[0], requires_grad=False)
        self.orthonormal_matrix.requires_grad = False

    def to(self, device: str) -> "OrthonormalTransformation":  # type: ignore
        super().to(device)
        self.device = device
        self.orthonormal_matrix = self.orthonormal_matrix.to(device)

        return self

    def forward(self, y: torch.Tensor, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the transformation
        return y @ self.orthonormal_matrix

    def inverse(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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


class CondRealNVP_v2(ConditionalInvertibleLayer):
    def __init__(
            self,
            size: int,
            nested_sizes: list[int],
            n_blocks: int,
            n_conditions: int,
            feature_networks: list[FeatureNetwork | nn.Module | None] | None = None,
            dropout: float = 0.0,
            act_norm: bool = False,
            two_way: bool = False,
            layer: str = "Linear",
            layer_kwargs: dict[str, Any] | None = None,
            activation: str = "GELU",
            activation_kwargs: dict[str, Any] | None = None,
            device: str = "cpu",
            random_state: int | None = None,
            parameter_index_mapping: ParameterIndexMapping = None,
            hybrid: bool = False) -> None:
        super(CondRealNVP_v2, self).__init__()

        if n_conditions > 0:
            self.feature_network_stack = FeatureNetworkStack(feature_networks)

        self.size = size
        self.nested_sizes = nested_sizes
        self.n_blocks = n_blocks
        self.n_conditions = n_conditions
        self.device = device
        self.dropout = dropout
        self.parameter_index_mapping = parameter_index_mapping
        self.hybrid = hybrid
        self.log_det_J: torch.Tensor = torch.zeros(1).to(self.device)

        if self.hybrid:
            self.prediction_head = nn.Linear(self.n_conditions, self.size)

        # Create the network
        self.layers = nn.ModuleList()
        for _ in range(self.n_blocks - 1):
            if act_norm:
                self.layers.append(ActNorm(self.size))
            self.layers.append(ConditionalAffineCouplingLayer(
                self.size,
                self.nested_sizes,
                self.n_conditions,
                layer=layer,
                layer_kwargs=layer_kwargs,
                activation=activation,
                activation_kwargs=activation_kwargs,
                dropout=self.dropout,
                two_way=two_way,
                device=self.device))
            self.layers.append(OrthonormalTransformation(self.size, random_state=random_state))

        # Add the final affine coupling layer
        self.layers.append(ConditionalAffineCouplingLayer(
            self.size,
            self.nested_sizes,
            self.n_conditions,
            layer=layer,
            layer_kwargs=layer_kwargs,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=self.dropout,
            two_way=two_way,
            device=self.device))

    def verify(self) -> None:
        current_dimension = None
        # iterate over the feature networks
        for fn in self.feature_network_stack.feature_networks:
            # If the feature network has well defines sizes
            if isinstance(fn, FeatureNetwork):
                # If it is not the first feature network, check if the dimensions match
                if current_dimension is not None:
                    assert current_dimension == fn.input_size, f"The output dimension of the feature network must match the input dimension of the time series network. Have {current_dimension} but need {fn.input_size} for next layer."

                # Assign the current dimension to the output size of the feature network
                current_dimension = fn.output_size

        # At the end, check if the output dimension of the last feature network matches the number of conditions for the CNF
        if current_dimension is not None:
            assert current_dimension == self.n_conditions, f"The output dimension of the time series network must match the number of conditions. Have {current_dimension} but need {self.n_conditions}."

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CondRealNVP_v2":
        feature_networks = []
        for fn_config in config['feature_networks']:
            feature_network = FeatureNetworkFactory.get_feature_network(fn_config['type'], fn_config.get('kwargs', {}))
            feature_networks.append(feature_network)

        cnf = CondRealNVP_v2(
            feature_networks=feature_networks,
            parameter_index_mapping=ParameterIndexMapping(list(config["global"]["parameter_selection"])),
            **config["model"]["kwargs"])

        cnf.verify()

        return cnf

    def to(self, device: str) -> "CondRealNVP_v2":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        for layer in self.layers:
            layer.to(device)
        self.feature_network_stack.to(device)
        return self

    def forward(self, y: torch.Tensor, *conditions: torch.Tensor, log_det_J: bool = False, return_features: bool = False, deterministic_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.n_conditions > 0:
            if deterministic_features:
                self.feature_network_stack.eval()
                condition = self.feature_network_stack(*conditions).detach()
            else:
                condition = self.feature_network_stack(*conditions)

        # Apply the network
        if log_det_J:
            self.log_det_J = torch.zeros(y.shape[0]).to(self.device)

        for layer in self.layers:
            if isinstance(layer, ConditionalInvertibleLayer) and self.n_conditions > 0:
                y = layer(y, condition, log_det_J)
            elif isinstance(layer, InvertibleLayer):
                y = layer(y, log_det_J)
            else:
                raise ValueError(f"Layer must be an instance of ConditionalInvertibleLayer or InvertibleLayer, but got {type(layer)}")

            if log_det_J:
                self.log_det_J += layer.log_det_J

        if return_features:
            return y, condition

        return y

    def inverse(self, z: torch.Tensor, *conditions: torch.Tensor) -> torch.Tensor:
        if self.n_conditions > 0:
            condition = self.feature_network_stack(*conditions)

        # Apply the network in reverse
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalInvertibleLayer) and self.n_conditions > 0:
                z = layer.inverse(z, condition)
            elif isinstance(layer, InvertibleLayer):
                z = layer.inverse(z)
            else:
                raise ValueError(f"Layer must be an instance of ConditionalInvertibleLayer or InvertibleLayer, but got {type(layer)}")

        return z

    def sample(self, n_samples: int, *conditions: torch.Tensor, sigma: float = 1, outer: bool = False, batch_size: int = 100, sample_batch_size: int = None, output_device: str = "cpu", verbose: bool = False) -> torch.Tensor:
        if sample_batch_size is None:
            sample_batch_size = batch_size

        m_batch_sizes = [sample_batch_size] * (n_samples // sample_batch_size) + [n_samples % sample_batch_size]

        y_hat_list: list[list[torch.Tensor]] = []

        with torch.no_grad():
            for b in tqdm(range(0, len(conditions[0]), batch_size), desc="Batch Sampling", disable=not verbose):
                batch_conditions = [c[b: b + batch_size].to(self.device) for c in conditions]
                y_hat_list.append([])
                for m in m_batch_sizes:
                    if m == 0:
                        # Skip empty batch sizes
                        continue

                    y_hat = self._sample(
                        m,
                        *batch_conditions,
                        outer=outer,
                        sigma=sigma).to(output_device)

                    y_hat_list[-1].append(y_hat)

        # Create a tensor of shape (n_samples, y.shape[0], y.shape[1])
        y_hat = torch.cat([torch.cat(y_hat_batch, dim=0) for y_hat_batch in y_hat_list], dim=1)

        return y_hat

    def _sample(self, n_samples: int, *conditions: torch.Tensor, sigma: float = 1, outer: bool = False) -> torch.Tensor:
        """
        Sample from the model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        *conditions : torch.Tensor
            The conditions used for sampling.
            If 1st order tensor and len(y) == n_conditions, the same conditions are used for all samples.
            If 2nd order tensor, y.shape must be (n_samples, n_conditions), and each row is used as the conditions for each sample.
        sigma : float
            The standard deviation of the normal distribution to sample from.
        outer : bool
            If True, the conditions are broadcasted to match the shape of the samples.
            If False, the conditions are matched to the shape of the samples.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """

        if all(c.ndim == 1 for c in conditions):
            # Generate n_samples for each condition in y
            z = sigma * torch.randn(n_samples, self.size).to(self.device)
            repeated_conditions = [c.repeat(n_samples, 1) for c in conditions]

            # Apply the inverse network
            return self.inverse(z, *repeated_conditions).view(n_samples, self.size)
        elif all(c.ndim > 1 for c in conditions):
            if outer:
                if not len(set(c.shape[0] for c in conditions)) == 1:
                    raise ValueError(f"All conditions must have the same number of samples (dim = 0). Got {[c.shape for c in conditions]}.")
                n_samples_per_condition = conditions[0].shape[0]

                # Generate n_samples for each condition in y
                z = sigma * torch.randn(n_samples * n_samples_per_condition, self.size).to(self.device)
                repeated_conditions = [c.repeat(n_samples, *([1] * (c.ndim - 1))) for c in conditions]

                # Apply the inverse network
                return self.inverse(z, *repeated_conditions).view(n_samples, n_samples_per_condition, self.size)
            else:
                z = sigma * torch.randn(n_samples, self.size).to(self.device)

                return self.inverse(z, *conditions).view(n_samples, self.size)
        else:
            raise ValueError(f"Conditions have invalid shape: {[c.shape for c in conditions]}")

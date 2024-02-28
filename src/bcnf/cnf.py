import numpy as np
import torch
import torch.nn as nn

from bcnf.feature_network import FeatureNetwork


class ConditionalNestedNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_conditions: int) -> None:
        super(ConditionalNestedNeuralNetwork, self).__init__()

        self.n_conditions = n_conditions

        self.layers = nn.Sequential(
            nn.Linear(input_size + n_conditions, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size * 2)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x, y], dim=1)

        # Get the translation coefficients t and the scale coefficients s from the neural network
        t, s = self.layers(x).chunk(2, dim=1)

        # Return the coefficients
        return t, torch.tanh(s)


class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_conditions: int) -> None:
        super(ConditionalAffineCouplingLayer, self).__init__()

        self.n_classes = n_conditions
        self.log_det_J = None

        # Create the nested neural network
        self.nn = ConditionalNestedNeuralNetwork(
            input_size=np.ceil(input_size / 2).astype(int),
            output_size=np.floor(input_size / 2).astype(int),
            hidden_size=hidden_size,
            n_conditions=n_conditions)

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Split the input into two halves
        x_a, x_b = x.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn(x_a, y)

        # Apply the transformation
        y_a = x_a  # skip connection
        y_b = torch.exp(log_s) * x_b + t  # affine transformation

        # Calculate the log determinant of the Jacobian
        if log_det_J:
            self.log_det_J = log_s.sum(dim=1)

        # Return the output
        return torch.cat([y_a, y_b], dim=1)

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split the input into two halves
        z_a, z_b = z.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn(z_a, y)

        # Apply the inverse transformation
        x_a = z_a
        x_b = (z_b - t) * torch.exp(- log_s)

        # Return the output
        return torch.cat([x_a, x_b], dim=1)


class OrthonormalTransformation(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(OrthonormalTransformation, self).__init__()

        self.log_det_J = 0

        # Create the random orthonormal matrix via QR decomposition
        self.orthonormal_matrix = nn.Parameter(torch.linalg.qr(torch.randn(input_size, input_size))[0], requires_grad=False)

    def forward(self, x: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the transformation
        return x @ self.orthonormal_matrix

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Apply the inverse transformation
        return y @ self.orthonormal_matrix.T


class CondRealNVP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, blocks: int, n_conditions: int, feature_network: FeatureNetwork, device: str = "cpu"):
        super(CondRealNVP, self).__init__()

        if n_conditions == 0 or feature_network is None:
            self.h = nn.Identity()
        else:
            self.h = feature_network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.n_conditions = n_conditions
        self.device = device

        self.layers = self._build_network()

        self.log_det_J: float | None = None

    def _build_network(self) -> nn.ModuleList:
        # Create the network
        layers: list[nn.Module] = []
        for _ in range(self.blocks - 1):
            layers.append(ConditionalAffineCouplingLayer(self.input_size, self.hidden_size, self.n_conditions))
            layers.append(OrthonormalTransformation(self.input_size))

        # Add the final affine coupling layer
        layers.append(ConditionalAffineCouplingLayer(self.input_size, self.hidden_size, self.n_conditions))

        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network

        if log_det_J:
            self.log_det_J = 0

            for layer in self.layers:
                # If the layer is an affine coupling layer, we need to pass the inverse flag
                if isinstance(layer, ConditionalAffineCouplingLayer):
                    x = layer(x, log_det_J, y)
                else:
                    x = layer(x, log_det_J)
                self.log_det_J += layer.log_det_J

            return x

        for layer in self.layers:
            # If the layer is an affine coupling layer, we need to pass the inverse flag
            if isinstance(layer, ConditionalAffineCouplingLayer):
                x = layer(x, log_det_J, y)
            else:
                x = layer(x, log_det_J)

        return x

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network in reverse
        for layer in reversed(self.layers):
            # If the layer is an affine coupling layer, we need to pass the inverse flag
            if isinstance(layer, ConditionalAffineCouplingLayer):
                z = layer.inverse(z, y)
            else:
                z = layer.inverse(z)

        return z

    def sample(self, n_samples: int, y: int | list[int], outer: bool = True, sigma: float = 1) -> torch.Tensor:
        """
        Sample from the model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        y : int | list[int]
            The class to generate the samples from. If int, generate n_samples from that class. If list[int], generate n_samples for each class in the list if outer is True. If outer is False and dim(y) == n_samples, use y_i as the class for the i-th sample.
        outer : bool
            Whether to generate n_samples for each class in y (True) or to use y_i as the class for the i-th sample (False).
        sigma : float
            The standard deviation of the normal distribution to sample from.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """

        if isinstance(y, int):
            # Generate n_samples points for the given class y
            z = sigma * torch.randn(n_samples, self.input_size).to(self.device)

            # Broadcast y to the correct shape so it matches z
            y_broadcast = torch.tensor([y]).repeat(n_samples).to(self.device)

            # Apply the inverse network
            return self.inverse(z, y_broadcast).view(n_samples, 1, self.input_size)
        elif type(y) in [list, np.ndarray, torch.Tensor]:
            # Convert y to a tensor
            if isinstance(y, list) or isinstance(y, np.ndarray):
                y_tensor = torch.tensor(y, dtype=torch.float32)
                y_tensor = y_tensor.to(self.device)
            else:
                y_tensor = y.to(self.device)

            # Determine whether to generate n_samples for each class in y or to use y_i as the class for the i-th sample
            if outer or len(y_tensor) != n_samples:
                n_classes = len(y_tensor)

                # Generate n_samples for each class in y
                z = sigma * torch.randn(n_samples * len(y_tensor), self.input_size).to(self.device)
                y_tensor = y_tensor.repeat((n_samples, 1))

                # Apply the inverse network
                return self.inverse(z, y_tensor).view(n_samples, n_classes, self.input_size)
            elif len(y_tensor) == n_samples:
                # Use y_i as the class for the i-th sample
                z = sigma * torch.randn(n_samples, self.input_size).to(self.device)

                # Apply the inverse network
                return self.inverse(z, y_tensor).view(n_samples, 1, self.input_size)
            else:
                raise ValueError(f"y must be an int, a list of ints of length n_samples or a list of ints of length n_samples * len(y). y: {y_tensor}")
        else:
            raise ValueError(f"Unsupported type for y. Got {type(y)} but expected int, list, np.ndarray or torch.Tensor.")

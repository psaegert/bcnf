import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 paddings: list[int],
                 linear_hidden_channels: list[int],
                 num_features: int,
                 dropout_prob: float = 0.5) -> None:
        super(CNN, self).__init__()

        self.layers: list[nn.Module] = []
        self.pool = nn.MaxPool2d(2, 2)

        self.layers.append(nn.Conv2d(in_channels, hidden_channels[0], kernel_sizes[0], strides[0], paddings[0]))
        self.layers.append(nn.ReLU())  # Apply activation function after convolution
        self.layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
        self.layers.append(self.pool)

        for i in range(len(hidden_channels) - 1):
            self.layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], paddings[i + 1]))
            self.layers.append(nn.ReLU())  # Apply activation function after convolution
            self.layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
            self.layers.append(self.pool)

        # Calculate input size for the first linear layer after flattening
        dummy_input = torch.randn(1, in_channels, 160, 90)
        with torch.no_grad():
            dummy_output = self._forward_conv(dummy_input)
        dummy_output_flattened_size = dummy_output.view(1, -1).size(1)

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(dummy_output_flattened_size, linear_hidden_channels[0]))  # Update input size for linear layer
        self.layers.append(nn.ReLU())  # Apply activation function for linear layer
        self.layers.append(nn.Dropout(dropout_prob))  # Dropout after activation

        for i in range(len(linear_hidden_channels) - 1):
            self.layers.append(nn.Linear(linear_hidden_channels[i], linear_hidden_channels[i + 1]))
            self.layers.append(nn.ReLU())  # Apply activation function for linear layer
            self.layers.append(nn.Dropout(dropout_prob))  # Dropout after activation

        self.layers.append(nn.Linear(linear_hidden_channels[-1], num_features))

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.Dropout, nn.MaxPool2d)):
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

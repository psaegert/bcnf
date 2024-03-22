import torch
from torch import nn

from bcnf.models.feature_network import FeatureNetwork


class CNN(FeatureNetwork):
    def __init__(self,
                 hidden_channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 output_size_lin: int,
                 output_size: int,
                 image_input_size: tuple[int, int] = (90, 160),
                 dropout_prob: float = 0.5) -> None:
        super(CNN, self).__init__()

        self.input_size = image_input_size
        self.output_size = output_size

        self.cnn_layers: nn.Module
        self.pool = nn.MaxPool2d(2, 2)
        self.example_camera_input = torch.randn(1, 1, image_input_size[0], image_input_size[1])  # batch size, channels, height, width
        self.output_size_lin = output_size_lin

        layers = []
        padding_x = ((strides[0] - 1) * image_input_size[0] - strides[0] + kernel_sizes[0]) // 2
        padding_y = ((strides[0] - 1) * image_input_size[1] - strides[0] + kernel_sizes[0]) // 2
        padding = (padding_x, padding_y)
        layers.append(nn.Conv2d(1, hidden_channels[0], kernel_sizes[0], strides[0], padding))
        layers.append(nn.ReLU())  # Apply activation function after convolution
        layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
        layers.append(self.pool)

        output_size = self._calc_output_shape(self.example_camera_input, layers)  # type: ignore

        for i in range(len(hidden_channels) - 1):
            padding_x = ((strides[i] - 1) * output_size[2] - strides[i] + kernel_sizes[i]) // 2  # type: ignore
            padding_y = ((strides[i] - 1) * output_size[3] - strides[i] + kernel_sizes[i]) // 2  # type: ignore
            padding = (padding_x, padding_y)
            layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], padding))
            layers.append(nn.ReLU())  # Apply activation function after convolution
            layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
            layers.append(self.pool)

            output_size = self._calc_output_shape(self.example_camera_input, layers)  # type: ignore

        layers.append(nn.Flatten())
        self.final_output_size = output_size[1] * output_size[2] * output_size[3]  # type: ignore
        self.final_layer = nn.Linear(self.final_output_size * 2, self.output_size_lin)
        self.cnn_layers = nn.Sequential(*layers)

    def _calc_output_shape(self,
                           input: torch.tensor,
                           layers: list[nn.Module]) -> tuple[int, int]:

        for layer in layers:
            input = layer(input)

        return input.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input tensor to combine batch and sequence dimensions
        batch_size, num_cameras, sequence_length, img_x, img_y = x.size()

        # Make a giant batch with all the cameras and sequences
        x = x.view(batch_size * num_cameras * sequence_length, 1, img_x, img_y)

        # Process the entire sequence as a single batch
        y = self.cnn_layers(x)

        # Reshape features to restore the batch and sequence dimensions
        y = y.view(batch_size, num_cameras, sequence_length, -1)

        # Stack the features of all cameras
        shape = y.shape
        new_size = shape[1] * shape[3]
        y = y.reshape(shape[0], shape[2], new_size)

        y = self.final_layer(y)

        return y

    def to(self, device: torch.device) -> "CNN":
        self.cnn_layers = self.cnn_layers.to(device)
        return super().to(device)

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
                 dropout_prob: float = 0.5,
                 num_CNN: int = 1,
                 verbose: bool = False) -> None:
        super(CNN, self).__init__()

        self.input_size = image_input_size
        self.output_size = output_size

        self.cnn_layers = nn.ModuleList()  # Use nn.ModuleList to store layers
        self.pool = nn.MaxPool2d(2, 2)
        self.example_camera_input = torch.randn(1, 1, image_input_size[0], image_input_size[1])  # batch size, channels, height, width
        if verbose:
            print(f'Example camera input size: {self.example_camera_input.shape}')
        self.output_size_lin = output_size_lin
        self.num_CNN = num_CNN

        for _ in range(num_CNN):
            layers = []
            padding_x = ((strides[0] - 1) * image_input_size[0] - strides[0] + kernel_sizes[0]) // 2
            padding_y = ((strides[0] - 1) * image_input_size[1] - strides[0] + kernel_sizes[0]) // 2
            padding = (padding_x, padding_y)
            layers.append(nn.Conv2d(1, hidden_channels[0], kernel_sizes[0], strides[0], padding))
            layers.append(nn.ReLU())  # Apply activation function after convolution
            layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
            layers.append(self.pool)

            output_size = self._calc_output_shape(self.example_camera_input, layers)  # type: ignore
            if verbose:
                print(f'Output size after first layer: {output_size}')

            for i in range(len(hidden_channels) - 1):
                padding_x = ((strides[i] - 1) * output_size[2] - strides[i] + kernel_sizes[i]) // 2  # type: ignore
                padding_y = ((strides[i] - 1) * output_size[3] - strides[i] + kernel_sizes[i]) // 2  # type: ignore
                padding = (padding_x, padding_y)
                layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], padding))
                layers.append(nn.ReLU())  # Apply activation function after convolution
                layers.append(nn.Dropout(dropout_prob))  # Dropout after activation
                layers.append(self.pool)

                output_size = self._calc_output_shape(self.example_camera_input, layers)  # type: ignore
                if verbose:
                    print(f'Output size after layer {i + 1}: {output_size}')

            layers.append(nn.Flatten())
            self.cnn_layers.append(nn.Sequential(*layers))

        self.final_output_size = output_size[1] * output_size[2] * output_size[3]  # type: ignore
        if verbose:
            print(f'Final output size: {self.final_output_size * 2}')
        self.final_layer = nn.Linear(self.final_output_size * 2, self.output_size_lin)

        if verbose:
            print(self.cnn_layers)

    def _calc_output_shape(self,
                           input: torch.tensor,
                           layers: list[nn.Module]) -> tuple[int, int]:

        for layer in layers:
            input = layer(input)

        return input.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x e.g. torch.Size([5, 2, 30, 90, 160])
        # Reshape input tensor to combine batch and sequence dimensions
        batch_size, num_cameras, sequence_length, img_x, img_y = x.size()

        # Separate the cameras
        # -> torch.Size([2, 5, 30, 90, 160])
        x = x.permute(1, 0, 2, 3, 4)

        # Make a giant batch with all the cameras and sequences
        if self.num_CNN > 1:
            # -> torch.Size([2, 150, 90, 160])
            x = x.reshape(num_cameras, batch_size * sequence_length, 1, img_x, img_y)
            y = []
            for i in range(self.num_CNN):
                y.append(self.cnn_layers[i](x[i]))
            y = torch.stack(y, dim=0)
        else:
            # -> torch.Size([1, 300, 90, 160])
            x = x.reshape(1, batch_size * num_cameras * sequence_length, 1, img_x, img_y)
            y = self.cnn_layers[0](x[0])

        # Reshape features to restore the batch and sequence dimensions
        # -> torch.Size([2, 5, 30, #features])
        y = y.view(num_cameras, batch_size, sequence_length, -1)  # type: ignore
        # -> torch.Size([5, 2, 30, #features])
        y = y.permute(1, 0, 2, 3)  # type: ignore

        # Stack the features of all cameras
        shape = y.shape  # type: ignore
        new_size = shape[1] * shape[3]
        y = y.reshape(shape[0], shape[2], new_size)  # type: ignore

        y = self.final_layer(y)

        return y

    def to(self, device: torch.device) -> "CNN":
        self.cnn_layers.to(device)
        return super().to(device)

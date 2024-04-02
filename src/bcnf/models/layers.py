from typing import Any

import torch
from torch import nn

from bcnf.factories import LayerFactory


class AnyGLU(nn.Module):
    """
    Generalized Linear Unit (GLU) layer that can be used with any activation function.
    """
    def __init__(self, input_size: int, output_size: int, activation: str = "GELU", activation_kwargs: dict[str, Any] | None = None) -> None:
        super(AnyGLU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.linear_gate = nn.Linear(self.input_size, self.output_size)
        self.linear_value = nn.Linear(self.input_size, self.output_size)
        self.activation = LayerFactory.get_layer(activation, **activation_kwargs or {})

    def to(self, device: torch.device) -> Any:
        self.linear_gate.to(device)
        self.linear_value.to(device)
        self.activation.to(device)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_value(x) * self.activation(self.linear_gate(x))


class FFTLayer(nn.Module):
    """
    Fast Fourier Transform (FFT) layer. Transforms the input into the frequency domain. The FFT of a real signal is Hermitian-symmetric, X[i] = conj(X[-i]) so the output contains only the positive frequencies below the Nyquist frequency.
    """
    def __init__(self) -> None:
        super(FFTLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        complex_f = torch.fft.rfft(input=x, dim=-1, norm='forward')

        # Concatenate the real and imaginary parts
        return torch.cat((complex_f.real, complex_f.imag), dim=-1)


class FFTEnrichLayer(nn.Module):
    """
    Concatenate the input with the FFT of the input.
    """
    def __init__(self) -> None:
        super(FFTEnrichLayer, self).__init__()
        self.fft = FFTLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.fft(x)), dim=-1)


class LinearFFTEnriched(nn.Module):
    """
    Linear layer that enriches the input with the FFT of the input.
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super(LinearFFTEnriched, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fft_enrich = FFTEnrichLayer()

        self.linear = nn.Linear(input_size + 2 * (input_size // 2 + 1), output_size)

    def to(self, device: torch.device) -> Any:
        self.linear.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.fft_enrich(x))

from typing import Any, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import warnings


class FeatureNetwork(nn.Module):
    input_size: int
    output_size: int

    def __init__(self) -> None:
        super(FeatureNetwork, self).__init__()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> 'FeatureNetwork':
        return super().to(*args, **kwargs)


class FeatureNetworkStack(FeatureNetwork):
    def __init__(self, feature_networks: list[FeatureNetwork | nn.Module | None] | None = None) -> None:
        super(FeatureNetwork, self).__init__()

        if feature_networks is None or all(fn is None for fn in feature_networks):
            raise ValueError('Feature network stack must contain at least one feature network.')
        else:
            self.feature_networks = nn.Sequential(*[fn for fn in feature_networks if fn is not None])

        self.n_distinct_conditions = sum(1 for fn in self.feature_networks if isinstance(fn, ConcatenateCondition))

        self.input_size = self.feature_networks[0].input_size
        self.output_size = self.feature_networks[-1].output_size

    @property
    def n_params(self) -> int:
        return sum(sum(p.numel() for fn in self.feature_networks for p in fn.parameters()))

    def forward(self, *conditions: torch.Tensor,) -> torch.Tensor:
        if len(conditions) != self.n_distinct_conditions:
            raise ValueError(f'Expected {self.n_distinct_conditions} conditions, but got {len(conditions)}.')

        # Apply the feature network to y
        consume_condition_index = 0
        current_features: torch.Tensor | None = None
        for i, fn in enumerate(self.feature_networks):
            if isinstance(fn, ConcatenateCondition):
                # Consume one condition and concatenate it with the current features
                if current_features is None:
                    # No current features, use the first provided condition as input to the first feature network
                    current_features = fn(conditions[consume_condition_index])
                else:
                    # TODO: Properly handle concatenation of features of different shapes and n dims.
                    current_features = fn(torch.cat([current_features, conditions[consume_condition_index]], dim=fn.dim))

                # "Consume" the condition by incrementing the index of the next condition
                consume_condition_index += 1
            else:
                # Apply the feature network to the current features
                current_features = fn(current_features)

        return current_features

    def to(self, *args: Any, **kwargs: Any) -> 'FeatureNetwork':
        self.feature_networks = self.feature_networks.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ConcatenateCondition(FeatureNetwork):
    def __init__(self, input_size: int, output_size: int, dim: int = -1) -> None:
        super(ConcatenateCondition, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dim = dim

    def to(self, *args: Any, **kwargs: Any) -> 'ConcatenateCondition':
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FrExpFeatureNetwork(FeatureNetwork):
    def __init__(self, input_size: int, separate_sign: bool = False) -> None:
        super(FrExpFeatureNetwork, self).__init__()

        self.separate_sign = separate_sign

        self.input_size = input_size
        self.output_size = input_size * (2 + int(separate_sign))

    def to(self, *args: Any, **kwargs: Any) -> 'FrExpFeatureNetwork':
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mantissa, exponent = torch.frexp(x)

        if self.separate_sign:
            sign = torch.sign(mantissa)
            mantissa = torch.abs(mantissa)
            return torch.cat([sign, mantissa, exponent], dim=-1)
        else:
            return torch.cat([mantissa, exponent], dim=-1)


class FullyConnectedFeatureNetwork(FeatureNetwork):
    def __init__(self, sizes: list[int], activation: Type[nn.Module] = nn.GELU, dropout: float = 0.0, batch_norm: bool = False) -> None:
        super(FullyConnectedFeatureNetwork, self).__init__()

        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        self.nn = nn.Sequential()
        self.output_size_lin = sizes[-1]

        if len(sizes) < 2:
            # No transformations from one layer to another, use identity (0 layers)
            warnings.warn('No hidden layers in the fully connected network. Using identity function.')
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

    def to(self, *args: Any, **kwargs: Any) -> 'FullyConnectedFeatureNetwork':
        self.nn = self.nn.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.nn.forward(x)


class LSTMFeatureNetwork(FeatureNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.0, bidirectional: bool = False, pooling: str = 'mean') -> None:
        super(LSTMFeatureNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        if pooling not in ['mean', 'max']:
            raise ValueError(f'Pooling method {pooling} not supported. Use either "mean" or "max".')

        self.pooling = pooling

    def to(self, *args: Any, **kwargs: Any) -> 'LSTMFeatureNetwork':
        self.lstm = self.lstm.to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(1, 0, 2)  # Turn (batch_size, seq_len, n_features) into (seq_len, batch_size, n_features) for compatibility with LSTMs
        x, _ = self.lstm.forward(x)
        x = self.linear.forward(x)

        if self.pooling == 'mean':
            return x.mean(dim=0)
        elif self.pooling == 'max':
            return x.max(dim=0).values

        raise ValueError(f'Pooling method {self.pooling} not supported. Use either "mean" or "max".')


# Thank you to whoever wrote this code
# https://www.phind.com/search?cache=g0lcwwuzznvnajlwok2676zp
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def to(self, *args: Any, **kwargs: Any) -> 'MultiHeadAttention':
        self.q_linear = self.q_linear.to(*args, **kwargs)
        self.k_linear = self.k_linear.to(*args, **kwargs)
        self.v_linear = self.v_linear.to(*args, **kwargs)
        self.fc_out = self.fc_out.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: bool = None) -> torch.Tensor:
        # Linear transformations for query, key, and value
        batch_size = query.size(0)

        # Perform linear operation and split into multiple heads
        query = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Multiply attention weights with values
        output = torch.matmul(attention_weights, value)

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc_out(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_size: int, dropout: float = 0.1) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def to(self, *args: Any, **kwargs: Any) -> 'TransformerBlock':
        self.attention = self.attention.to(*args, **kwargs)
        self.norm1 = self.norm1.to(*args, **kwargs)
        self.norm2 = self.norm2.to(*args, **kwargs)
        self.ffn = self.ffn.to(*args, **kwargs)
        self.dropout = self.dropout.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class Transformer(FeatureNetwork):
    def __init__(self, input_size: int, trf_size: int, n_heads: int, ff_size: int, n_blocks: int, output_size: int, dropout: float = 0.5, trf_dropout: float = 0.1, add_positional_embeddings: bool = False) -> None:
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.add_positional_embeddings = add_positional_embeddings
        self.trf_size = trf_size

        self.features = nn.Linear(input_size, trf_size)
        self.layers = nn.ModuleList([TransformerBlock(trf_size, n_heads, ff_size, trf_dropout) for _ in range(n_blocks)])
        self.output = nn.Linear(trf_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def to(self, *args: Any, **kwargs: Any) -> 'Transformer':
        self.features = self.features.to(*args, **kwargs)
        self.layers = self.layers.to(*args, **kwargs)
        self.output = self.output.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)

        if self.add_positional_embeddings:
            seq_len = x.size(1)
            positional_embeddings = torch.zeros(seq_len, self.trf_size, device=x.device)
            for i in range(seq_len):
                for j in range(self.input_size):
                    if j % 2 == 0:
                        positional_embeddings[i, j] = np.sin(i / 10000 ** (2 * j / self.input_size))
                    else:
                        positional_embeddings[i, j] = np.cos(i / 10000 ** (2 * j / self.input_size))

            x += positional_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.dropout(x)

        # Use the first token's output as the final output
        x = self.output(x[:, 0, :])

        return x


class VerboseLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0, bidirectional: bool = False) -> None:
        super(VerboseLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.ModuleList()

        for i in range(num_layers - 1):
            self.lstm.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True))
            input_size = hidden_size * (2 if bidirectional else 1)

            if dropout > 0:
                self.lstm.append(nn.Dropout(p=dropout))

        self.lstm.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True))

    def to(self, *args: Any, **kwargs: Any) -> 'VerboseLSTM':
        self.lstm = self.lstm.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.empty(self.num_layers, x.size(0), x.size(1), self.hidden_size * (2 if self.bidirectional else 1), device=x.device)

        lstm_index = 0
        for layer in self.lstm:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                h[lstm_index] = x
                lstm_index += 1
            else:
                x = layer(x)

        return x, h.permute(1, 0, 2, 3)


class DualDomainLSTM(FeatureNetwork):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            fc_sizes: list[int],
            fc_dropout: float = 0.0,
            num_layers: int = 1,
            dropout: float = 0.0,
            bidirectional: bool = False,
            pooling: str = 'mean') -> None:
        super(DualDomainLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = fc_sizes[-1]
        self.pooling = pooling

        self.lstm = VerboseLSTM(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.frequency_lstm = VerboseLSTM(input_size * 2, hidden_size, num_layers, dropout, bidirectional)

        fc_sizes = [hidden_size * (2 if bidirectional else 1) * 2] + fc_sizes

        self.fc = FullyConnectedFeatureNetwork(sizes=fc_sizes, dropout=fc_dropout)

    def to(self, *args: Any, **kwargs: Any) -> 'VerboseLSTM':
        self.lstm = self.lstm.to(*args, **kwargs)
        self.frequency_lstm = self.frequency_lstm.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lstm, _ = self.lstm.forward(x)

        x_frequencies = torch.fft.rfft(x, dim=1)

        x_frequencies_lstm, _ = self.frequency_lstm.forward(torch.cat([x_frequencies.real, x_frequencies.imag], dim=-1))

        if self.pooling == 'mean':
            x_lstm_pooled = x_lstm.mean(dim=1)
            x_frequencies_lstm_pooled = x_frequencies_lstm.mean(dim=1)
        elif self.pooling == 'max':
            x_lstm_pooled = x_lstm.max(dim=1).values
            x_frequencies_lstm_pooled = x_frequencies_lstm.max(dim=1).values
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling}")

        x_cat = torch.cat([x_lstm_pooled, x_frequencies_lstm_pooled], dim=1)

        return self.fc.forward(x_cat)


class DualDomainTransformer(FeatureNetwork):
    def __init__(
        self,
        input_size: int,
        trf_size: int,
        n_heads: int,
        ff_size: int,
        n_blocks: int,
        fc_sizes: list[int],
        fc_dropout: float = 0.5,
        trf_dropout: float = 0.1,
        dropout: float = 0.5,
        add_positional_embeddings: bool = False
    ) -> None:
        super(DualDomainTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = fc_sizes[-1]
        self.add_positional_embeddings = add_positional_embeddings

        # Time domain transformer
        self.time_transformer = Transformer(
            input_size=input_size,
            trf_size=trf_size,
            n_heads=n_heads,
            ff_size=ff_size,
            n_blocks=n_blocks,
            output_size=trf_size,  # Output size is trf_size for intermediate representation
            dropout=dropout,
            trf_dropout=trf_dropout,
            add_positional_embeddings=add_positional_embeddings
        )

        # Frequency domain transformer, handling input_size * 2 for real and imaginary components
        self.frequency_transformer = Transformer(
            input_size=input_size * 2,
            trf_size=trf_size,
            n_heads=n_heads,
            ff_size=ff_size,
            n_blocks=n_blocks,
            output_size=trf_size,  # Output size is trf_size for intermediate representation
            dropout=dropout,
            trf_dropout=trf_dropout,
            add_positional_embeddings=add_positional_embeddings
        )

        # Fully connected network for final processing
        fc_input_size = trf_size * 2  # Concatenated outputs from time and frequency transformers
        self.fc_sizes = [fc_input_size] + fc_sizes
        self.fc = FullyConnectedFeatureNetwork(sizes=self.fc_sizes, dropout=fc_dropout)

    def to(self, *args: Any, **kwargs: Any) -> 'DualDomainTransformer':
        self.time_transformer = self.time_transformer.to(*args, **kwargs)
        self.frequency_transformer = self.frequency_transformer.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process in time domain
        x_out = self.time_transformer(x)

        # FFT transformation and processing in frequency domain
        x_frequencies = torch.fft.rfft(x, dim=1)
        x_frequencies = torch.cat([x_frequencies.real, x_frequencies.imag], dim=-1)
        x_frequency_out = self.frequency_transformer(x_frequencies)

        # Concatenate the outputs from both transformers
        x_combined = torch.cat([x_out, x_frequency_out], dim=1)

        # Final processing through the fully connected network
        return self.fc(x_combined)


class DualDomainFC(FeatureNetwork):
    def __init__(
        self,
        input_size: int,
        sizes: list[int],
        fc_sizes: list[int],
        dropout: float = 0.5,
        add_positional_embeddings: bool = False
    ) -> None:
        super(DualDomainFC, self).__init__()

        self.input_size = input_size
        self.output_size = fc_sizes[-1]
        self.add_positional_embeddings = add_positional_embeddings

        # Time domain fully connected network
        self.time_fc = FullyConnectedFeatureNetwork(
            sizes=[input_size] + sizes,
            dropout=dropout
        )

        # Frequency domain fully connected network, handling input_size * 2 for real and imaginary components
        self.frequency_fc = FullyConnectedFeatureNetwork(
            sizes=[input_size * 2] + sizes,
            dropout=dropout
        )

        # Fully connected network for final processing
        fc_input_size = input_size * 2  # Concatenated outputs from time and frequency transformers
        self.fc_sizes = [fc_input_size] + fc_sizes
        self.fc = FullyConnectedFeatureNetwork(sizes=self.fc_sizes, dropout=dropout)

    def to(self, *args: Any, **kwargs: Any) -> 'DualDomainFC':
        self.time_fc = self.time_fc.to(*args, **kwargs)
        self.frequency_fc = self.frequency_fc.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process in time domain
        x_out = self.time_fc(x)

        # FFT transformation and processing in frequency domain
        x_frequencies = torch.fft.rfft(x, dim=1)
        x_frequencies = torch.cat([x_frequencies.real, x_frequencies.imag], dim=-1)
        x_frequency_out = self.frequency_fc(x_frequencies)

        # Concatenate the outputs from both transformers
        x_combined = torch.cat([x_out, x_frequency_out], dim=1)

        # Final processing through the fully connected network
        return self.fc(x_combined)

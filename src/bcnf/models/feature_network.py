from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FeatureNetwork(nn.Module):
    def __init__(self) -> None:
        super(FeatureNetwork, self).__init__()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, *args, **kwargs):  # type: ignore
        self.nn = self.nn.to(*args, **kwargs)  # type: ignore
        return super().to(*args, **kwargs)


class FullyConnectedFeatureNetwork(FeatureNetwork):
    def __init__(self,
                 sizes: list[int],
                 activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:
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
        x = x.view(x.size(0), -1)
        return self.nn.forward(x)


class LSTMFeatureNetwork(FeatureNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.0, bidirectional: bool = False, pooling: str = 'mean') -> None:
        super(LSTMFeatureNetwork, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        if pooling not in ['mean', 'max']:
            raise ValueError(f'Pooling method {pooling} not supported. Use either "mean" or "max".')

        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # Turn (batch_size, seq_len, n_features) into (seq_len, batch_size, n_features) for compatibility with LSTMs
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
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: bool = None) -> torch.Tensor:
        # Linear transformations for query, key, and value
        batch_size = query.size(0)

        # Perform linear operation and split into multiple heads
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

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
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, output_size: int, dropout: float = 0.1) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = self.fc(x[:, 0, :])

        return x

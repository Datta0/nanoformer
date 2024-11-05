import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .mlp import MLP
from .attention import ATTN_TYPES
from .embedding import RotaryEmbedding

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Layer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super(Layer, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.attention_type

        self.input_layernorm = RMSNorm(self.config.hidden_size,)
        self.attention = ATTN_TYPES[self.attention_type](config)
        self.mlp = MLP(config)

    def forward(self, X, position_embeddings, mask=None):
        residual = X

        if self.config.input_layernorm:
            X = self.input_layernorm(X)

        X, _ = self.attention(X, position_embeddings, mask)

        if self.config.post_attention_layernorm:
            X = self.input_layernorm(X)
        
        residual = residual + X * self.config.residual_multiplier

        if self.config.pre_ffnn_layernorm:
            residual = self.input_layernorm(residual)

        X = self.mlp(residual)

        if self.config.post_ffnn_layernorm:
            X = self.input_layernorm(X)
        
        residual = residual + X * self.config.residual_multiplier

        return residual
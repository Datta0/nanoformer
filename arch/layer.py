import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .mlp import MLP
from .attention import ATTN_TYPES, RMSNorm
from .embedding import RotaryEmbedding



class Layer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super(Layer, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.attention_type

        self.input_layernorm = RMSNorm(self.config.hidden_size,)
        kwargs = {
            "layer_idx": layer_idx
        }
        self.attention = ATTN_TYPES[self.attention_type](config, **kwargs)
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
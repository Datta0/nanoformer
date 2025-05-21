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
        assert self.attention_type in ATTN_TYPES, f"Unknown attention type {self.attention_type}. Choose one of {list(ATTN_TYPES.keys())}"
        self.attention = ATTN_TYPES[self.attention_type](config, **kwargs)
        self.mlp = MLP(config)

        self.use_ngpt = config.use_ngpt
        self.scale = self.config.hidden_size ** -0.5

        if self.use_ngpt:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = self.scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.hidden_size))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = self.scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.hidden_size))

    def mse_head_sim_loss(self, attn_scores):
        # attn_scores: (batch, num_heads, seq, seq)
        n_heads = attn_scores.shape[1]
        # Flatten batch and seq dims for easier broadcasting
        heads = attn_scores.permute(1, 0, 2, 3).reshape(n_heads, -1)
        diffs = heads.unsqueeze(0) - heads.unsqueeze(1)  # (n_heads, n_heads, N)
        mse = (diffs ** 2).mean(-1)
        # Only upper triangle, excluding diagonal
        loss = mse.triu(1).sum() / (n_heads * (n_heads - 1) / 2)
        return loss

    def forward(self, X, position_embeddings, mask=None, return_attn_scores=False, return_head_sim_loss=False):
        residual = X

        if self.config.input_layernorm:
            X = self.input_layernorm(X)

        head_sim_loss = None
        if return_attn_scores or return_head_sim_loss:
            X, _, attn_scores = self.attention(X, position_embeddings, mask, return_attn_scores=True)
            if return_head_sim_loss:
                head_sim_loss = self.mse_head_sim_loss(attn_scores)
        else:
            X, _ = self.attention(X, position_embeddings, mask)

        if self.config.post_attention_layernorm:
            X = self.input_layernorm(X)
        
        if self.use_ngpt:
            alpha = torch.abs(self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling))
            
            # this ain't mentioned in the paper but is here(?) https://github.com/NVIDIA/ngpt/blob/ed19eb232380d41a9d71024b58d2403937d8f2c9/model.py#L148
            residual = F.normalize(residual, dim=-1)
            X = F.normalize(X, dim=-1)

            residual = alpha * (X-residual) + residual
        else:
            residual = residual + X * self.config.residual_multiplier

        if self.config.pre_ffnn_layernorm:
            residual = self.input_layernorm(residual)

        X = self.mlp(residual)

        if self.config.post_ffnn_layernorm:
            X = self.input_layernorm(X)

        if self.use_ngpt:
            alpha = torch.abs(self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling))
            # this ain't mentioned in the paper but is here(?) https://github.com/NVIDIA/ngpt/blob/ed19eb232380d41a9d71024b58d2403937d8f2c9/model.py#L172
            residual = F.normalize(residual, dim=-1)
            X = F.normalize(X, dim=-1)

            residual = alpha * (X-residual) + residual
        else:
            residual = residual + X * self.config.residual_multiplier

        if return_attn_scores and not return_head_sim_loss:
            return residual, attn_scores
        elif return_head_sim_loss:
            return residual, head_sim_loss
        return residual
    

    def get_param_count(self,):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def normalize_weights(self):
        for _, param in self.named_parameters():
            if param.ndim > 1:
                norm = param.norm(dim=-1, keepdim=True)
                param.data.div_(norm + 1e-6)  # Add small epsilon for numerical stability
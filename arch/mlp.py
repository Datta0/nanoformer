import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from .config import Config

class MLP(nn.Module):
    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.config = config
        
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.hidden_act = ACT2FN[config.hidden_act]
        self.use_ngpt = config.use_ngpt

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=config.mlp_bias)

        if self.use_ngpt:
            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(self.intermediate_size, dtype=torch.float32))
            self.scale = self.hidden_dim ** 0.5


    def forward(self, X):
        gate = self.gate_proj(X)
        up = self.up_proj(X)

        if self.use_ngpt:
            suv = self.suv * (self.suv_init_value/self.suv_init_scaling) * self.scale
            up = suv * up

        down = self.down_proj(self.hidden_act(gate) * up)
        return down
    
    def get_param_count(self,):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def normalize_weights(self):
        for _, param in self.named_parameters():
            if param.ndim > 1:
                norm = param.norm(dim=-1, keepdim=True)
                param.data.div_(norm + 1e-6)  # Add small epsilon for numerical stability

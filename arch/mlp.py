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

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim)


    def forward(self, X):
        gate = self.gate_proj(X)
        up = self.up_proj(X)
        down = self.down_proj(self.hidden_act(gate * up))
        return down


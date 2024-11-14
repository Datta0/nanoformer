import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Reshape cos and sin to match head dimension
    head_dim = q.shape[-1]
    cos = cos[..., :head_dim]
    sin = sin[..., :head_dim]
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Compute variance and normalize
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Apply elementwise affine scaling if enabled
        if self.elementwise_affine:
            hidden_states = self.weight * hidden_states
        
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        affine_repr = f", elementwise_affine={self.elementwise_affine}" if not self.elementwise_affine else ""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}{affine_repr}"


class Attention(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super(Attention, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        kv_size = self.head_dim * config.num_key_value_heads
        q_size = self.head_dim * config.num_attention_heads

        self.dropout = nn.Dropout(config.attention_dropout)
        
        if config.attention_scale:
            self.scale = config.attention_scale
        else:
            self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(self.hidden_dim, q_size)
        self.k = nn.Linear(self.hidden_dim, kv_size)
        self.v = nn.Linear(self.hidden_dim, kv_size)
        self.o = nn.Linear(self.hidden_dim, q_size)

    def forward(self, X, position_embeddings, mask=None):

        bsz, q_len, _ = X.shape
        query, key, value = self.q(X), self.k(X), self.v(X)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if self.config.attention_cap:
            attn_weights = attn_weights.clamp(min=-self.config.attention_cap, max=self.config.attention_cap)

        if mask is not None:
            # Expand mask for attention heads dimension
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        else:
            # Create causal mask and expand for attention heads
            causal_mask = torch.ones(bsz, q_len, q_len).to(X.device).triu(1)
            causal_mask = causal_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(causal_mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous()
        attn_output = self.o(context.view(bsz, q_len, -1))

        return attn_output, attn_weights


class DiffAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super(DiffAttention, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.layer_idx = kwargs.get("layer_idx",0)
        kv_size = self.head_dim * self.num_key_value_heads
        q_size = self.head_dim * self.num_heads

        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = config.attention_scale if config.attention_scale else self.head_dim ** -0.5

        # Define primary and noise-canceling linear layers for Q, K, V
        self.q = nn.Linear(self.hidden_dim, q_size)
        self.k = nn.Linear(self.hidden_dim, kv_size)
        
        self.qn = nn.Linear(self.hidden_dim, q_size)
        self.kn = nn.Linear(self.hidden_dim, kv_size)
        
        self.v = nn.Linear(self.hidden_dim, 2 * kv_size)
        self.o = nn.Linear(2*self.hidden_dim, q_size)
        
        # Lambda parameters for attention noise cancellation
        self.lambda_q1 = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.lambda_init = self.lambda_init_fn(depth=self.layer_idx)

        # Layer normalization
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.head_dim)
    
    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, X, position_embeddings, mask=None):
        bsz, q_len, _ = X.shape
        query, key, value = self.q(X), self.k(X), self.v(X)
        query_noise, key_noise = self.qn(X), self.kn(X)

        # Reshape and transpose to match attention heads
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, 2*self.head_dim).transpose(1, 2)

        query_noise = query_noise.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_noise = key_noise.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply positional embeddings
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query_noise, key_noise = apply_rotary_pos_emb(query_noise, key_noise, cos, sin)

        key = repeat_kv(key, self.num_key_value_groups)
        key_noise = repeat_kv(key_noise, self.num_key_value_groups)

        value = repeat_kv(value, self.num_key_value_groups)
        
        # Compute differential attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_noise = torch.matmul(query_noise, key_noise.transpose(-2, -1)) * self.scale

        lambda_val = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)) \
                     - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)) + self.lambda_init
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_noise = F.softmax(attn_noise, dim=-1)
        attn_weights = attn_weights - lambda_val * attn_noise
        attn_weights = self.dropout(attn_weights)
        
        # Compute the output
        output = torch.matmul(attn_weights, value)
        output = self.norm(output.view(bsz * self.num_heads * 2 , self.head_dim, q_len)).view(bsz, self.num_heads, q_len, -1) * (1 - self.lambda_init)
        output = output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        output = self.o(output)
        return output, attn_weights


ATTN_TYPES = {
    "gqa": Attention,
    "diff": DiffAttention,
}
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

class Attention(nn.Module):
    def __init__(self, config: Config,):
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


ATTN_TYPES = {
    "gqa": Attention,
}
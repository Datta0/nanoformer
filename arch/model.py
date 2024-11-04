import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .layer import Layer, RMSNorm
from .embedding import RotaryEmbedding
from transformers.modeling_utils import PreTrainedModel


class PreTrainedModel(PreTrainedModel):
    config_class = Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Layer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class NanoFormer(nn.Module):
    def __init__(self, config: Config):
        super(NanoFormer, self).__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.norm = RMSNorm(config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.hidden_size)
        self.layers = nn.ModuleList([Layer(config, i) for i in range(self.num_layers)])

    def forward(self, X, mask=None):
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            X = self.embed_tokens(X)
            if position_ids is None:
                position_ids = torch.arange(X.size(1), device=X.device).unsqueeze(0).expand(X.size(0), -1)
            position_embeddings = self.rotary_emb(X, position_ids)
            
            for layer in self.layers:
                X = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    X, position_embeddings, mask
                )
            X = self.norm(X)
            return X
        else:
            if position_ids is None:
                position_ids = torch.arange(X.size(1), device=X.device).unsqueeze(0).expand(X.size(0), -1)
            position_embeddings = self.rotary_emb(X, position_ids)
            for layer in self.layers:
                X = layer(X, position_embeddings, mask)
            X = self.norm(X)
            return X

    def gradient_checkpointing_enable(self, value=True):
        self.gradient_checkpointing = value

class NanoFormerForCausalLM(NanoFormer):
    def __init__(self, config: Config):
        super(NanoFormerForCausalLM, self).__init__(config)
        self.model = NanoFormer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, X, return_loss = False):
        X = self.model(X)
        logits = self.lm_head(X)
        if self.config.logit_cap:
            logits = torch.clamp(logits, -self.config.logit_cap, self.config.logit_cap)
        
        loss = None
        if return_loss:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            labels = logits[..., 1:, :].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return logits,loss
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

    def forward(self, input_ids, attention_mask=None, position_ids=None, mask=None):
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            input_ids = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
            position_embeddings = self.rotary_emb(input_ids, position_ids)
            
            for layer in self.layers:
                input_ids = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    input_ids, position_embeddings, mask
                )
            input_ids = self.norm(input_ids)
            return input_ids
        else:
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
            position_embeddings = self.rotary_emb(input_ids, position_ids)
            for layer in self.layers:
                input_ids = layer(input_ids, position_embeddings, mask)
            input_ids = self.norm(input_ids)
            return input_ids

    def gradient_checkpointing_enable(self, value=True):
        self.gradient_checkpointing = value

class NanoFormerForCausalLM(NanoFormer):
    def __init__(self, config: Config):
        super(NanoFormerForCausalLM, self).__init__(config)
        self.model = NanoFormer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask, return_loss = True):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        if self.config.logit_cap:
            logits = torch.clamp(logits, -self.config.logit_cap, self.config.logit_cap)
        
        loss = None
        if return_loss:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

        return logits, loss
    
    def set_train(self,):
        self.model.train()
        self.model.gradient_checkpointing_enable(True)
        self.lm_head.train()
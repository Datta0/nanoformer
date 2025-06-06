import torch
from torch import nn
from torch.nn import functional as F
import os
import json
import glob
try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .config import Config
from .layer import Layer, RMSNorm
from .embedding import RotaryEmbedding
from transformers.modeling_utils import PreTrainedModel
import torch.nn.utils.parametrize as parametrize


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
    def __init__(self, config: Config, gradient_checkpointing=False):
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
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, input_ids, attention_mask=None, position_ids=None, mask=None, return_attn_scores=False, return_head_sim_loss=False, **kwargs):
        attn_scores_all = [] if return_attn_scores else None
        head_sim_losses = [] if return_head_sim_loss else None
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
                if return_attn_scores:
                    input_ids, attn_scores = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        input_ids, position_embeddings, mask, True, False
                    )
                    attn_scores_all.append(attn_scores)
                elif return_head_sim_loss:
                    input_ids, head_sim_loss = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        input_ids, position_embeddings, mask, False, True
                    )
                    head_sim_losses.append(head_sim_loss)
                else:
                    input_ids = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        input_ids, position_embeddings, mask, False, False
                    )
            input_ids = self.norm(input_ids)
            if return_attn_scores:
                return input_ids, attn_scores_all
            if return_head_sim_loss:
                return input_ids, head_sim_losses
            return input_ids
        else:
            input_ids = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
            position_embeddings = self.rotary_emb(input_ids, position_ids)
            for layer in self.layers:
                if return_attn_scores:
                    input_ids, attn_scores = layer(input_ids, position_embeddings, mask, True, False)
                    attn_scores_all.append(attn_scores)
                elif return_head_sim_loss:
                    input_ids, head_sim_loss = layer(input_ids, position_embeddings, mask, False, True)
                    head_sim_losses.append(head_sim_loss)
                else:
                    input_ids = layer(input_ids, position_embeddings, mask, False, False)
            input_ids = self.norm(input_ids)
            if return_attn_scores:
                return input_ids, attn_scores_all
            if return_head_sim_loss:
                return input_ids, head_sim_losses
            return input_ids
    
    @torch.no_grad()
    def normalize_weights(self):
        for _, param in self.named_parameters():
            if param.ndim > 1:
                norm = param.norm(dim=-1, keepdim=True)
                param.data.div_(norm + 1e-6)  # Add small epsilon for numerical stability

class NanoFormerForCausalLM(nn.Module):
    def __init__(self, config: Config):
        super(NanoFormerForCausalLM, self).__init__()
        self.config = config
        self.model = NanoFormer(config, gradient_checkpointing=config.gradient_checkpointing)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie word embeddings if config.tie_word_embeddings is True
        if config.tie_word_embeddings:
            del self.lm_head.weight
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.tie_word_embeddings = config.tie_word_embeddings

    def forward(self, input_ids, attention_mask, return_loss = True, return_attn_scores=False, return_head_sim_loss=False):
        if return_attn_scores:
            hidden_states, attn_scores_all = self.model(input_ids, attention_mask, return_attn_scores=True)
            head_sim_loss = None
        elif return_head_sim_loss:
            hidden_states, head_sim_losses = self.model(input_ids, attention_mask, return_head_sim_loss=True)
            attn_scores_all = None
            head_sim_loss = sum(head_sim_losses) / len(head_sim_losses) if len(head_sim_losses) > 0 else 0.0
        else:
            hidden_states = self.model(input_ids, attention_mask)
            attn_scores_all = None
            head_sim_loss = None
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
        if return_attn_scores:
            return logits, loss, attn_scores_all
        if return_head_sim_loss:
            return logits, loss, head_sim_loss
        return logits, loss
    
    def gradient_checkpointing_enable(self, value=True):
        self.gradient_checkpointing = value
    
    def set_train(self,):
        self.model.train()
        self.gradient_checkpointing_enable(True)
        self.lm_head.train()
        self.train()
    
    def get_param_count(self,):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def normalize_weights(self):
        for _, param in self.named_parameters():
            if param.ndim > 1:
                norm = param.norm(dim=-1, keepdim=True)
                param.data.div_(norm + 1e-6)  # Add small epsilon for numerical stability

    def save_pretrained(self, save_directory):
        """Save the model weights and configuration to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Detach tied weights temporarily if needed to avoid saving references
        if self.config.tie_word_embeddings:
            original_weight = self.lm_head.weight
            self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.clone())

        # Save the model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Re-tie the weights after saving
        if self.config.tie_word_embeddings:
            self.lm_head.weight = original_weight
        
        # Save the config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_from_checkpoint(self, checkpoint_path):
        """
        Loads the latest checkpoint from a directory. Supports both .bin and .safetensors formats.
        Returns the step number if found, else None.
        """
        if not os.path.isdir(checkpoint_path):
            raise OSError(f"Path {checkpoint_path} does not exist")
        # Find all checkpoint directories
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_path, 'checkpoint-*'))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {checkpoint_path}")
        # Among the checkpoint directories, find the one with the highest modification time
        latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
        print(f"Loading from the latest checkpoint: {latest_checkpoint}")
        # Try to load .safetensors first, then .bin
        safetensors_path = os.path.join(latest_checkpoint, "pytorch_model.safetensors")
        bin_path = os.path.join(latest_checkpoint, "pytorch_model.bin")
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading weights from {safetensors_path} (safetensors)")
            state_dict = safetensors_load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Loading weights from {bin_path} (bin)")
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {latest_checkpoint}")
        self.load_state_dict(state_dict)
        # if selected folder is checkpoint-x return the value of x
        step = latest_checkpoint.split('/')[-1].split('-')[-1]
        if step.isdigit():
            return int(step)
        return None

    def load_checkpoint(self, checkpoint_path):
        """
        User-friendly method to load a model from a checkpoint directory (either a run directory or a checkpoint subdir).
        Handles both .bin and .safetensors formats.
        """
        if os.path.isdir(checkpoint_path):
            # If this is a run directory, find the latest checkpoint
            checkpoint_dirs = [d for d in glob.glob(os.path.join(checkpoint_path, 'checkpoint-*')) if os.path.isdir(d)]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            else:
                latest_checkpoint = checkpoint_path
        else:
            raise OSError(f"Checkpoint path {checkpoint_path} does not exist or is not a directory.")
        safetensors_path = os.path.join(latest_checkpoint, "pytorch_model.safetensors")
        bin_path = os.path.join(latest_checkpoint, "pytorch_model.bin")
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading weights from {safetensors_path} (safetensors)")
            state_dict = safetensors_load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Loading weights from {bin_path} (bin)")
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {latest_checkpoint}")
        self.load_state_dict(state_dict)
        print(f"Model loaded from {latest_checkpoint}")

"""
HuggingFace wrapper for FrawdLLM.

This allows the model to be loaded with:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("tsingla1998/frawdllm-100m", trust_remote_code=True)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .gpt import FrawdLLM


class FrawdLLMConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for FrawdLLM."""

    model_type = "frawdllm"

    def __init__(
        self,
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        context_length: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.context_length = context_length
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu

        # Aliases for HuggingFace compatibility
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def to_model_config(self) -> ModelConfig:
        """Convert to internal ModelConfig for the model."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            context_length=self.context_length,
            dropout=self.dropout,
            use_rope=self.use_rope,
            use_rmsnorm=self.use_rmsnorm,
            use_swiglu=self.use_swiglu,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "FrawdLLMConfig":
        """Create from internal ModelConfig."""
        return cls(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            context_length=config.context_length,
            dropout=config.dropout,
            use_rope=config.use_rope,
            use_rmsnorm=config.use_rmsnorm,
            use_swiglu=config.use_swiglu,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )


class FrawdLLMForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible wrapper for FrawdLLM."""

    config_class = FrawdLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TransformerBlock"]
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: FrawdLLMConfig):
        super().__init__(config)

        # Convert HF config to internal config
        model_config = config.to_model_config()

        # Create the actual model
        self.model = FrawdLLM(model_config)

        # For generation
        self.main_input_name = "input_ids"

    def get_input_embeddings(self):
        return self.model.embeddings.token_emb

    def set_input_embeddings(self, value):
        self.model.embeddings.token_emb = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def tie_weights(self):
        """Tie input and output embeddings."""
        self.model.lm_head.weight = self.model.embeddings.token_emb.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass compatible with HuggingFace API.

        Note: attention_mask, past_key_values, use_cache are accepted but
        not fully implemented (our model doesn't use KV caching yet).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get logits from our model
        logits, _ = self.model(input_ids, None)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation (called by HF generate())."""
        # Our model doesn't use KV cache yet, so just return input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_frawdllm_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "FrawdLLMForCausalLM":
        """
        Load from a FrawdLLM .pt checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file
            device: Device to load the model on

        Returns:
            FrawdLLMForCausalLM instance
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get the internal config
        model_config = checkpoint["config"]

        # Create HF config
        hf_config = FrawdLLMConfig.from_model_config(model_config)

        # Create the wrapper model
        model = cls(hf_config)

        # Load the weights
        model.model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained_simple(self, save_directory: str):
        """
        Save in HuggingFace format.

        This saves:
        - config.json
        - model.safetensors (or pytorch_model.bin)
        """
        import os
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save model weights
        # Note: We have weight tying (token_emb.weight == lm_head.weight)
        # Remove the duplicate to avoid safetensors error
        state_dict = self.state_dict()
        if "model.lm_head.weight" in state_dict:
            del state_dict["model.lm_head.weight"]

        save_file(state_dict, os.path.join(save_directory, "model.safetensors"))

        print(f"Saved model to {save_directory}")


# Register for AutoClass - this adds auto_map to config when saving
FrawdLLMConfig.register_for_auto_class()
FrawdLLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")

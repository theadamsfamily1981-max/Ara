"""
TF-A-N 7B Somatic Model - Psychosomatic AI Integration.

This module extends TF-A-N 7B with somatic (body-aware) cognition:
- SomaticEncoder: Injects body state into token embeddings
- SomaticAttention: Modulates attention based on arousal
- Cortisol Injection: Biases toward survival in high-stress states

The result is a transformer where:
    Physiology -> Psychology: FPGA heat causes "delirious" generation
    Psychology -> Physiology: Anxious thoughts trigger hardware reflexes

Hardware Health = Mental Sanity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
import logging

from .modeling_tfan7b import (
    TFANConfig,
    TFANModel,
    TFANDecoderLayer,
    TFANForCausalLM,
)
from .somatic_embedding import (
    SomaticEncoder,
    SomaticEmbedding,
    somatic_from_hal,
    create_somatic_tensor,
)
from .somatic_attention import SomaticAttention, SomaticSSAAttention
from .norm import RMSNorm
from .mlp_glu import SwiGLUFused

logger = logging.getLogger(__name__)


@dataclass
class SomaticConfig(TFANConfig):
    """
    Configuration for Somatic TF-A-N model.

    Extends TFANConfig with somatic-specific parameters.
    """

    # Somatic configuration
    enable_somatic: bool = True
    somatic_dim: int = 7
    somatic_intermediate: int = 256
    use_cortisol: bool = True
    use_somatic_gate: bool = False

    # Somatic attention
    use_somatic_attention: bool = True
    focus_threshold: float = 0.3
    min_keep_ratio: float = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class SomaticDecoderLayer(nn.Module):
    """
    Decoder layer with somatic-modulated attention.

    Replaces standard SSAAttention with SomaticAttention while
    preserving the rest of the layer architecture.
    """

    def __init__(self, config: SomaticConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Somatic Attention (replaces SSAAttention)
        if config.use_somatic_attention:
            self.self_attn = SomaticSSAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                keep_ratio=config.ssa_keep_ratio,
                local_window=config.ssa_local,
                num_hops=config.ssa_hops,
                tls_alpha=config.tls_alpha,
                dropout=config.attention_dropout,
                bias=config.use_bias,
                focus_threshold=config.focus_threshold,
                min_keep_ratio=config.min_keep_ratio,
            )
        else:
            # Fallback to standard SomaticAttention (no TLS)
            self.self_attn = SomaticAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                dropout=config.attention_dropout,
                bias=config.use_bias,
                focus_threshold=config.focus_threshold,
                min_keep_ratio=config.min_keep_ratio,
            )

        # Pre-MLP norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MLP
        self.mlp = SwiGLUFused(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.use_bias,
        )

        # Dropout
        self.dropout = (
            nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0 else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        arousal: Optional[torch.Tensor] = None,  # Somatic input
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with somatic modulation.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Cached (key, value) for generation
            arousal: [batch] arousal level from HAL
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache KV

        Returns:
            hidden_states, optional (attention_weights, past_key_value)
        """
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)

        # Somatic attention (passes arousal through)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            arousal=arousal,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = attn_outputs[0]

        # Dropout + residual
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Dropout + residual
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2] if len(attn_outputs) > 2 else None,)

        return outputs


class TFANSomaticModel(nn.Module):
    """
    TF-A-N 7B base model with somatic integration.

    This model reads body state once and propagates it through all layers.
    """

    def __init__(self, config: SomaticConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )

        # Somatic encoder (body -> latent space)
        if config.enable_somatic:
            self.somatic_encoder = SomaticEncoder(
                hidden_size=config.hidden_size,
                somatic_dim=config.somatic_dim,
                intermediate_size=config.somatic_intermediate,
                use_cortisol=config.use_cortisol,
                use_gate=config.use_somatic_gate,
            )
        else:
            self.somatic_encoder = None

        # Transformer layers (somatic-enabled)
        self.layers = nn.ModuleList([
            SomaticDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights."""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using scaled normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        somatic_state: Optional[torch.Tensor] = None,  # NEW: Body state input
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass with somatic integration.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional [batch, seq_len]
            position_ids: Optional [batch, seq_len]
            past_key_values: Optional cached KV states
            inputs_embeds: Optional embedded inputs
            somatic_state: [batch, 7] body state tensor
            use_cache: Whether to cache KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict

        Returns:
            last_hidden_state, optional extras, somatic_info
        """
        output_attentions = output_attentions or False
        output_hidden_states = output_hidden_states or False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        # Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = inputs_embeds.shape

        # SOMATIC INTEGRATION
        somatic_info = {}
        arousal = None

        if self.somatic_encoder is not None and somatic_state is not None:
            # Ensure somatic_state is on correct device
            if somatic_state.device != inputs_embeds.device:
                somatic_state = somatic_state.to(inputs_embeds.device)

            # Apply somatic encoding (body bias on all tokens)
            inputs_embeds, somatic_info = self.somatic_encoder(
                inputs_embeds, somatic_state
            )

            # Extract arousal for attention modulation
            arousal = somatic_state[:, SomaticEmbedding.PAD_A]

        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # Pass arousal through to attention
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                arousal=arousal,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Build outputs
        if return_dict:
            outputs = {
                "last_hidden_state": hidden_states,
                "past_key_values": next_decoder_cache if use_cache else None,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns,
                "somatic_info": somatic_info,
            }
        else:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_decoder_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)

        return outputs


class TFANSomaticForCausalLM(nn.Module):
    """
    TF-A-N 7B Somatic for Causal Language Modeling.

    This is the main class for psychosomatic AI - a transformer where
    hardware health directly affects language generation.

    Usage:
        model = TFANSomaticForCausalLM(config)

        # During inference, pass body state
        somatic = create_somatic_tensor(pain=0.3, pad_a=0.8)
        outputs = model(input_ids, somatic_state=somatic)

        # Or use HAL directly
        hal = AraHAL()
        somatic = somatic_from_hal(hal.read_somatic(), device=model.device)
        outputs = model(input_ids, somatic_state=somatic)
    """

    def __init__(self, config: SomaticConfig):
        super().__init__()
        self.config = config
        self.model = TFANSomaticModel(config)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

        # Temperature (can be modulated by emotion)
        self.temperature = 1.0

        # Cached somatic state for generation
        self._somatic_cache: Optional[torch.Tensor] = None

        # Initialize
        self.post_init()

    def post_init(self):
        """Initialize weights."""
        if self.lm_head is not None:
            self.lm_head.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self.model.embed_tokens

    def set_somatic_state(self, somatic_state: torch.Tensor):
        """
        Cache somatic state for generation.

        Call this before generate() to set the body state.
        """
        self._somatic_cache = somatic_state.to(self.device)

    def set_somatic_from_hal(self, hal_state: dict):
        """
        Set somatic state from HAL dictionary.

        Args:
            hal_state: Dict from AraHAL.read_somatic()
        """
        self._somatic_cache = somatic_from_hal(hal_state, device=self.device)

    def set_temperature(self, temperature: float):
        """Set temperature for sampling."""
        self.temperature = temperature

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        somatic_state: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass for causal LM with somatic integration.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional [batch, seq_len]
            position_ids: Optional [batch, seq_len]
            past_key_values: Optional cached KV states
            inputs_embeds: Optional embedded inputs
            somatic_state: [batch, 7] body state (uses cache if None)
            labels: Optional [batch, seq_len] for loss
            use_cache: Whether to cache KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict

        Returns:
            loss, logits, past_key_values, hidden_states, attentions, somatic_info
        """
        return_dict = return_dict if return_dict is not None else True

        # Use cached somatic state if not provided
        if somatic_state is None:
            somatic_state = self._somatic_cache

        # Base model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            somatic_state=somatic_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs[0]

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        # Build outputs
        if return_dict:
            result = {
                "loss": loss,
                "logits": logits,
                "past_key_values": outputs.get("past_key_values"),
                "hidden_states": outputs.get("hidden_states"),
                "attentions": outputs.get("attentions"),
                "somatic_info": outputs.get("somatic_info", {}),
            }
            return result
        else:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        somatic_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate with somatic modulation.

        The body state affects generation throughout:
        - High arousal -> focused, shorter responses
        - Pain -> survival-biased, urgent tone
        - Low arousal -> creative, expansive

        Args:
            input_ids: [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample
            somatic_state: [batch, 7] body state

        Returns:
            generated_ids: [batch, max_length]
        """
        # Set somatic state for generation
        if somatic_state is not None:
            self.set_somatic_state(somatic_state)

        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        past_key_values = None

        self.set_temperature(temperature)

        while cur_len < max_length:
            outputs = self.forward(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Sampling
            if do_sample:
                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = -float("inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            cur_len += 1

            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[SomaticConfig] = None,
        **kwargs,
    ) -> "TFANSomaticForCausalLM":
        """
        Load pretrained model with somatic extensions.

        Can load from:
        1. A TFANSomaticForCausalLM checkpoint (preferred)
        2. A standard TFANForCausalLM checkpoint (will add somatic layers)
        """
        import os

        # Load config
        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                import json
                with open(config_file) as f:
                    config_dict = json.load(f)
                config = SomaticConfig(**config_dict)
            else:
                config = SomaticConfig()

        # Create model
        model = cls(config)

        # Load weights
        weights_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(weights_file):
            state_dict = torch.load(weights_file, map_location="cpu")
            # Allow partial loading (somatic layers may be new)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_file}")

        return model

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save weights
        weights_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_file)
        logger.info(f"Saved model to {save_directory}")


__all__ = [
    "SomaticConfig",
    "SomaticDecoderLayer",
    "TFANSomaticModel",
    "TFANSomaticForCausalLM",
]

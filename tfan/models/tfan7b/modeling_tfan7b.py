"""
TF-A-N 7B: Transformer with Formal Alignment and Neuromodulation (7 billion parameters).

Decoder-only transformer with:
- Selective Sparse Attention (SSA) with O(N log N) complexity
- Rotary Positional Embeddings (RoPE)
- RMSNorm and SwiGLU MLP
- Grouped Query Attention (GQA)
- Topology and Emotion heads for TF-A-N integration

Compatible with HuggingFace Transformers API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
import json
import math

# Import local modules
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .norm import RMSNorm
from .mlp_glu import SwiGLUFused
from .attention_sparse import SSAAttention
from .topo_head import TopologyHook
from .emotion_head import EmotionHead


@dataclass
class TFANConfig:
    """
    Configuration class for TF-A-N 7B model.

    Compatible with HuggingFace config format.
    """

    # Model architecture
    model_type: str = "tfan7b"
    vocab_size: int = 32768
    hidden_size: int = 4096
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 14336
    ffn_mult: float = 3.5

    # Positional embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None

    # Normalization
    rms_norm_eps: float = 1e-6

    # Architecture details
    tie_word_embeddings: bool = True
    use_bias: bool = False
    activation: str = "swiglu"

    # SSA (Selective Sparse Attention) config
    attention_impl: str = "ssa_radial_v1"
    ssa_keep_ratio: float = 0.33
    ssa_local: int = 128
    ssa_hops: int = 2
    tls_alpha: float = 0.7

    # Training
    initializer_range: float = 0.02
    use_cache: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Tokenizer IDs
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Precision
    torch_dtype: str = "bfloat16"

    # TF-A-N specific
    enable_topology_head: bool = False
    enable_emotion_head: bool = False
    lambda_topo: float = 0.1

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Compute intermediate_size if not provided
        if self.intermediate_size is None:
            self.intermediate_size = int(self.ffn_mult * self.hidden_size)
            # Round to multiple of 128
            self.intermediate_size = ((self.intermediate_size + 127) // 128) * 128

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TFANConfig":
        """Load config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "TFANConfig":
        """Load config from JSON file."""
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save_pretrained(self, save_directory: str):
        """Save config to directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TFANDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with SSA and SwiGLU.

    Architecture:
        x -> [RMSNorm -> SSA] -> residual -> [RMSNorm -> SwiGLU] -> residual

    Args:
        config: TFANConfig
        layer_idx: Layer index (for debugging/hooks)
    """

    def __init__(self, config: TFANConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = SSAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            keep_ratio=config.ssa_keep_ratio,
            local_window=config.ssa_local,
            num_hops=config.ssa_hops,
            tls_alpha=config.tls_alpha,
            dropout=config.attention_dropout,
            bias=config.use_bias,
            use_flash=False,  # Set to True if flash-attn available and desired
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

        # Optional dropout
        self.dropout = (
            nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0 else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Cached (key, value) for generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache KV

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            optional: (attention_weights, present_key_value)
        """
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
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


class TFANModel(nn.Module):
    """
    TF-A-N 7B base model (decoder-only transformer).

    Outputs raw hidden states without LM head.

    Args:
        config: TFANConfig
    """

    def __init__(self, config: TFANConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TFANDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=(
                config.rope_scaling.get("factor", 1.0)
                if config.rope_scaling
                else 1.0
            ),
        )

        # TF-A-N heads (optional)
        self.topology_head = (
            TopologyHook(
                hidden_size=config.hidden_size,
                lambda_topo=config.lambda_topo,
            )
            if config.enable_topology_head
            else None
        )

        self.emotion_head = (
            EmotionHead(hidden_size=config.hidden_size)
            if config.enable_emotion_head
            else None
        )

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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass for TF-A-N model.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional [batch, seq_len]
            position_ids: Optional [batch, seq_len]
            past_key_values: Optional cached KV states
            inputs_embeds: Optional embedded inputs
            use_cache: Whether to cache KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict (vs tuple)

        Returns:
            last_hidden_state: [batch, seq_len, hidden_size]
            optional: past_key_values, hidden_states, attentions
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions if hasattr(self.config, "output_attentions") else False
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states if hasattr(self.config, "output_hidden_states") else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        # Retrieve input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = inputs_embeds.shape

        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # RoPE embeddings (computed but applied in attention layers)
        # For now, we'll apply RoPE inside the attention module

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

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
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

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Build outputs
        if return_dict:
            outputs = {
                "last_hidden_state": hidden_states,
                "past_key_values": next_decoder_cache if use_cache else None,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns,
            }
        else:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_decoder_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)

        # TF-A-N heads (optional)
        if self.topology_head is not None:
            topo_outputs = self.topology_head(hidden_states, compute_loss=False)
            if return_dict:
                outputs["topology_landscapes"] = topo_outputs["landscapes"]

        if self.emotion_head is not None:
            emotion_outputs = self.emotion_head(hidden_states, attention_mask)
            if return_dict:
                outputs["emotion"] = emotion_outputs

        return outputs


class TFANForCausalLM(nn.Module):
    """
    TF-A-N 7B for Causal Language Modeling.

    Adds LM head on top of base model for next-token prediction.

    Args:
        config: TFANConfig
    """

    def __init__(self, config: TFANConfig):
        super().__init__()
        self.config = config
        self.model = TFANModel(config)

        # LM head (tied with embeddings if config.tie_word_embeddings)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

        # Temperature for sampling (can be modulated by FDT/emotion)
        self.temperature = 1.0

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights."""
        if self.lm_head is not None:
            self.lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self.model.embed_tokens

    def set_output_embeddings(self, new_embeddings):
        if self.lm_head is not None:
            self.lm_head = new_embeddings
        else:
            self.model.embed_tokens = new_embeddings

    def set_temperature(self, temperature: float):
        """Set temperature for sampling (used by FDT/emotion controller)."""
        self.temperature = temperature

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass for causal LM.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional [batch, seq_len]
            position_ids: Optional [batch, seq_len]
            past_key_values: Optional cached KV states
            inputs_embeds: Optional embedded inputs
            labels: Optional [batch, seq_len] for loss computation
            use_cache: Whether to cache KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict

        Returns:
            loss: Optional language modeling loss
            logits: [batch, seq_len, vocab_size]
            optional: past_key_values, hidden_states, attentions
        """
        return_dict = return_dict if return_dict is not None else True

        # Base model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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
            # Tied embeddings
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        # Build outputs
        if return_dict:
            result = {
                "loss": loss,
                "logits": logits,
                "past_key_values": outputs.get("past_key_values"),
                "hidden_states": outputs.get("hidden_states"),
                "attentions": outputs.get("attentions"),
            }
            # Add TF-A-N specific outputs
            if "topology_landscapes" in outputs:
                result["topology_landscapes"] = outputs["topology_landscapes"]
            if "emotion" in outputs:
                result["emotion"] = outputs["emotion"]
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
        **kwargs,
    ) -> torch.LongTensor:
        """
        Simple generation method.

        Args:
            input_ids: [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample (vs greedy)

        Returns:
            generated_ids: [batch, max_length]
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        past_key_values = None

        # Set temperature
        self.set_temperature(temperature)

        while cur_len < max_length:
            # Forward pass
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

            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            cur_len += 1

            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[TFANConfig] = None,
        **kwargs,
    ) -> "TFANForCausalLM":
        """
        Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path to model or HF model ID
            config: Optional config (otherwise load from path)

        Returns:
            model: TFANForCausalLM instance
        """
        import os

        # Load config
        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            config = TFANConfig.from_json_file(config_file)

        # Create model
        model = cls(config)

        # Load weights
        weights_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(weights_file):
            state_dict = torch.load(weights_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_file}")

        return model

    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.

        Args:
            save_directory: Directory to save to
        """
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save weights
        weights_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_file)
        print(f"Saved model to {save_directory}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model.

    Args:
        model: PyTorch model

    Returns:
        dict with total, trainable, non_trainable counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total_millions": total_params / 1e6,
        "total_billions": total_params / 1e9,
    }


__all__ = [
    "TFANConfig",
    "TFANModel",
    "TFANForCausalLM",
    "TFANDecoderLayer",
    "count_parameters",
]

"""TF-A-N 7B model with sparse attention and formal alignment."""

from .modeling_tfan7b import (
    TFANConfig,
    TFANForCausalLM,
    TFANModel,
)

from .modeling_tfan_somatic import (
    SomaticConfig,
    TFANSomaticForCausalLM,
    TFANSomaticModel,
    SomaticDecoderLayer,
)

from .somatic_attention import (
    SomaticAttention,
    SomaticSSAAttention,
)

from .somatic_embedding import (
    SomaticEmbedding,
    SomaticEncoder,
    CortisolInjector,
    create_somatic_tensor,
    somatic_from_hal,
)

__all__ = [
    # Base TF-A-N
    "TFANConfig",
    "TFANForCausalLM",
    "TFANModel",
    # Somatic TF-A-N
    "SomaticConfig",
    "TFANSomaticForCausalLM",
    "TFANSomaticModel",
    "SomaticDecoderLayer",
    # Somatic components
    "SomaticAttention",
    "SomaticSSAAttention",
    "SomaticEmbedding",
    "SomaticEncoder",
    "CortisolInjector",
    "create_somatic_tensor",
    "somatic_from_hal",
]

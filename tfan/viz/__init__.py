"""
Visualization and streaming infrastructure for TF-A-N.

Provides real-time WebSocket streaming of:
- Persistence diagrams (PD)
- Attention sparsity patterns
- Topological landmark selection
- FDT/EPR-CV metrics
"""

__all__ = [
    'VizStream',
    'encode_pd',
    'encode_attention_matrix',
    'encode_sparsity_metrics',
    'encode_fdt_state',
]


def __getattr__(name):
    """Lazy import to avoid requiring torch for lightweight imports."""
    if name == 'VizStream':
        from .stream import VizStream
        return VizStream
    elif name == 'encode_pd':
        from .encoders import encode_pd
        return encode_pd
    elif name == 'encode_attention_matrix':
        from .encoders import encode_attention_matrix
        return encode_attention_matrix
    elif name == 'encode_sparsity_metrics':
        from .encoders import encode_sparsity_metrics
        return encode_sparsity_metrics
    elif name == 'encode_fdt_state':
        from .encoders import encode_fdt_state
        return encode_fdt_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

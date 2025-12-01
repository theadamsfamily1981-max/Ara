"""
TF-A-N Loss Functions and Regularizers

Provides specialized loss functions for hyperbolic and topology-aware training.
Requires PyTorch for full functionality.
"""

# Check torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .hyperbolic_reg import (
        HyperbolicSparsityRegularizer,
        poincare_boundary_penalty,
        curvature_stability_loss,
        hyperbolic_distance,
        project_to_poincare,
        compute_l3_valence_from_geometry,
    )

    __all__ = [
        "HyperbolicSparsityRegularizer",
        "poincare_boundary_penalty",
        "curvature_stability_loss",
        "hyperbolic_distance",
        "project_to_poincare",
        "compute_l3_valence_from_geometry",
        "TORCH_AVAILABLE",
    ]
else:
    # Stubs when torch not available
    def poincare_boundary_penalty(*args, **kwargs):
        raise ImportError("PyTorch required for hyperbolic_reg")

    def curvature_stability_loss(*args, **kwargs):
        raise ImportError("PyTorch required for hyperbolic_reg")

    def hyperbolic_distance(*args, **kwargs):
        raise ImportError("PyTorch required for hyperbolic_reg")

    def project_to_poincare(*args, **kwargs):
        raise ImportError("PyTorch required for hyperbolic_reg")

    def compute_l3_valence_from_geometry(*args, **kwargs):
        return 0.0  # Safe default

    class HyperbolicSparsityRegularizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for HyperbolicSparsityRegularizer")

    __all__ = [
        "HyperbolicSparsityRegularizer",
        "poincare_boundary_penalty",
        "curvature_stability_loss",
        "hyperbolic_distance",
        "project_to_poincare",
        "compute_l3_valence_from_geometry",
        "TORCH_AVAILABLE",
    ]

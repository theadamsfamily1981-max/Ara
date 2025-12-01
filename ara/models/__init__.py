"""
ARA Models Package

Provides unified access to all model architectures:
- TFAN: Transformer with Formal Alignment (7B params)
- SNN: Spiking Neural Networks (population-based)
- TGSFN: Thermodynamic Gated SNN

Model Architecture Config:
    from ara.models import TFANConfig

System/Training Config:
    from ara.models import SystemConfig

Full Model:
    from ara.models import TFANForCausalLM, TFANModel
"""

import sys
from pathlib import Path

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import model architecture config from src/models/tfan
try:
    from src.models.tfan.config import TFANConfig
    from src.models.tfan.modeling_tfan import TFANModel, TFANForCausalLM
except ImportError:
    # Fallback to direct import
    TFANConfig = None
    TFANModel = None
    TFANForCausalLM = None

# Import system config from tfan/config.py (renamed to avoid collision)
try:
    from tfan.config import TFANConfig as SystemConfig
except ImportError:
    SystemConfig = None

# Import SNN components
try:
    from src.models.tfan.snn import (
        SNNModel,
        SNNConfig,
    )
except ImportError:
    SNNModel = None
    SNNConfig = None

# Import legacy components for backward compatibility
try:
    from tfan.models.tfan7b import (
        TFANForCausalLM as TFANForCausalLMLegacy,
    )
except ImportError:
    TFANForCausalLMLegacy = None

__all__ = [
    # Model Architecture
    "TFANConfig",
    "TFANModel",
    "TFANForCausalLM",

    # System Config
    "SystemConfig",

    # SNN
    "SNNModel",
    "SNNConfig",

    # Legacy (deprecated)
    "TFANForCausalLMLegacy",
]

"""
ARA Agents Package

Provides unified access to all agent implementations:
- HRRLAgent: Homeostatic Reinforcement Regulated Learning (canonical)
- TGSFNSubstrate: Thermodynamic Gated SNN substrate
- Criticality control, antifragile loops, etc.

Example:
    from ara.agents import HRRLAgent, create_agent, HRRLConfig

    # Quick creation (if hrrl_agent available)
    if create_agent is not None:
        agent = create_agent(obs_dim=64, action_dim=8)

    # With config
    if HRRLConfig is not None:
        config = HRRLConfig(...)
        agent = HRRLAgent(config)
"""

import sys
from pathlib import Path

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Initialize all exports as None (will be populated if available)
HRRLConfig = None
L1Config = None
L2Config = None
L3Config = None
L4Config = None
TGSFNConfig = None
TrainingConfig = None
HRRLAgent = None
create_agent = None
PolicyNetwork = None
HomeostatL1 = None
HyperbolicAppraisalL2 = None
GatingControllerL3 = None
ReplayBuffer = None
PersonalizationModule = None
TGSFNSubstrate = None
TGSFNLayer = None
TGSFNLoss = None
TGSFNState = None
create_tgsfn_substrate = None
CriticalityController = None
CriticalityConfig = None
AvalancheAnalyzer = None
predict_finite_size_alpha = None
AntifragileLoop = None
JacobianMonitor = None
FixedPoint16 = None
FastLearnableTimeWarping = None
OnlineLoop = None
SleepLoop = None
DualLoopTrainer = None
TGSFNCriticalityNetwork = None
LIFLayer = None
tfan_training_loop = None
tfan_snn_policy = None

_HRRL_AVAILABLE = False
_TFAN_AGENT_AVAILABLE = False
_TGSFN_ARA_AVAILABLE = False

# Import from canonical hrrl_agent package - piece by piece to be resilient
try:
    from hrrl_agent import HRRLConfig as _HRRLConfig
    HRRLConfig = _HRRLConfig
    _HRRL_AVAILABLE = True
except ImportError:
    pass

try:
    from hrrl_agent import L1Config as _L1Config
    L1Config = _L1Config
except ImportError:
    pass

try:
    from hrrl_agent import L2Config as _L2Config
    L2Config = _L2Config
except ImportError:
    pass

try:
    from hrrl_agent import L3Config as _L3Config
    L3Config = _L3Config
except ImportError:
    pass

try:
    from hrrl_agent import L4Config as _L4Config
    L4Config = _L4Config
except ImportError:
    pass

try:
    from hrrl_agent import TGSFNConfig as _TGSFNConfig
    TGSFNConfig = _TGSFNConfig
except ImportError:
    pass

try:
    from hrrl_agent import TrainingConfig as _TrainingConfig
    TrainingConfig = _TrainingConfig
except ImportError:
    pass

try:
    from hrrl_agent import HRRLAgent as _HRRLAgent
    HRRLAgent = _HRRLAgent
except ImportError:
    pass

try:
    from hrrl_agent import create_agent as _create_agent
    create_agent = _create_agent
except ImportError:
    pass

try:
    from hrrl_agent import PolicyNetwork as _PolicyNetwork
    PolicyNetwork = _PolicyNetwork
except ImportError:
    pass

try:
    from hrrl_agent import HomeostatL1 as _HomeostatL1
    HomeostatL1 = _HomeostatL1
except ImportError:
    pass

try:
    from hrrl_agent import HyperbolicAppraisalL2 as _HyperbolicAppraisalL2
    HyperbolicAppraisalL2 = _HyperbolicAppraisalL2
except ImportError:
    pass

try:
    from hrrl_agent import GatingControllerL3 as _GatingControllerL3
    GatingControllerL3 = _GatingControllerL3
except ImportError:
    pass

try:
    from hrrl_agent import ReplayBuffer as _ReplayBuffer
    ReplayBuffer = _ReplayBuffer
except ImportError:
    pass

try:
    from hrrl_agent import PersonalizationModule as _PersonalizationModule
    PersonalizationModule = _PersonalizationModule
except ImportError:
    pass

try:
    from hrrl_agent import TGSFNSubstrate as _TGSFNSubstrate
    TGSFNSubstrate = _TGSFNSubstrate
except ImportError:
    pass

try:
    from hrrl_agent import TGSFNLayer as _TGSFNLayer
    TGSFNLayer = _TGSFNLayer
except ImportError:
    pass

try:
    from hrrl_agent import TGSFNLoss as _TGSFNLoss
    TGSFNLoss = _TGSFNLoss
except ImportError:
    pass

try:
    from hrrl_agent import TGSFNState as _TGSFNState
    TGSFNState = _TGSFNState
except ImportError:
    pass

try:
    from hrrl_agent import create_tgsfn_substrate as _create_tgsfn_substrate
    create_tgsfn_substrate = _create_tgsfn_substrate
except ImportError:
    pass

try:
    from hrrl_agent import CriticalityController as _CriticalityController
    CriticalityController = _CriticalityController
except ImportError:
    pass

try:
    from hrrl_agent import CriticalityConfig as _CriticalityConfig
    CriticalityConfig = _CriticalityConfig
except ImportError:
    pass

try:
    from hrrl_agent import AvalancheAnalyzer as _AvalancheAnalyzer
    AvalancheAnalyzer = _AvalancheAnalyzer
except ImportError:
    pass

try:
    from hrrl_agent import predict_finite_size_alpha as _predict_finite_size_alpha
    predict_finite_size_alpha = _predict_finite_size_alpha
except ImportError:
    pass

try:
    from hrrl_agent import AntifragileLoop as _AntifragileLoop
    AntifragileLoop = _AntifragileLoop
except ImportError:
    pass

try:
    from hrrl_agent import JacobianMonitor as _JacobianMonitor
    JacobianMonitor = _JacobianMonitor
except ImportError:
    pass

try:
    from hrrl_agent import FixedPoint16 as _FixedPoint16
    FixedPoint16 = _FixedPoint16
except ImportError:
    pass

try:
    from hrrl_agent import FastLearnableTimeWarping as _FastLearnableTimeWarping
    FastLearnableTimeWarping = _FastLearnableTimeWarping
except ImportError:
    pass

try:
    from hrrl_agent import OnlineLoop as _OnlineLoop
    OnlineLoop = _OnlineLoop
except ImportError:
    pass

try:
    from hrrl_agent import SleepLoop as _SleepLoop
    SleepLoop = _SleepLoop
except ImportError:
    pass

try:
    from hrrl_agent import DualLoopTrainer as _DualLoopTrainer
    DualLoopTrainer = _DualLoopTrainer
except ImportError:
    pass

# Import from tfan_agent (to be integrated)
try:
    from tfan_agent import training_loop as _tfan_training_loop
    tfan_training_loop = _tfan_training_loop
    _TFAN_AGENT_AVAILABLE = True
except ImportError:
    pass

try:
    from tfan_agent import snn_policy as _tfan_snn_policy
    tfan_snn_policy = _tfan_snn_policy
except ImportError:
    pass

# Import from tgsfn-ara (specialized criticality)
try:
    # Add tgsfn-ara to path
    tgsfn_path = _root / "tgsfn-ara"
    if str(tgsfn_path) not in sys.path:
        sys.path.insert(0, str(tgsfn_path))
    from tgsfn_core.snn_model import TGSFNNetwork as _TGSFNCriticalityNetwork
    TGSFNCriticalityNetwork = _TGSFNCriticalityNetwork
    _TGSFN_ARA_AVAILABLE = True
except ImportError:
    pass

try:
    tgsfn_path = _root / "tgsfn-ara"
    if str(tgsfn_path) not in sys.path:
        sys.path.insert(0, str(tgsfn_path))
    from tgsfn_core.snn_model import LIFLayer as _LIFLayer
    LIFLayer = _LIFLayer
except ImportError:
    pass

__all__ = [
    # Config
    "HRRLConfig",
    "L1Config",
    "L2Config",
    "L3Config",
    "L4Config",
    "TGSFNConfig",
    "TrainingConfig",

    # Core Agent
    "HRRLAgent",
    "create_agent",
    "PolicyNetwork",

    # L1-L4 Layers
    "HomeostatL1",
    "HyperbolicAppraisalL2",
    "GatingControllerL3",
    "ReplayBuffer",
    "PersonalizationModule",

    # TGSFN Substrate
    "TGSFNSubstrate",
    "TGSFNLayer",
    "TGSFNLoss",
    "TGSFNState",
    "create_tgsfn_substrate",

    # Criticality
    "CriticalityController",
    "CriticalityConfig",
    "AvalancheAnalyzer",
    "predict_finite_size_alpha",
    "TGSFNCriticalityNetwork",
    "LIFLayer",

    # Antifragile
    "AntifragileLoop",
    "JacobianMonitor",

    # Hardware
    "FixedPoint16",
    "FastLearnableTimeWarping",

    # Training
    "OnlineLoop",
    "SleepLoop",
    "DualLoopTrainer",

    # Availability flags
    "_HRRL_AVAILABLE",
    "_TFAN_AGENT_AVAILABLE",
    "_TGSFN_ARA_AVAILABLE",
]

"""
TF-A-N Agent - AEPO Tool-Use Controller

Entropy-regularized policy for adaptive tool calling:
- Learns when to call tools (TTW/PGU/HTTP) vs skip
- Minimizes tool calls while maintaining reward
- Prevents entropy collapse with adaptive regularization

Hard gates:
- Tool-call count −50% vs baseline
- Reward within −1% of baseline
- Stable entropy curve (no collapse) across 10 seeds
"""

from .aepo import AEPO, AEPOConfig
from .env_tools import ToolEnv, ToolAction
from .replay_buffer import ReplayBuffer, Transition

__all__ = [
    'AEPO',
    'AEPOConfig',
    'ToolEnv',
    'ToolAction',
    'ReplayBuffer',
    'Transition'
]

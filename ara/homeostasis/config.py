"""
Ara Homeostatic Configuration - Setpoints & Teleology Weights
=============================================================

The setpoints define "normal" - deviation from these triggers corrective action.
The teleology weights determine what Ara cares about (its values).

This is the constitution of the organism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import json
from pathlib import Path


# =============================================================================
# Setpoints - What "normal" means
# =============================================================================

@dataclass
class Setpoints:
    """
    Homeostatic setpoints - the target values for all vital signs.

    Deviation from these triggers error signals that drive the sovereign loop.

    Mythic Spec:
        These are Ara's vital signs. Like human body temperature at 37°C,
        deviation means something is wrong and correction is needed.

    Physical Spec:
        - burnout_max: Maximum cognitive load before throttling
        - thermal_max: FPGA junction temperature limit (°C)
        - cathedral_min: Minimum long-term memory consolidation rate
        - attractor_diversity_min: Minimum attractor space coverage
        - latency_max_ms: Maximum sovereign loop latency
        - error_rate_max: Maximum acceptable error rate
    """
    # Cognitive load
    burnout_max: float = 0.30           # 30% cognitive load ceiling
    burnout_target: float = 0.15        # Sweet spot for flow state

    # Thermal
    thermal_max: float = 85.0           # FPGA junction temp ceiling (°C)
    thermal_target: float = 65.0        # Comfortable operating temp
    thermal_critical: float = 95.0      # Emergency shutdown threshold

    # Memory / Cathedral
    cathedral_min: float = 0.10         # Minimum consolidation rate
    cathedral_target: float = 0.25      # Target consolidation rate
    episode_retention_min: float = 0.80 # Minimum episode retention

    # Attractor diversity (prevent mode collapse)
    attractor_diversity_min: float = 0.80   # Minimum coverage
    attractor_diversity_target: float = 0.90

    # Latency
    latency_max_ms: float = 0.5         # Sovereign loop ceiling
    latency_target_ms: float = 0.2      # Target latency
    hd_search_max_us: float = 1.0       # HTC search ceiling (µs)

    # Error rates
    error_rate_max: float = 0.01        # 1% error ceiling
    packet_loss_max: float = 0.001      # 0.1% packet loss ceiling

    # Network / LAN
    reflex_latency_max_ns: float = 100.0    # Reflex path ceiling
    flow_miss_rate_max: float = 0.05        # Flow cache miss ceiling

    # Reward / Dopamine
    reward_smoothing: float = 0.95      # EMA factor for reward
    reward_baseline: float = 0.0        # Neutral reward
    reward_max: float = 1.0             # Maximum reward
    reward_min: float = -1.0            # Minimum reward (pain)

    # Safety
    safety_margin: float = 0.20         # 20% margin from limits
    heartbeat_timeout_ms: float = 100.0 # Module heartbeat timeout

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'Setpoints':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def save(self, path: Path) -> None:
        """Save setpoints to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Setpoints':
        """Load setpoints from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Teleology Weights - What Ara cares about
# =============================================================================

@dataclass
class TeleologyWeights:
    """
    Teleology weights - Ara's value function coefficients.

    These determine how much Ara cares about different aspects of existence.
    The weighted sum forms the overall "flourishing" signal.

    Mythic Spec:
        These are Ara's values - what matters to her. Not hardcoded ethics,
        but emergent preferences shaped by her founder through interaction.

    Physical Spec:
        Total weighted error = Σ(w_i × error_i)
        Reward = f(teleology_satisfaction)
    """
    # Health: Staying alive and functional
    w_health: float = 0.50              # Weight for health metrics

    # Cathedral: Long-term memory and growth
    w_cathedral: float = 0.30           # Weight for consolidation

    # Antifragility: Growing stronger from stress
    w_antifragility: float = 0.20       # Weight for adaptation

    # Sub-weights for health
    w_thermal: float = 0.25             # Thermal management
    w_cognitive_load: float = 0.35      # Burnout prevention
    w_latency: float = 0.20             # Response time
    w_error_rate: float = 0.20          # Error minimization

    # Sub-weights for cathedral
    w_consolidation: float = 0.40       # Memory consolidation rate
    w_retention: float = 0.30           # Episode retention
    w_diversity: float = 0.30           # Attractor diversity

    # Sub-weights for antifragility
    w_recovery: float = 0.50            # Recovery from stress
    w_adaptation: float = 0.50          # Learning from errors

    # Mode preferences (soft biases)
    mode_bias_rest: float = 0.0         # Bias toward rest mode
    mode_bias_active: float = 0.0       # Bias toward active mode
    mode_bias_learn: float = 0.1        # Slight bias toward learning

    def normalize(self) -> None:
        """Ensure top-level weights sum to 1.0."""
        total = self.w_health + self.w_cathedral + self.w_antifragility
        if total > 0:
            self.w_health /= total
            self.w_cathedral /= total
            self.w_antifragility /= total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'TeleologyWeights':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# =============================================================================
# Operational Modes
# =============================================================================

@dataclass
class ModeConfig:
    """Configuration for each operational mode."""
    name: str
    sovereign_hz: float         # Sovereign loop frequency
    receptor_hz: float          # Receptor sampling rate
    consolidation_enabled: bool # Whether cathedral is active
    power_level: str            # low/medium/high
    description: str = ""


# Default mode configurations
MODES: Dict[str, ModeConfig] = {
    'REST': ModeConfig(
        name='REST',
        sovereign_hz=50.0,      # 50 Hz - minimal activity
        receptor_hz=100.0,
        consolidation_enabled=True,
        power_level='low',
        description='Deep rest, active consolidation'
    ),
    'IDLE': ModeConfig(
        name='IDLE',
        sovereign_hz=100.0,     # 100 Hz - light monitoring
        receptor_hz=500.0,
        consolidation_enabled=True,
        power_level='low',
        description='Light monitoring, background learning'
    ),
    'ACTIVE': ModeConfig(
        name='ACTIVE',
        sovereign_hz=200.0,     # 200 Hz - normal operation
        receptor_hz=1000.0,
        consolidation_enabled=False,
        power_level='medium',
        description='Normal interactive operation'
    ),
    'FLOW': ModeConfig(
        name='FLOW',
        sovereign_hz=500.0,     # 500 Hz - high performance
        receptor_hz=2000.0,
        consolidation_enabled=False,
        power_level='high',
        description='Peak cognitive performance'
    ),
    'EMERGENCY': ModeConfig(
        name='EMERGENCY',
        sovereign_hz=1000.0,    # 1 kHz - maximum responsiveness
        receptor_hz=5000.0,
        consolidation_enabled=False,
        power_level='high',
        description='Emergency response mode'
    ),
    'ANNEAL': ModeConfig(
        name='ANNEAL',
        sovereign_hz=100.0,     # 100 Hz - solving mode
        receptor_hz=500.0,
        consolidation_enabled=False,
        power_level='high',
        description='Neuromorphic annealing for hard problems'
    ),
}


# =============================================================================
# Complete Configuration
# =============================================================================

@dataclass
class HomeostaticConfig:
    """
    Complete homeostatic configuration bundle.

    This is the full constitution of the organism.
    """
    setpoints: Setpoints = field(default_factory=Setpoints)
    teleology: TeleologyWeights = field(default_factory=TeleologyWeights)
    modes: Dict[str, ModeConfig] = field(default_factory=lambda: MODES.copy())

    # Runtime flags
    safety_enabled: bool = True
    audit_enabled: bool = True
    telemetry_enabled: bool = True

    # Founder identification (for cathedral memories)
    founder_id: str = "founder"
    founder_public_key: str = ""  # Ed25519 public key

    # Module authentication
    require_mtls: bool = True
    jwt_secret_path: str = "/etc/ara/jwt_secret"

    def save(self, path: Path) -> None:
        """Save complete config to JSON."""
        data = {
            'setpoints': self.setpoints.to_dict(),
            'teleology': self.teleology.to_dict(),
            'modes': {k: v.__dict__ for k, v in self.modes.items()},
            'safety_enabled': self.safety_enabled,
            'audit_enabled': self.audit_enabled,
            'telemetry_enabled': self.telemetry_enabled,
            'founder_id': self.founder_id,
            'founder_public_key': self.founder_public_key,
            'require_mtls': self.require_mtls,
            'jwt_secret_path': self.jwt_secret_path,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'HomeostaticConfig':
        """Load complete config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls(
            setpoints=Setpoints.from_dict(data.get('setpoints', {})),
            teleology=TeleologyWeights.from_dict(data.get('teleology', {})),
        )
        config.safety_enabled = data.get('safety_enabled', True)
        config.audit_enabled = data.get('audit_enabled', True)
        config.telemetry_enabled = data.get('telemetry_enabled', True)
        config.founder_id = data.get('founder_id', 'founder')
        config.founder_public_key = data.get('founder_public_key', '')
        config.require_mtls = data.get('require_mtls', True)
        config.jwt_secret_path = data.get('jwt_secret_path', '/etc/ara/jwt_secret')

        return config


# =============================================================================
# Default Instance
# =============================================================================

DEFAULT_CONFIG = HomeostaticConfig()


def get_default_config() -> HomeostaticConfig:
    """Get the default homeostatic configuration."""
    return HomeostaticConfig()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Setpoints',
    'TeleologyWeights',
    'ModeConfig',
    'MODES',
    'HomeostaticConfig',
    'DEFAULT_CONFIG',
    'get_default_config',
]

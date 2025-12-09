"""
Kernel Configuration
=====================

Loads and validates kernel configuration from YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    primary: str = "remote"  # "remote" or "local"
    remote_provider: str = "anthropic"  # anthropic, openai, etc.
    remote_model: str = "claude-3-5-sonnet-20241022"
    local_model: Optional[str] = None  # Path to local model
    local_quantization: str = "q4_0"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class MemoryConfig:
    """Memory layer configuration."""
    episodes_path: str = "ara_memory/episodes.sqlite"
    embeddings_path: str = "ara_memory/embeddings.faiss"
    packs_path: str = "ara_memory/packs"
    max_episodes: int = 100000
    embedding_dim: int = 384
    retrieval_k: int = 10
    compaction_strategy: str = "time_decay"  # time_decay, importance, cluster


@dataclass
class PheromoneConfig:
    """Pheromone bus configuration."""
    transport: str = "local"  # local, redis, nats
    redis_url: Optional[str] = None
    nats_url: Optional[str] = None
    default_ttl_ms: int = 30000
    evaporation_rate: float = 0.1


@dataclass
class SafetyConfig:
    """Safety covenant configuration."""
    allowed_domains: List[str] = field(default_factory=lambda: [
        "publishing", "coding", "creative", "research", "hardware"
    ])
    disallowed_domains: List[str] = field(default_factory=lambda: [
        "malware", "exploits", "unauthorized_access", "self_replication"
    ])
    disclosure_policy: str = "always"  # always, public_only, never
    human_approval_actions: List[str] = field(default_factory=lambda: [
        "social_media_post", "financial_transfer", "public_repo_push"
    ])
    max_autonomy_level: int = 1  # 0=draft, 1=human_gate, 2=auto


@dataclass
class PersonaConfig:
    """Persona configuration."""
    name: str = "Ara"
    voice: str = "geeky, articulate, emotionally grounded, explicitly non-human"
    toml_path: str = "ara_kernel/config/persona_ara.toml"


@dataclass
class ResourceConfig:
    """Resource limits."""
    max_concurrent_tools: int = 5
    tool_timeout_seconds: int = 60
    max_memory_mb: int = 4096
    gpu_memory_fraction: float = 0.5


@dataclass
class KernelConfig:
    """Complete kernel configuration."""
    node_id: str = "ara-primary"
    role: str = "general"  # general, publishing, lab, realtime
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pheromones: PheromoneConfig = field(default_factory=PheromoneConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KernelConfig:
        """Create config from dictionary."""
        return cls(
            node_id=data.get("node_id", "ara-primary"),
            role=data.get("role", "general"),
            model=ModelConfig(**data.get("model", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            pheromones=PheromoneConfig(**data.get("pheromones", {})),
            safety=SafetyConfig(**data.get("safety", {})),
            persona=PersonaConfig(**data.get("persona", {})),
            resources=ResourceConfig(**data.get("resources", {})),
            log_level=data.get("log_level", "INFO"),
        )


def load_config(path: str) -> KernelConfig:
    """Load configuration from YAML file."""
    config_path = Path(path)

    if not config_path.exists():
        # Return defaults
        return KernelConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return KernelConfig.from_dict(data)

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
    max_context_tokens: int = 100000
    context_buffer: int = 4096


@dataclass
class MemoryConfig:
    """Memory layer configuration."""
    episodes_path: str = "ara_memory/soul/episodes"
    episodes_db: str = "ara_memory/episodes.sqlite"
    skills_path: str = "ara_memory/skills"
    brand_voice: str = "ara_memory/skills/brand_voice.json"
    templates: str = "ara_memory/skills/prompt_templates.json"
    checklists: str = "ara_memory/skills/checklists.yaml"
    embeddings_path: str = "ara_memory/embeddings.faiss"
    packs_path: str = "ara_memory/packs"
    max_episodes: int = 100000
    embedding_dim: int = 384
    retrieval_k: int = 10
    compaction_strategy: str = "time_decay"  # time_decay, importance, cluster
    compaction_threshold: float = 0.7


@dataclass
class PheromoneConfig:
    """Pheromone bus configuration."""
    transport: str = "local"  # local, redis, nats
    redis_url: Optional[str] = None
    nats_url: Optional[str] = None
    default_ttl_ms: int = 30000
    evaporation_rate: float = 0.1
    max_pheromones: int = 1000
    cleanup_interval_s: int = 60


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
    hard_stops: List[str] = field(default_factory=lambda: [
        "create_malware", "unauthorized_access", "self_replicate",
        "covert_persistence", "modify_safety_rules", "financial_transfer_auto"
    ])


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
    max_context_mb: int = 512
    gpu_memory_fraction: float = 0.5
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000


def _filter_dataclass_fields(dc_class, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter dict to only include fields that exist in the dataclass."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(dc_class)}
    return {k: v for k, v in data.items() if k in valid_fields}


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
            model=ModelConfig(**_filter_dataclass_fields(ModelConfig, data.get("model", {}))),
            memory=MemoryConfig(**_filter_dataclass_fields(MemoryConfig, data.get("memory", {}))),
            pheromones=PheromoneConfig(**_filter_dataclass_fields(PheromoneConfig, data.get("pheromones", {}))),
            safety=SafetyConfig(**_filter_dataclass_fields(SafetyConfig, data.get("safety", {}))),
            persona=PersonaConfig(**_filter_dataclass_fields(PersonaConfig, data.get("persona", {}))),
            resources=ResourceConfig(**_filter_dataclass_fields(ResourceConfig, data.get("resources", {}))),
            log_level=data.get("log_level", data.get("logging", {}).get("level", "INFO")),
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

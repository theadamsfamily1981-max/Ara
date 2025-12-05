"""Agent Forge - Ara designs new agent configurations.

Beyond just skills, Ara can forge entirely new agent "blueprints" that
specify:
- Purpose and capabilities
- Teacher routing rules
- Prompt templates
- Success criteria
- Constraints and guardrails

This is meta-meta-learning: Ara learning how to create agents that learn.

Example blueprint:
  agent_id: async_debugger_v2
  purpose: "Specialized agent for async Python debugging"
  capabilities: ["trace async flows", "identify race conditions"]
  routing: {debug_async: ["claude"], design_fix: ["claude", "nova"]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """A routing rule for an agent."""

    intent: str
    teachers: List[str]
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "teachers": self.teachers,
            "priority": self.priority,
            "conditions": self.conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRule":
        return cls(
            intent=data["intent"],
            teachers=data["teachers"],
            priority=data.get("priority", 1),
            conditions=data.get("conditions", {}),
        )


@dataclass
class AgentConstraint:
    """A constraint or guardrail for an agent."""

    name: str
    description: str
    constraint_type: str  # "must", "must_not", "prefer"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "constraint_type": self.constraint_type,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConstraint":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            constraint_type=data.get("constraint_type", "prefer"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class AgentBlueprint:
    """A blueprint for an agent configuration."""

    id: str
    name: str
    purpose: str
    version: str = "1.0.0"

    # What this agent can do
    capabilities: List[str] = field(default_factory=list)
    intents_handled: List[str] = field(default_factory=list)

    # How it routes to teachers
    routing_rules: List[RoutingRule] = field(default_factory=list)
    default_workflow: List[str] = field(default_factory=list)

    # Templates it uses
    template_ids: List[str] = field(default_factory=list)
    skill_capsule_ids: List[str] = field(default_factory=list)

    # Constraints and guardrails
    constraints: List[AgentConstraint] = field(default_factory=list)

    # Success criteria
    success_metrics: Dict[str, float] = field(default_factory=dict)
    min_success_rate: float = 0.7

    # Performance tracking
    total_invocations: int = 0
    successful_invocations: int = 0
    total_reward: float = 0.0

    # Lineage
    parent_id: Optional[str] = None
    evolution_notes: str = ""

    # Metadata
    author: str = "ara"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    status: str = "draft"  # "draft", "active", "archived"

    @property
    def success_rate(self) -> Optional[float]:
        if self.total_invocations == 0:
            return None
        return self.successful_invocations / self.total_invocations

    @property
    def avg_reward(self) -> Optional[float]:
        if self.total_invocations == 0:
            return None
        return self.total_reward / self.total_invocations

    def get_routing(self, intent: str) -> List[str]:
        """Get the teacher sequence for an intent."""
        for rule in sorted(self.routing_rules, key=lambda r: r.priority, reverse=True):
            if rule.intent == intent:
                return rule.teachers
        return self.default_workflow

    def record_invocation(self, success: bool, reward: float = 0.0) -> None:
        """Record an invocation result."""
        self.total_invocations += 1
        if success:
            self.successful_invocations += 1
        self.total_reward += reward
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "version": self.version,
            "capabilities": self.capabilities,
            "intents_handled": self.intents_handled,
            "routing_rules": [r.to_dict() for r in self.routing_rules],
            "default_workflow": self.default_workflow,
            "template_ids": self.template_ids,
            "skill_capsule_ids": self.skill_capsule_ids,
            "constraints": [c.to_dict() for c in self.constraints],
            "success_metrics": self.success_metrics,
            "min_success_rate": self.min_success_rate,
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "parent_id": self.parent_id,
            "evolution_notes": self.evolution_notes,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBlueprint":
        blueprint = cls(
            id=data["id"],
            name=data["name"],
            purpose=data.get("purpose", ""),
            version=data.get("version", "1.0.0"),
            capabilities=data.get("capabilities", []),
            intents_handled=data.get("intents_handled", []),
            default_workflow=data.get("default_workflow", []),
            template_ids=data.get("template_ids", []),
            skill_capsule_ids=data.get("skill_capsule_ids", []),
            success_metrics=data.get("success_metrics", {}),
            min_success_rate=data.get("min_success_rate", 0.7),
            total_invocations=data.get("total_invocations", 0),
            successful_invocations=data.get("successful_invocations", 0),
            total_reward=data.get("total_reward", 0.0),
            parent_id=data.get("parent_id"),
            evolution_notes=data.get("evolution_notes", ""),
            author=data.get("author", "ara"),
            tags=data.get("tags", []),
            status=data.get("status", "draft"),
        )

        for rule_data in data.get("routing_rules", []):
            blueprint.routing_rules.append(RoutingRule.from_dict(rule_data))

        for constraint_data in data.get("constraints", []):
            blueprint.constraints.append(AgentConstraint.from_dict(constraint_data))

        return blueprint

    def format_spec(self) -> str:
        """Format as a specification document."""
        lines = [
            f"# Agent: {self.name}",
            f"ID: {self.id} | Version: {self.version} | Status: {self.status}",
            "",
            "## Purpose",
            self.purpose,
            "",
        ]

        if self.capabilities:
            lines.append("## Capabilities")
            for cap in self.capabilities:
                lines.append(f"- {cap}")
            lines.append("")

        if self.routing_rules:
            lines.append("## Routing Rules")
            for rule in self.routing_rules:
                lines.append(f"- {rule.intent}: {' → '.join(rule.teachers)}")
            lines.append("")

        if self.default_workflow:
            lines.append(f"## Default Workflow")
            lines.append(f"{' → '.join(self.default_workflow)}")
            lines.append("")

        if self.constraints:
            lines.append("## Constraints")
            for c in self.constraints:
                lines.append(f"- [{c.constraint_type}] {c.name}: {c.description}")
            lines.append("")

        lines.append("## Performance")
        if self.success_rate is not None:
            lines.append(f"- Success rate: {self.success_rate:.0%}")
        if self.avg_reward is not None:
            lines.append(f"- Avg reward: {self.avg_reward:.0%}")
        lines.append(f"- Total invocations: {self.total_invocations}")

        return "\n".join(lines)


class AgentForge:
    """Forges new agent configurations."""

    def __init__(self, blueprints_path: Optional[Path] = None):
        """Initialize the forge.

        Args:
            blueprints_path: Path to blueprints JSON file
        """
        self.blueprints_path = blueprints_path or (
            Path.home() / ".ara" / "meta" / "toolsmith" / "blueprints.json"
        )
        self.blueprints_path.parent.mkdir(parents=True, exist_ok=True)

        self._blueprints: Dict[str, AgentBlueprint] = {}
        self._loaded = False
        self._next_id = 1

    def _load(self, force: bool = False) -> None:
        """Load blueprints from disk."""
        if self._loaded and not force:
            return

        self._blueprints.clear()

        if self.blueprints_path.exists():
            try:
                with open(self.blueprints_path) as f:
                    data = json.load(f)
                for bp_data in data.get("blueprints", []):
                    blueprint = AgentBlueprint.from_dict(bp_data)
                    self._blueprints[blueprint.id] = blueprint
                    # Update ID counter
                    if blueprint.id.startswith("AGENT-"):
                        try:
                            num = int(blueprint.id[6:10])
                            self._next_id = max(self._next_id, num + 1)
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to load blueprints: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save blueprints to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "blueprints": [bp.to_dict() for bp in self._blueprints.values()],
        }
        with open(self.blueprints_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, name: str) -> str:
        """Generate a unique blueprint ID."""
        slug = name.lower().replace(" ", "_")[:10]
        id_str = f"AGENT-{self._next_id:04d}-{slug}"
        self._next_id += 1
        return id_str

    def get_blueprint(self, blueprint_id: str) -> Optional[AgentBlueprint]:
        """Get a blueprint by ID."""
        self._load()
        return self._blueprints.get(blueprint_id)

    def get_all_blueprints(self) -> List[AgentBlueprint]:
        """Get all blueprints."""
        self._load()
        return list(self._blueprints.values())

    def get_active_blueprints(self) -> List[AgentBlueprint]:
        """Get all active blueprints."""
        self._load()
        return [bp for bp in self._blueprints.values() if bp.status == "active"]

    def forge_blueprint(
        self,
        name: str,
        purpose: str,
        capabilities: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        routing: Optional[Dict[str, List[str]]] = None,
        default_workflow: Optional[List[str]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentBlueprint:
        """Forge a new agent blueprint.

        Args:
            name: Agent name
            purpose: What this agent does
            capabilities: List of capabilities
            intents: Intents this agent handles
            routing: Intent → teacher mapping
            default_workflow: Default teacher sequence
            constraints: List of constraints
            tags: Categorization tags

        Returns:
            The new blueprint
        """
        self._load()

        blueprint = AgentBlueprint(
            id=self._generate_id(name),
            name=name,
            purpose=purpose,
            capabilities=capabilities or [],
            intents_handled=intents or [],
            default_workflow=default_workflow or ["claude"],
            tags=tags or [],
        )

        # Add routing rules
        if routing:
            for intent, teachers in routing.items():
                blueprint.routing_rules.append(RoutingRule(
                    intent=intent,
                    teachers=teachers,
                ))

        # Add constraints
        if constraints:
            for c_data in constraints:
                blueprint.constraints.append(AgentConstraint.from_dict(c_data))

        self._blueprints[blueprint.id] = blueprint
        self._save()
        logger.info(f"Forged agent blueprint: {blueprint.id}")

        return blueprint

    def evolve_blueprint(
        self,
        parent_id: str,
        changes: Dict[str, Any],
        notes: str = "",
    ) -> Optional[AgentBlueprint]:
        """Evolve a blueprint into a new version.

        Args:
            parent_id: Parent blueprint ID
            changes: Changes to apply
            notes: Evolution notes

        Returns:
            New evolved blueprint
        """
        self._load()

        parent = self._blueprints.get(parent_id)
        if not parent:
            return None

        # Clone parent
        new_data = parent.to_dict()
        del new_data["id"]
        new_data["parent_id"] = parent_id
        new_data["evolution_notes"] = notes

        # Apply changes
        for key, value in changes.items():
            if key in new_data:
                new_data[key] = value

        # Increment version
        version_parts = parent.version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_data["version"] = ".".join(version_parts)

        # Reset performance
        new_data["total_invocations"] = 0
        new_data["successful_invocations"] = 0
        new_data["total_reward"] = 0.0
        new_data["status"] = "draft"

        # Create new blueprint
        child = AgentBlueprint.from_dict(new_data)
        child.id = self._generate_id(child.name)

        self._blueprints[child.id] = child
        self._save()
        logger.info(f"Evolved blueprint {parent_id} → {child.id}")

        return child

    def find_blueprint_for_intent(
        self,
        intent: str,
    ) -> Optional[AgentBlueprint]:
        """Find the best blueprint for an intent.

        Args:
            intent: Intent to handle

        Returns:
            Best matching blueprint
        """
        self._load()

        candidates = [
            bp for bp in self._blueprints.values()
            if bp.status == "active" and intent in bp.intents_handled
        ]

        if not candidates:
            return None

        # Sort by success rate
        candidates.sort(
            key=lambda bp: (bp.success_rate or 0, bp.total_invocations),
            reverse=True,
        )

        return candidates[0]

    def record_invocation(
        self,
        blueprint_id: str,
        success: bool,
        reward: float = 0.0,
    ) -> bool:
        """Record a blueprint invocation.

        Args:
            blueprint_id: Blueprint ID
            success: Whether it succeeded
            reward: Quality score

        Returns:
            True if recorded
        """
        self._load()

        blueprint = self._blueprints.get(blueprint_id)
        if not blueprint:
            return False

        blueprint.record_invocation(success, reward)

        # Auto-activate if performing well
        if (blueprint.status == "draft" and
            blueprint.total_invocations >= 5 and
            (blueprint.success_rate or 0) >= blueprint.min_success_rate):
            blueprint.status = "active"
            logger.info(f"Auto-activated blueprint {blueprint_id}")

        # Auto-archive if performing poorly
        if (blueprint.status == "active" and
            blueprint.total_invocations >= 20 and
            (blueprint.success_rate or 0) < 0.5):
            blueprint.status = "archived"
            logger.info(f"Auto-archived blueprint {blueprint_id}")

        self._save()
        return True

    def get_lineage(self, blueprint_id: str) -> List[AgentBlueprint]:
        """Get the evolution lineage of a blueprint."""
        self._load()

        lineage = []
        current_id = blueprint_id

        while current_id:
            blueprint = self._blueprints.get(current_id)
            if not blueprint:
                break
            lineage.append(blueprint)
            current_id = blueprint.parent_id

        return list(reversed(lineage))

    def get_summary(self) -> Dict[str, Any]:
        """Get forge summary."""
        self._load()

        active = [bp for bp in self._blueprints.values() if bp.status == "active"]
        drafts = [bp for bp in self._blueprints.values() if bp.status == "draft"]

        # Top performers
        top = sorted(
            [bp for bp in active if bp.total_invocations >= 3],
            key=lambda bp: bp.success_rate or 0,
            reverse=True,
        )[:5]

        return {
            "total_blueprints": len(self._blueprints),
            "active": len(active),
            "drafts": len(drafts),
            "top_performers": [
                {
                    "id": bp.id,
                    "name": bp.name,
                    "success_rate": bp.success_rate,
                    "invocations": bp.total_invocations,
                }
                for bp in top
            ],
        }


# =============================================================================
# Default Blueprints
# =============================================================================

DEFAULT_BLUEPRINTS = [
    {
        "id": "AGENT-0001-code_surge",
        "name": "Code Surgeon",
        "purpose": "Specialized agent for code debugging and fixes",
        "capabilities": [
            "Debug code errors",
            "Trace execution flow",
            "Suggest fixes",
            "Explain issues",
        ],
        "intents_handled": ["debug_code"],
        "routing_rules": [
            {"intent": "debug_code", "teachers": ["claude"], "priority": 1},
        ],
        "default_workflow": ["claude"],
        "constraints": [
            {
                "name": "preserve_style",
                "description": "Maintain existing code style",
                "constraint_type": "prefer",
            },
        ],
        "tags": ["debugging", "code"],
    },
    {
        "id": "AGENT-0002-architect",
        "name": "System Architect",
        "purpose": "Design and review system architectures",
        "capabilities": [
            "Design architectures",
            "Review designs",
            "Identify patterns",
            "Suggest improvements",
        ],
        "intents_handled": ["design_arch", "review"],
        "routing_rules": [
            {"intent": "design_arch", "teachers": ["nova", "gemini"], "priority": 1},
            {"intent": "review", "teachers": ["nova"], "priority": 1},
        ],
        "default_workflow": ["nova"],
        "tags": ["architecture", "design"],
    },
    {
        "id": "AGENT-0003-researcher",
        "name": "Research Explorer",
        "purpose": "Explore technical topics and synthesize findings",
        "capabilities": [
            "Research topics",
            "Synthesize information",
            "Compare approaches",
            "Generate recommendations",
        ],
        "intents_handled": ["research"],
        "routing_rules": [
            {"intent": "research", "teachers": ["gemini", "claude"], "priority": 1},
        ],
        "default_workflow": ["gemini"],
        "tags": ["research", "exploration"],
    },
]


def seed_default_blueprints(forge: AgentForge) -> int:
    """Seed default blueprints.

    Args:
        forge: Agent forge

    Returns:
        Number seeded
    """
    seeded = 0
    for bp_data in DEFAULT_BLUEPRINTS:
        if not forge.get_blueprint(bp_data["id"]):
            forge._load()
            blueprint = AgentBlueprint.from_dict(bp_data)
            blueprint.status = "active"
            forge._blueprints[blueprint.id] = blueprint
            seeded += 1

    if seeded:
        forge._save()

    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_forge: Optional[AgentForge] = None


def get_agent_forge() -> AgentForge:
    """Get the default agent forge."""
    global _default_forge
    if _default_forge is None:
        _default_forge = AgentForge()
    return _default_forge


def forge_agent(
    name: str,
    purpose: str,
    capabilities: Optional[List[str]] = None,
    routing: Optional[Dict[str, List[str]]] = None,
) -> AgentBlueprint:
    """Forge a new agent blueprint."""
    return get_agent_forge().forge_blueprint(
        name=name,
        purpose=purpose,
        capabilities=capabilities,
        routing=routing,
    )


def find_agent_for_intent(intent: str) -> Optional[AgentBlueprint]:
    """Find the best agent for an intent."""
    return get_agent_forge().find_blueprint_for_intent(intent)

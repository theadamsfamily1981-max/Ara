# ara/ktp/schema.py
"""
KTP Schema - The structure of a Knowledge Transfer Protocol entry.

Each entry encodes a concept with four layers:
    allegory → plain_meaning → contract → anchors

This is how we make Tamarian portable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
import json


@dataclass
class KTPContract:
    """
    Formal contract for a KTP concept.

    Specifies inputs, outputs, and invariants that must hold.
    This is the "API" of the allegory.
    """
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "invariants": self.invariants,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
        }


@dataclass
class KTPAnchors:
    """
    Implementation anchors for a KTP concept.

    Links the allegory to actual code, files, and metrics.
    """
    code: List[str] = field(default_factory=list)       # File paths
    functions: List[str] = field(default_factory=list)  # Function names
    metrics: List[str] = field(default_factory=list)    # Observable metrics
    configs: List[str] = field(default_factory=list)    # Config files

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "functions": self.functions,
            "metrics": self.metrics,
            "configs": self.configs,
        }


@dataclass
class KTPEntry:
    """
    A complete Knowledge Transfer Protocol entry.

    Four layers:
        1. allegory - The Tamarian phrase
        2. plain_meaning - Normal language explanation
        3. contract - Formal inputs/outputs/invariants
        4. anchors - Code/metrics implementation

    Example:
        entry = KTPEntry(
            id="gauntlets_three",
            allegory="In order to cross my bridge you see, you must pass my gauntlets three.",
            plain_meaning="Any new component must pass technical, antifragility, and council approval.",
            contract=KTPContract(
                inputs=[{"candidate": "code + config + metadata"}],
                outputs=[{"verdict": "pass | fail"}],
                invariants=["No component touches live without all 3 passes"],
            ),
            anchors=KTPAnchors(
                code=["ara/dojo/gauntlet.py"],
                metrics=["gauntlet_pass_rate"],
            ),
        )
    """
    id: str                           # Unique identifier
    allegory: str                     # The mythic phrase
    plain_meaning: str                # Normal language explanation
    contract: KTPContract = field(default_factory=KTPContract)
    anchors: KTPAnchors = field(default_factory=KTPAnchors)

    # Metadata
    category: str = "core"            # core, perception, learning, etc.
    related: List[str] = field(default_factory=list)  # Related KTP ids
    version: str = "1.0"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "allegory": self.allegory,
            "plain_meaning": self.plain_meaning,
            "contract": self.contract.to_dict(),
            "anchors": self.anchors.to_dict(),
            "category": self.category,
            "related": self.related,
            "version": self.version,
        }

    def to_yaml(self) -> str:
        """Export as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Export as human-readable markdown."""
        lines = [
            f"## {self.id}",
            "",
            f"**Allegory:** *\"{self.allegory}\"*",
            "",
            f"**Plain Meaning:** {self.plain_meaning}",
            "",
            "### Contract",
            "",
        ]

        if self.contract.inputs:
            lines.append("**Inputs:**")
            for inp in self.contract.inputs:
                for k, v in inp.items():
                    lines.append(f"- `{k}`: {v}")
            lines.append("")

        if self.contract.outputs:
            lines.append("**Outputs:**")
            for out in self.contract.outputs:
                for k, v in out.items():
                    lines.append(f"- `{k}`: {v}")
            lines.append("")

        if self.contract.invariants:
            lines.append("**Invariants:**")
            for inv in self.contract.invariants:
                lines.append(f"- {inv}")
            lines.append("")

        if self.anchors.code or self.anchors.functions or self.anchors.metrics:
            lines.append("### Implementation")
            lines.append("")
            if self.anchors.code:
                lines.append("**Code:**")
                for c in self.anchors.code:
                    lines.append(f"- `{c}`")
            if self.anchors.functions:
                lines.append("**Functions:**")
                for f in self.anchors.functions:
                    lines.append(f"- `{f}`")
            if self.anchors.metrics:
                lines.append("**Metrics:**")
                for m in self.anchors.metrics:
                    lines.append(f"- `{m}`")
            lines.append("")

        return "\n".join(lines)

    def to_prompt_snippet(self) -> str:
        """
        Export as a compact prompt snippet for injecting into LLM context.

        This is what you paste at the top of a Claude/Perplexity/Gemini session.
        """
        return f"""**{self.id}**
- Allegory: "{self.allegory}"
- Meaning: {self.plain_meaning}
- Contract: inputs={[list(i.keys())[0] for i in self.contract.inputs]}, outputs={[list(o.keys())[0] for o in self.contract.outputs]}
- Invariants: {'; '.join(self.contract.invariants[:2])}"""


class KTPRegistry:
    """
    Registry of all KTP entries.

    Provides lookup, search, and export functionality.
    """

    def __init__(self):
        self._entries: Dict[str, KTPEntry] = {}

    def register(self, entry: KTPEntry) -> None:
        """Register a KTP entry."""
        self._entries[entry.id] = entry

    def get(self, entry_id: str) -> Optional[KTPEntry]:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def list_all(self) -> List[str]:
        """List all entry IDs."""
        return list(self._entries.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List entries by category."""
        return [
            e.id for e in self._entries.values()
            if e.category == category
        ]

    def search(self, query: str) -> List[KTPEntry]:
        """Search entries by keyword in allegory or plain_meaning."""
        query_lower = query.lower()
        return [
            e for e in self._entries.values()
            if query_lower in e.allegory.lower() or query_lower in e.plain_meaning.lower()
        ]

    def get_related(self, entry_id: str) -> List[KTPEntry]:
        """Get entries related to the given entry."""
        entry = self.get(entry_id)
        if not entry:
            return []
        return [
            self._entries[rid] for rid in entry.related
            if rid in self._entries
        ]

    def export_all_yaml(self) -> str:
        """Export all entries as YAML."""
        data = {eid: e.to_dict() for eid, e in self._entries.items()}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def export_prompt_header(self, entry_ids: Optional[List[str]] = None) -> str:
        """
        Export entries as a prompt header for other LLMs.

        Args:
            entry_ids: Specific entries to export, or None for all

        Returns:
            Markdown/text suitable for pasting into an LLM prompt
        """
        ids = entry_ids or list(self._entries.keys())
        entries = [self._entries[eid] for eid in ids if eid in self._entries]

        lines = [
            "# Knowledge Transfer Protocol (KTP)",
            "",
            "When I reference these concepts, operate according to these contracts:",
            "",
        ]

        for entry in entries:
            lines.append(entry.to_prompt_snippet())
            lines.append("")

        return "\n".join(lines)


# Global registry
_registry: Optional[KTPRegistry] = None


def get_ktp_registry() -> KTPRegistry:
    """Get the global KTP registry."""
    global _registry
    if _registry is None:
        _registry = KTPRegistry()
        # Auto-register core entries
        from . import entries
        for name in dir(entries):
            obj = getattr(entries, name)
            if isinstance(obj, KTPEntry):
                _registry.register(obj)
    return _registry

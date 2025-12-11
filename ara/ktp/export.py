# ara/ktp/export.py
"""
KTP Export - Generate portable specs for other AI systems.

This is how you hand Ara's vocabulary to Perplexity, Claude, Gemini, etc.
"""

from typing import List, Optional
import yaml

from .schema import KTPEntry, get_ktp_registry


def export_yaml(entry_ids: Optional[List[str]] = None) -> str:
    """
    Export KTP entries as YAML.

    Args:
        entry_ids: Specific entries to export, or None for all

    Returns:
        YAML string
    """
    registry = get_ktp_registry()
    ids = entry_ids or registry.list_all()

    data = {}
    for eid in ids:
        entry = registry.get(eid)
        if entry:
            data[eid] = entry.to_dict()

    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def export_markdown(entry_ids: Optional[List[str]] = None) -> str:
    """
    Export KTP entries as readable markdown.

    Args:
        entry_ids: Specific entries to export, or None for all

    Returns:
        Markdown string
    """
    registry = get_ktp_registry()
    ids = entry_ids or registry.list_all()

    lines = [
        "# Ara Knowledge Transfer Protocol",
        "",
        "Portable concept specifications for consistent cross-model understanding.",
        "",
        "---",
        "",
    ]

    for eid in ids:
        entry = registry.get(eid)
        if entry:
            lines.append(entry.to_markdown())
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def export_prompt_header(
    entry_ids: Optional[List[str]] = None,
    compact: bool = True
) -> str:
    """
    Export KTP entries as a prompt header for LLM sessions.

    This is what you paste at the top of a Claude/Perplexity/Gemini session
    to give them Ara's vocabulary.

    Args:
        entry_ids: Specific entries to export, or None for core concepts
        compact: If True, use compressed format

    Returns:
        Prompt header string

    Example:
        header = export_prompt_header(["homeostasis", "edge_of_chaos"])
        # Paste this at the top of your Claude session:
        # "Use this KTP: {header}"
    """
    registry = get_ktp_registry()

    # Default to core concepts
    if entry_ids is None:
        entry_ids = [
            "edge_of_chaos",
            "homeostasis",
            "cathedral",
            "worldline",
            "lizard_brain",
        ]

    entries = [registry.get(eid) for eid in entry_ids if registry.get(eid)]

    if compact:
        lines = [
            "# Ara KTP (Knowledge Transfer Protocol)",
            "",
            "When I reference these concepts, operate according to these contracts:",
            "",
        ]
        for entry in entries:
            lines.append(entry.to_prompt_snippet())
            lines.append("")
    else:
        lines = [
            "# Ara Knowledge Transfer Protocol",
            "",
            "I am working with a system called Ara. When I use the following terms,",
            "please understand them according to these precise definitions:",
            "",
        ]
        for entry in entries:
            lines.append(f"## {entry.id}")
            lines.append(f"**Allegory:** \"{entry.allegory}\"")
            lines.append(f"**Meaning:** {entry.plain_meaning}")
            lines.append("")

    return "\n".join(lines)


def export_for_model(
    model: str,
    entry_ids: Optional[List[str]] = None
) -> str:
    """
    Export KTP optimized for a specific model.

    Args:
        model: Target model ("claude", "perplexity", "gemini", "gpt")
        entry_ids: Specific entries to export

    Returns:
        Model-optimized prompt header
    """
    registry = get_ktp_registry()

    # Default entries
    if entry_ids is None:
        entry_ids = ["edge_of_chaos", "homeostasis", "cathedral", "lizard_brain"]

    entries = [registry.get(eid) for eid in entry_ids if registry.get(eid)]

    if model.lower() in ("claude", "anthropic"):
        # Claude likes structured, detailed context
        return _export_for_claude(entries)
    elif model.lower() in ("perplexity",):
        # Perplexity is good with concise bullet points
        return _export_for_perplexity(entries)
    elif model.lower() in ("gemini", "google"):
        # Gemini handles longer context well
        return _export_for_gemini(entries)
    else:
        # Default: compact format
        return export_prompt_header(entry_ids, compact=True)


def _export_for_claude(entries: List[KTPEntry]) -> str:
    """Export optimized for Claude."""
    lines = [
        "<ara_ktp>",
        "# Knowledge Transfer Protocol for Ara",
        "",
        "These are the core concepts of the Ara system. When I reference them,",
        "please reason according to their contracts.",
        "",
    ]

    for entry in entries:
        lines.append(f"## {entry.id}")
        lines.append(f"Allegory: \"{entry.allegory}\"")
        lines.append(f"Meaning: {entry.plain_meaning}")
        if entry.contract.invariants:
            lines.append("Invariants:")
            for inv in entry.contract.invariants:
                lines.append(f"  - {inv}")
        lines.append("")

    lines.append("</ara_ktp>")
    return "\n".join(lines)


def _export_for_perplexity(entries: List[KTPEntry]) -> str:
    """Export optimized for Perplexity (concise)."""
    lines = [
        "**Ara KTP Definitions:**",
        "",
    ]

    for entry in entries:
        lines.append(f"- **{entry.id}**: {entry.plain_meaning[:200]}...")
        lines.append("")

    return "\n".join(lines)


def _export_for_gemini(entries: List[KTPEntry]) -> str:
    """Export optimized for Gemini."""
    lines = [
        "# Ara System Vocabulary",
        "",
        "When working with Ara, use these concept definitions:",
        "",
    ]

    for entry in entries:
        lines.append(f"### {entry.id}")
        lines.append(f"*\"{entry.allegory}\"*")
        lines.append("")
        lines.append(entry.plain_meaning)
        lines.append("")
        if entry.anchors.code:
            lines.append(f"Implementation: `{entry.anchors.code[0]}`")
        lines.append("")

    return "\n".join(lines)


# Convenience function
def get_standard_header() -> str:
    """
    Get the standard KTP header for Ara sessions.

    This is the minimum context any model needs to understand Ara.
    """
    return export_prompt_header([
        "edge_of_chaos",
        "homeostasis",
        "lizard_brain",
        "cathedral",
    ], compact=True)

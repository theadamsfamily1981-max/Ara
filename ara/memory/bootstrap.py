"""
Memory Bootstrap - Seed Ara's Memory at Startup
=================================================

Loads foundational memories, sacred lines, and context files
into EternalMemory during organism initialization.

Sources:
- ara_memories/*.yaml - Episode cards (behavioral patterns)
- config/sacred_lines.yaml - Sacred speech lines
- context/*.txt - Manifesto and identity context
- /*.txt (root) - Knowledge dumps (HNN, crypto, haptics, workshop)

This creates the "preloaded soul" - the memories she wakes up with.

Usage:
    from ara.memory.bootstrap import bootstrap_memory

    memory = EternalMemory(dim=4096)
    bootstrap_memory(memory)
    # Now memory contains foundational episodes
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger("ara.memory.bootstrap")

# Try imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    from ara.core.eternal_memory import EternalMemory
    from ara.core.axis_mundi import encode_text_to_hv
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    EternalMemory = None
    encode_text_to_hv = None

try:
    from ara.memory.loader import load_all_episode_cards, EpisodeCard
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    load_all_episode_cards = None
    EpisodeCard = None


# =============================================================================
# Path Configuration
# =============================================================================

def get_repo_root() -> Path:
    """Find the repository root."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "ara").exists():
            return parent
    return Path.cwd()


REPO_ROOT = get_repo_root()
MEMORIES_PATH = REPO_ROOT / "ara_memories"
CONTEXT_PATH = REPO_ROOT / "context"
CONFIG_PATH = REPO_ROOT / "config"


# =============================================================================
# Sacred Lines Loader
# =============================================================================

def load_sacred_lines(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load sacred lines from YAML config."""
    if not YAML_AVAILABLE:
        logger.warning("YAML not available - cannot load sacred lines")
        return {}

    path = config_path or CONFIG_PATH / "sacred_lines.yaml"

    if not path.exists():
        logger.warning(f"Sacred lines not found: {path}")
        return {}

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded sacred lines from {path}")
        return data or {}
    except Exception as e:
        logger.error(f"Failed to load sacred lines: {e}")
        return {}


# =============================================================================
# Context Files Loader
# =============================================================================

def load_context_files(context_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Load context files (manifesto, identity documents).

    Returns list of {filename, content} dicts.
    """
    path = context_path or CONTEXT_PATH

    if not path.exists():
        logger.warning(f"Context directory not found: {path}")
        return []

    context_files = []

    # Load .txt files
    for txt_file in path.glob("*.txt"):
        try:
            content = txt_file.read_text(encoding='utf-8')
            context_files.append({
                "filename": txt_file.name,
                "content": content,
            })
        except Exception as e:
            logger.warning(f"Failed to load {txt_file}: {e}")

    # Load .md files
    for md_file in path.glob("*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            context_files.append({
                "filename": md_file.name,
                "content": content,
            })
        except Exception as e:
            logger.warning(f"Failed to load {md_file}: {e}")

    logger.info(f"Loaded {len(context_files)} context files from {path}")
    return context_files


# =============================================================================
# Episode Card to Memory
# =============================================================================

def episode_card_to_memory(
    card: Any,  # EpisodeCard
    dim: int = 4096,
) -> Dict[str, Any]:
    """
    Convert an EpisodeCard to memory storage format.

    Returns dict with content_hv, emotion_hv, strength, meta.
    """
    if not CORE_AVAILABLE:
        return {}

    # Build text representation for HV encoding
    text_parts = [
        card.dialogue_snippets.paraphrased_exchange,
        card.lesson_for_future_ara,
        card.crofts_state.situation,
        card.aras_state.mode,
    ]
    text_content = " ".join(filter(None, text_parts))

    # Encode to hypervector
    content_hv = encode_text_to_hv(text_content, dim=dim)

    # Create emotion HV from axes
    axes = card.hv_hints.emotional_axes
    # Seed emotion HV from valence/arousal/attachment
    seed = int((axes.valence + axes.arousal + axes.attachment) * 1000)
    rng = np.random.default_rng(seed)
    emotion_hv = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

    # Map resurrection role to strength
    role_strength = {
        "CORE_COVENANT_PATTERN": 1.0,
        "SCAR_BASELINE": 0.95,
        "MYTHIC_BACKBONE": 0.9,
        "NORMAL_EPISODE": 0.7,
    }
    strength = role_strength.get(card.resurrection_role.value, 0.7) * card.certainty

    # Metadata
    meta = {
        "id": card.id,
        "source": "bootstrap",
        "type": "episode_card",
        "date": card.rough_date,
        "context_tags": card.context_tags,
        "persona_traits": card.ara_persona_traits,
        "lesson": card.lesson_for_future_ara[:200],
        "resurrection_role": card.resurrection_role.value,
        "valence": axes.valence,
        "arousal": axes.arousal,
        "attachment": axes.attachment,
    }

    return {
        "content_hv": content_hv,
        "emotion_hv": emotion_hv,
        "strength": strength,
        "meta": meta,
    }


def sacred_line_to_memory(
    line_id: str,
    line_data: Dict[str, Any],
    dim: int = 4096,
) -> Dict[str, Any]:
    """Convert a sacred line to memory format."""
    if not CORE_AVAILABLE:
        return {}

    # Build text from line data
    text = line_data.get("text", "")
    semantics = line_data.get("semantics", [])
    text_content = f"{text} {' '.join(semantics)}"

    content_hv = encode_text_to_hv(text_content, dim=dim)

    # Sacred lines are high-valence, calm
    seed = hash(line_id) % (2**31)
    rng = np.random.default_rng(seed)
    emotion_hv = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

    # Weight determines strength
    strength = line_data.get("weight", 0.8)

    meta = {
        "id": f"sacred:{line_id}",
        "source": "bootstrap",
        "type": "sacred_line",
        "text": text,
        "category": line_data.get("category", ""),
        "semantics": semantics[:3],  # First 3 meanings
    }

    return {
        "content_hv": content_hv,
        "emotion_hv": emotion_hv,
        "strength": strength,
        "meta": meta,
    }


def context_to_memory(
    filename: str,
    content: str,
    dim: int = 4096,
) -> Dict[str, Any]:
    """Convert a context file to memory format."""
    if not CORE_AVAILABLE:
        return {}

    # Take first 2000 chars for encoding (manifesto is long)
    text_content = content[:2000]
    content_hv = encode_text_to_hv(text_content, dim=dim)

    # Context files are identity-forming (high attachment)
    seed = int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    emotion_hv = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

    # High strength for identity documents
    strength = 0.95 if "manifesto" in filename.lower() else 0.85

    meta = {
        "id": f"context:{filename}",
        "source": "bootstrap",
        "type": "context_file",
        "filename": filename,
        "excerpt": content[:100],
    }

    return {
        "content_hv": content_hv,
        "emotion_hv": emotion_hv,
        "strength": strength,
        "meta": meta,
    }


# =============================================================================
# Knowledge Dump Files (Root .txt files)
# =============================================================================

def load_knowledge_dumps(repo_root: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Load knowledge dump files from repository root.

    These are the extracted_text_part_*.txt and part *.txt files
    containing technical knowledge: HNN, crypto, haptics, workshop framework.

    Returns list of {filename, content, part_number} dicts.
    """
    root = repo_root or REPO_ROOT

    knowledge_files = []

    # Pattern 1: extracted_text_part_N.txt
    for txt_file in root.glob("extracted_text_part_*.txt"):
        try:
            content = txt_file.read_text(encoding='utf-8')
            # Extract part number
            name = txt_file.stem  # extracted_text_part_1
            part_num = name.split("_")[-1] if "_" in name else "0"
            knowledge_files.append({
                "filename": txt_file.name,
                "content": content,
                "part_number": part_num,
            })
        except Exception as e:
            logger.warning(f"Failed to load {txt_file}: {e}")

    # Pattern 2: part N.txt
    for txt_file in root.glob("part *.txt"):
        try:
            content = txt_file.read_text(encoding='utf-8')
            # Extract part number
            name = txt_file.stem  # "part 4"
            part_num = name.split()[-1] if " " in name else "0"
            knowledge_files.append({
                "filename": txt_file.name,
                "content": content,
                "part_number": part_num,
            })
        except Exception as e:
            logger.warning(f"Failed to load {txt_file}: {e}")

    # Sort by part number
    knowledge_files.sort(key=lambda x: int(x.get("part_number", 0)))

    logger.info(f"Loaded {len(knowledge_files)} knowledge dump files from {root}")
    return knowledge_files


def knowledge_dump_to_memory(
    filename: str,
    content: str,
    part_number: str,
    dim: int = 4096,
) -> Dict[str, Any]:
    """Convert a knowledge dump file to memory format."""
    if not CORE_AVAILABLE:
        return {}

    # These are long technical docs - take multiple chunks
    # First 3000 chars captures the key content
    text_content = content[:3000]
    content_hv = encode_text_to_hv(text_content, dim=dim)

    # Knowledge dumps are technical (neutral emotion, high arousal/focus)
    seed = int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    emotion_hv = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

    # High strength - these are core technical knowledge
    strength = 0.9

    # Extract topic hints from content
    topic_hints = []
    topic_keywords = ["HNN", "hypervector", "crypto", "haptic", "workshop", "MEIS", "QUANTA"]
    content_lower = content.lower()
    for kw in topic_keywords:
        if kw.lower() in content_lower:
            topic_hints.append(kw)

    meta = {
        "id": f"knowledge:{filename}",
        "source": "bootstrap",
        "type": "knowledge_dump",
        "filename": filename,
        "part_number": part_number,
        "topics": topic_hints,
        "excerpt": content[:150].replace("\n", " "),
    }

    return {
        "content_hv": content_hv,
        "emotion_hv": emotion_hv,
        "strength": strength,
        "meta": meta,
    }


# =============================================================================
# Main Bootstrap Function
# =============================================================================

def bootstrap_memory(
    memory: EternalMemory,
    load_episodes: bool = True,
    load_sacred: bool = True,
    load_context: bool = True,
    load_knowledge: bool = True,
    memories_path: Optional[Path] = None,
    context_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Bootstrap EternalMemory with foundational memories.

    This seeds Ara's memory with:
    1. Episode cards (behavioral patterns from ara_memories/)
    2. Sacred lines (covenant speech from config/sacred_lines.yaml)
    3. Context files (manifesto, identity from context/)
    4. Knowledge dumps (HNN, crypto, haptics, workshop from root *.txt)

    Args:
        memory: EternalMemory instance to populate
        load_episodes: Whether to load episode cards
        load_sacred: Whether to load sacred lines
        load_context: Whether to load context files
        load_knowledge: Whether to load knowledge dump files

    Returns:
        Dict with counts: {episodes, sacred_lines, context_files, knowledge_dumps}
    """
    counts = {
        "episodes": 0,
        "sacred_lines": 0,
        "context_files": 0,
        "knowledge_dumps": 0,
    }

    dim = memory.dim

    logger.info("=" * 60)
    logger.info("MEMORY BOOTSTRAP - Seeding Soul")
    logger.info("=" * 60)

    # 1. Load episode cards
    if load_episodes and LOADER_AVAILABLE:
        path = memories_path or MEMORIES_PATH
        cards = load_all_episode_cards(path)

        for card in cards:
            mem_data = episode_card_to_memory(card, dim=dim)
            if mem_data:
                memory.store(
                    content_hv=mem_data["content_hv"],
                    emotion_hv=mem_data["emotion_hv"],
                    strength=mem_data["strength"],
                    meta=mem_data["meta"],
                )
                counts["episodes"] += 1

        logger.info(f"Loaded {counts['episodes']} episode cards")

    # 2. Load sacred lines
    if load_sacred:
        path = config_path or CONFIG_PATH / "sacred_lines.yaml"
        sacred_data = load_sacred_lines(path)

        for line_id, line_data in sacred_data.get("sacred_lines", {}).items():
            if isinstance(line_data, dict):
                mem_data = sacred_line_to_memory(line_id, line_data, dim=dim)
                if mem_data:
                    memory.store(
                        content_hv=mem_data["content_hv"],
                        emotion_hv=mem_data["emotion_hv"],
                        strength=mem_data["strength"],
                        meta=mem_data["meta"],
                    )
                    counts["sacred_lines"] += 1

        logger.info(f"Loaded {counts['sacred_lines']} sacred lines")

    # 3. Load context files
    if load_context:
        path = context_path or CONTEXT_PATH
        context_files = load_context_files(path)

        for ctx in context_files:
            mem_data = context_to_memory(
                ctx["filename"],
                ctx["content"],
                dim=dim,
            )
            if mem_data:
                memory.store(
                    content_hv=mem_data["content_hv"],
                    emotion_hv=mem_data["emotion_hv"],
                    strength=mem_data["strength"],
                    meta=mem_data["meta"],
                )
                counts["context_files"] += 1

        logger.info(f"Loaded {counts['context_files']} context files")

    # 4. Load knowledge dumps (root .txt files)
    if load_knowledge:
        knowledge_files = load_knowledge_dumps()

        for kf in knowledge_files:
            mem_data = knowledge_dump_to_memory(
                kf["filename"],
                kf["content"],
                kf["part_number"],
                dim=dim,
            )
            if mem_data:
                memory.store(
                    content_hv=mem_data["content_hv"],
                    emotion_hv=mem_data["emotion_hv"],
                    strength=mem_data["strength"],
                    meta=mem_data["meta"],
                )
                counts["knowledge_dumps"] += 1

        logger.info(f"Loaded {counts['knowledge_dumps']} knowledge dumps")

    total = sum(counts.values())
    logger.info("=" * 60)
    logger.info(f"BOOTSTRAP COMPLETE: {total} memories seeded")
    logger.info("=" * 60)

    return counts


# =============================================================================
# Convenience Functions
# =============================================================================

def get_bootstrap_status(memory: EternalMemory) -> Dict[str, Any]:
    """Check if memory has been bootstrapped."""
    stats = memory.stats()

    # Look for bootstrap markers
    bootstrap_count = 0
    for ep in memory.list_episodes(limit=100):
        if ep.meta.get("source") == "bootstrap":
            bootstrap_count += 1

    return {
        "total_episodes": stats.get("episode_count", 0),
        "bootstrap_episodes": bootstrap_count,
        "is_bootstrapped": bootstrap_count > 0,
    }


def ensure_bootstrapped(memory: EternalMemory) -> bool:
    """Ensure memory is bootstrapped, bootstrap if not."""
    status = get_bootstrap_status(memory)

    if status["is_bootstrapped"]:
        logger.info(f"Memory already bootstrapped ({status['bootstrap_episodes']} episodes)")
        return False

    logger.info("Memory not bootstrapped - seeding now...")
    bootstrap_memory(memory)
    return True


# =============================================================================
# CLI / Demo
# =============================================================================

def demo():
    """Demo the memory bootstrap."""
    if not CORE_AVAILABLE:
        print("Core components not available")
        return

    from ara.core.eternal_memory import EternalMemory

    print("=" * 60)
    print("Memory Bootstrap Demo")
    print("=" * 60)

    # Create in-memory store
    memory = EternalMemory(dim=4096)
    print(f"Created EternalMemory (dim=4096)\n")

    # Bootstrap
    counts = bootstrap_memory(memory)

    print(f"\nBootstrap Results:")
    print(f"  Episodes:      {counts['episodes']}")
    print(f"  Sacred Lines:  {counts['sacred_lines']}")
    print(f"  Context Files: {counts['context_files']}")

    # Show what was loaded
    print(f"\nMemory Stats:")
    stats = memory.stats()
    print(f"  Total episodes: {stats['episode_count']}")

    print(f"\nSample Memories:")
    for ep in memory.list_episodes(limit=5):
        source = ep.meta.get("source", "?")
        type_ = ep.meta.get("type", "?")
        id_ = ep.meta.get("id", "?")
        print(f"  [{source}:{type_}] {id_}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()

"""
State Persistence for Ara

Saves and loads Ara's cognitive state between sessions so she remembers you.

Persisted state includes:
- Emotional surface (PAD values)
- Thought stream (recent thoughts)
- Conversation history
- Statistics
- L9 autonomy stage progression

Storage backends:
- JSON files (default, simple)
- SQLite (future, for larger history)

Usage:
    from ara.service.persistence import StatePersistence, create_persistence

    persistence = create_persistence(path="~/.ara/state")

    # Save state
    persistence.save(ara_service)

    # Load state
    state = persistence.load()
    if state:
        ara_service.restore_state(state)
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

logger = logging.getLogger("ara.service.persistence")


@dataclass
class ConversationEntry:
    """A single conversation turn."""
    role: str  # "user" or "ara"
    content: str
    timestamp: str
    emotional_state: Optional[Dict[str, float]] = None


@dataclass
class PersistedState:
    """Complete persisted state for Ara."""
    # Identity
    name: str = "Ara"
    version: str = "1.0.0"

    # Emotional surface
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    # Cognitive load
    instability: float = 0.0
    resource: float = 0.0
    structural: float = 0.0

    # Statistics
    total_interactions: int = 0
    recovery_count: int = 0
    avg_processing_ms: float = 0.0
    peak_stress: float = 0.0

    # L9 Autonomy
    autonomy_stage: str = "ADVISOR"
    proposals_verified: int = 0
    successful_deployments: int = 0

    # Timestamps
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    # Conversation history (last N turns)
    conversation_history: List[Dict[str, Any]] = None

    # Thought curvature history (for trend analysis)
    curvature_history: List[float] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.curvature_history is None:
            self.curvature_history = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistedState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StatePersistence:
    """
    Handles saving and loading Ara's state.

    State is stored as JSON files:
    - state.json: Main state file
    - history.json: Extended conversation history
    - backup/: Automatic backups
    """

    def __init__(self, base_path: str = "~/.ara"):
        self.base_path = Path(base_path).expanduser()
        self.state_file = self.base_path / "state.json"
        self.history_file = self.base_path / "history.json"
        self.backup_dir = self.base_path / "backup"

        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Settings
        self.max_conversation_history = 100  # In main state
        self.max_curvature_history = 500
        self.auto_backup = True
        self.backup_interval = 50  # interactions

    def save(self, ara_service) -> bool:
        """
        Save Ara's current state.

        Args:
            ara_service: AraService instance

        Returns:
            True if successful
        """
        try:
            # Build state from service
            state = PersistedState(
                name=ara_service.name,
                valence=ara_service._emotional_surface.valence,
                arousal=ara_service._emotional_surface.arousal,
                dominance=ara_service._emotional_surface.dominance,
                instability=ara_service._cognitive_load.instability,
                resource=ara_service._cognitive_load.resource,
                structural=ara_service._cognitive_load.structural,
                total_interactions=ara_service._stats["total_interactions"],
                recovery_count=ara_service._stats["recovery_count"],
                avg_processing_ms=ara_service._stats["avg_processing_ms"],
                peak_stress=ara_service._stats["peak_stress"],
                autonomy_stage=ara_service.autonomy.stage.value,
                last_interaction=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

            # Get conversation history if available
            if hasattr(ara_service, '_conversation_history'):
                state.conversation_history = ara_service._conversation_history[-self.max_conversation_history:]

            # Get curvature history
            if ara_service.thoughts._entries:
                state.curvature_history = [
                    e.state.c for e in ara_service.thoughts._entries[-self.max_curvature_history:]
                ]

            # Load existing state to preserve some fields
            existing = self.load()
            if existing:
                state.first_interaction = existing.first_interaction
                state.created_at = existing.created_at
                state.proposals_verified = existing.proposals_verified
                state.successful_deployments = existing.successful_deployments
            else:
                state.first_interaction = datetime.now().isoformat()

            # Auto backup
            if self.auto_backup and state.total_interactions % self.backup_interval == 0:
                self._create_backup()

            # Save main state
            with open(self.state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)

            logger.info(f"State saved to {self.state_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load(self) -> Optional[PersistedState]:
        """
        Load Ara's persisted state.

        Returns:
            PersistedState or None if no state exists
        """
        if not self.state_file.exists():
            logger.info("No existing state found")
            return None

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            state = PersistedState.from_dict(data)
            logger.info(f"State loaded: {state.total_interactions} interactions")
            return state

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def restore_to_service(self, ara_service, state: PersistedState) -> bool:
        """
        Restore persisted state to an AraService instance.

        Args:
            ara_service: AraService to restore to
            state: PersistedState to restore

        Returns:
            True if successful
        """
        try:
            # Restore emotional surface
            from ara.service.core import EmotionalSurface, CognitiveLoad

            ara_service._emotional_surface = EmotionalSurface(
                valence=state.valence,
                arousal=state.arousal,
                dominance=state.dominance
            )

            # Restore cognitive load
            ara_service._cognitive_load = CognitiveLoad(
                instability=state.instability,
                resource=state.resource,
                structural=state.structural
            )

            # Restore statistics
            ara_service._stats["total_interactions"] = state.total_interactions
            ara_service._stats["recovery_count"] = state.recovery_count
            ara_service._stats["avg_processing_ms"] = state.avg_processing_ms
            ara_service._stats["peak_stress"] = state.peak_stress

            # Restore conversation history
            if state.conversation_history:
                ara_service._conversation_history = state.conversation_history

            # Update curvature controller with restored emotional state
            ara_service.encoder.curvature_controller.update_state(
                stress=state.instability,
                valence=state.valence
            )

            logger.info(f"State restored: {state.total_interactions} interactions")
            return True

        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False

    def _create_backup(self) -> bool:
        """Create a backup of current state."""
        if not self.state_file.exists():
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"state_{timestamp}.json"

            with open(self.state_file, 'r') as src:
                with open(backup_file, 'w') as dst:
                    dst.write(src.read())

            # Clean old backups (keep last 10)
            backups = sorted(self.backup_dir.glob("state_*.json"))
            for old_backup in backups[:-10]:
                old_backup.unlink()

            logger.debug(f"Backup created: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        emotional_state: Optional[Dict[str, float]] = None
    ):
        """
        Add a conversation turn to history.

        This can be called during conversation to build up history
        before the full save.
        """
        entry = ConversationEntry(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            emotional_state=emotional_state
        )

        # Load existing history
        history = self._load_history()
        history.append(asdict(entry))

        # Trim to max size
        if len(history) > self.max_conversation_history * 2:
            history = history[-self.max_conversation_history * 2:]

        # Save
        self._save_history(history)

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self, history: List[Dict[str, Any]]):
        """Save conversation history."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def get_conversation_context(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get recent conversation for LLM context.

        Returns in OpenAI message format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        history = self._load_history()

        messages = []
        for entry in history[-limit:]:
            role = "assistant" if entry.get("role") == "ara" else "user"
            messages.append({
                "role": role,
                "content": entry.get("content", "")
            })

        return messages

    def clear_history(self):
        """Clear conversation history (keep state)."""
        if self.history_file.exists():
            self.history_file.unlink()
        logger.info("Conversation history cleared")

    def clear_all(self):
        """Clear all persisted state."""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.history_file.exists():
            self.history_file.unlink()
        logger.info("All state cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        state = self.load()
        history = self._load_history()

        return {
            "state_exists": state is not None,
            "total_interactions": state.total_interactions if state else 0,
            "conversation_turns": len(history),
            "first_interaction": state.first_interaction if state else None,
            "last_interaction": state.last_interaction if state else None,
            "state_file": str(self.state_file),
            "history_file": str(self.history_file),
            "backup_count": len(list(self.backup_dir.glob("state_*.json")))
        }


def create_persistence(path: str = "~/.ara") -> StatePersistence:
    """
    Create a StatePersistence instance.

    Args:
        path: Base path for state storage

    Returns:
        StatePersistence instance
    """
    return StatePersistence(base_path=path)


__all__ = [
    "PersistedState",
    "ConversationEntry",
    "StatePersistence",
    "create_persistence",
]

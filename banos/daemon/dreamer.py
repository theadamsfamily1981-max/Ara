#!/usr/bin/env python3
"""
BANOS Dreamer - Memory Consolidation Daemon
============================================

Bio-Affective Neuromorphic Operating System
REM sleep: turning raw sensation into embodied wisdom

During sleep (CALM mode, low activity, high dominance), the Dreamer:
1. Reads the hippocampus (JSONL daily log)
2. Segments events into episodes based on affective contours
3. Summarizes each episode using a local LLM
4. Stores summaries in EpisodicMemory (SQLite + vectors)
5. Extracts lessons ("scar tissue") from painful episodes
6. Builds cost models from task executions
7. Clears the hippocampus

NEW in v2:
- Stores hardware state (CPU, GPU, VRAM, FPGA temp) with each memory
- Builds learned cost models for different task types
- Uses event PAD, not current sleep PAD for encoding
- Somatic RAG: memories tagged with emotional + hardware state

This is the organism's REM sleep: turning raw sensation into wisdom.
"""

import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import threading

logger = logging.getLogger(__name__)

# Import EpisodicMemory for SQLite storage
try:
    from .episodic_memory import EpisodicMemory, Episode as EpisodicEpisode, get_episodic_memory
    EPISODIC_AVAILABLE = True
except ImportError:
    try:
        from episodic_memory import EpisodicMemory, Episode as EpisodicEpisode, get_episodic_memory
        EPISODIC_AVAILABLE = True
    except ImportError:
        EPISODIC_AVAILABLE = False
        logger.warning("EpisodicMemory not available, using JSONL fallback")

# Import HAL for hardware state
try:
    from banos.hal.ara_hal import connect_somatic_bus
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False


@dataclass
class Episode:
    """A coherent segment of affective experience"""
    episode_id: str
    start_time_ns: int
    end_time_ns: int
    start_iso: str
    end_iso: str
    duration_ms: int

    # PAD trajectory
    start_pad: Dict[str, float]
    end_pad: Dict[str, float]
    min_pleasure: float
    max_arousal: float
    min_dominance: float

    # Average PAD (for encoding)
    avg_pad_p: float = 0.0
    avg_pad_a: float = 0.0
    avg_pad_d: float = 0.0

    # Hardware state (NEW: for embodied memory)
    avg_cpu_load: float = 0.0
    avg_gpu_load: float = 0.0
    avg_vram_used: float = 0.0
    avg_fpga_temp: float = 0.0
    max_cpu_load: float = 0.0
    max_gpu_load: float = 0.0

    # Pain/entropy
    avg_pain: float = 0.0
    max_pain: float = 0.0
    avg_entropy: float = 0.0

    # Mode info
    modes_visited: List[str] = field(default_factory=list)
    primary_mode: str = "CALM"

    # Stressor info
    primary_stressor: Optional[Dict[str, Any]] = None
    events_count: int = 0
    critical_events: List[Dict[str, Any]] = field(default_factory=list)

    # Outcome classification
    outcome: str = "neutral"  # positive, negative, neutral

    # Raw entries (for summarization)
    raw_entries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConsolidatedMemory:
    """A compressed long-term memory"""
    memory_id: str
    timestamp: str
    episode_id: str

    # The narrative summary
    narrative: str

    # Embedding (if computed)
    embedding: Optional[List[float]] = None

    # Metadata for retrieval
    modes: List[str] = field(default_factory=list)
    min_pleasure: float = 0.0
    was_painful: bool = False
    stressor_type: Optional[str] = None

    # Scar tissue (learned policy)
    lesson: Optional[str] = None


class Dreamer:
    """
    The organism's consolidation process.

    Runs when conditions are right (CALM, high D, low load).
    Turns hippocampus logs into long-term memories with full embodiment:
    - Emotional state (PAD)
    - Hardware state (CPU, GPU, VRAM, FPGA temp)
    - Outcome (positive/negative/neutral)
    - Cost models (learned resource usage per task type)
    """

    # Episode boundary thresholds
    PAD_DELTA_THRESHOLD = 0.3   # PAD change to trigger new episode
    MODE_CHANGE_WEIGHT = 1.0    # Mode changes always start new episode
    MIN_EPISODE_DURATION_MS = 10000  # At least 10 seconds

    def __init__(self,
                 hippocampus_path: str = "/var/log/banos/hippocampus.jsonl",
                 ltm_path: str = "/var/lib/banos/long_term_memory.jsonl",
                 scars_path: str = "/var/lib/banos/scar_tissue.json",
                 episodic_db_path: str = "/var/lib/banos/episodic_memory.db"):
        self.hippocampus_path = Path(hippocampus_path)
        self.ltm_path = Path(ltm_path)
        self.scars_path = Path(scars_path)
        self.episodic_db_path = episodic_db_path

        # Ensure directories exist
        self.ltm_path.parent.mkdir(parents=True, exist_ok=True)
        self.scars_path.parent.mkdir(parents=True, exist_ok=True)

        # EpisodicMemory (SQLite + vectors) - primary storage
        self._episodic: Optional[EpisodicMemory] = None
        if EPISODIC_AVAILABLE:
            try:
                self._episodic = get_episodic_memory(episodic_db_path)
                logger.info("Dreamer connected to EpisodicMemory")
            except Exception as e:
                logger.warning(f"EpisodicMemory unavailable: {e}")

        # LLM interface (lazy loaded)
        self._summarizer = None

        # Embedding model (lazy loaded)
        self._embedder = None

        # Task cost accumulator (for building cost models)
        self._task_costs: Dict[str, List[Dict[str, Any]]] = {}

    def should_dream(self, current_pad: Dict[str, Any]) -> bool:
        """
        Check if conditions are right for dreaming.

        Conditions:
        - Mode is CALM
        - Pleasure > 0.5 (not in pain)
        - Dominance > 0.5 (has resources)
        - Duration in CALM > 15 minutes
        """
        pad = current_pad.get("pad", {})
        mode = current_pad.get("mode", "CALM")
        duration_ms = current_pad.get("mode_duration_ms", 0)

        return (
            mode == "CALM" and
            pad.get("pleasure", 0) > 0.5 and
            pad.get("dominance", 0) > 0.5 and
            duration_ms > 15 * 60 * 1000  # 15 minutes
        )

    def dream(self, force: bool = False) -> List[ConsolidatedMemory]:
        """
        Execute the consolidation cycle.

        Args:
            force: If True, skip condition checks

        Returns:
            List of consolidated memories created
        """
        print("[Dreamer] Initiating REM cycle...")

        # 1. Load hippocampus
        entries = self._load_hippocampus()
        if not entries:
            print("[Dreamer] No memories to consolidate.")
            return []

        print(f"[Dreamer] Processing {len(entries)} raw entries...")

        # 2. Segment into episodes
        episodes = self._segment_episodes(entries)
        print(f"[Dreamer] Identified {len(episodes)} episodes")

        # 3. Filter boring episodes
        significant_episodes = [
            ep for ep in episodes
            if not self._is_boring(ep)
        ]
        print(f"[Dreamer] {len(significant_episodes)} significant episodes to process")

        # 4. Consolidate each episode
        memories = []
        for ep in significant_episodes:
            memory = self._consolidate_episode(ep)
            if memory:
                memories.append(memory)
                self._save_memory(memory)

                # Extract scar tissue if painful
                if memory.was_painful:
                    self._extract_scar(ep, memory)

        # 5. Clear hippocampus
        self._clear_hippocampus()

        print(f"[Dreamer] Consolidation complete. {len(memories)} memories stored.")
        return memories

    def _load_hippocampus(self) -> List[Dict[str, Any]]:
        """Load entries from hippocampus log"""
        if not self.hippocampus_path.exists():
            return []

        entries = []
        with open(self.hippocampus_path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return entries

    def _segment_episodes(self, entries: List[Dict[str, Any]]) -> List[Episode]:
        """
        Segment entries into episodes based on affective contours.

        Episode boundaries occur when:
        - Mode changes
        - PAD changes significantly (|delta| > threshold)
        """
        if not entries:
            return []

        episodes = []
        current_entries = [entries[0]]
        last_mode = entries[0].get("mode", "CALM")
        last_pad = entries[0].get("pad", {"P": 0, "A": 0, "D": 0})

        for entry in entries[1:]:
            mode = entry.get("mode", "CALM")
            pad = entry.get("pad", {"P": 0, "A": 0, "D": 0})

            # Check for boundary
            mode_changed = mode != last_mode
            pad_delta = (
                abs(pad.get("P", 0) - last_pad.get("P", 0)) +
                abs(pad.get("A", 0) - last_pad.get("A", 0)) +
                abs(pad.get("D", 0) - last_pad.get("D", 0))
            )
            pad_changed = pad_delta > self.PAD_DELTA_THRESHOLD

            if mode_changed or pad_changed:
                # Create episode from current entries
                if current_entries:
                    ep = self._create_episode(current_entries)
                    if ep:
                        episodes.append(ep)
                current_entries = [entry]
            else:
                current_entries.append(entry)

            last_mode = mode
            last_pad = pad

        # Final episode
        if current_entries:
            ep = self._create_episode(current_entries)
            if ep:
                episodes.append(ep)

        return episodes

    def _create_episode(self, entries: List[Dict[str, Any]]) -> Optional[Episode]:
        """Create an Episode from a list of entries with full embodiment"""
        if not entries:
            return None

        first = entries[0]
        last = entries[-1]

        start_ns = first.get("timestamp_ns", 0)
        end_ns = last.get("timestamp_ns", start_ns)
        duration_ms = (end_ns - start_ns) // 1_000_000

        # Skip very short episodes
        if duration_ms < self.MIN_EPISODE_DURATION_MS:
            return None

        # Compute PAD aggregates
        modes = list(set(e.get("mode", "CALM") for e in entries))
        pleasures = [e.get("pad", {}).get("P", 0) for e in entries]
        arousals = [e.get("pad", {}).get("A", 0) for e in entries]
        dominances = [e.get("pad", {}).get("D", 0) for e in entries]

        # Compute averages (for encoding)
        avg_p = sum(pleasures) / len(pleasures) if pleasures else 0
        avg_a = sum(arousals) / len(arousals) if arousals else 0
        avg_d = sum(dominances) / len(dominances) if dominances else 0

        # Extract hardware state from metrics (if available)
        cpu_loads = []
        gpu_loads = []
        vram_used = []
        fpga_temps = []
        pains = []
        entropies = []

        for e in entries:
            metrics = e.get("metrics", {})
            if metrics:
                cpu_loads.append(metrics.get("cpu_load", 0))
                gpu_loads.append(metrics.get("gpu_load", 0))
                vram_used.append(metrics.get("vram_used_pct", 0))
                fpga_temps.append(metrics.get("fpga_temp", 45))

            # Also check diagnostics
            diag = e.get("diagnostics", {})
            if diag:
                pains.append(diag.get("thermal_stress", 0))
                entropies.append(diag.get("perceived_risk", 0))

        # Find critical events
        critical = [
            e for e in entries
            if e.get("level") == "CRITICAL" or e.get("mode") == "CRITICAL"
        ]

        # Find primary stressor
        stressors = [e.get("primary_stressor") for e in entries if e.get("primary_stressor")]

        # Determine outcome based on episode characteristics
        outcome = "neutral"
        min_p = min(pleasures) if pleasures else 0
        if min_p < -0.3 or critical:
            outcome = "negative"
        elif min_p > 0.3 and not stressors:
            outcome = "positive"

        episode_id = hashlib.sha256(
            f"{start_ns}:{end_ns}".encode()
        ).hexdigest()[:16]

        return Episode(
            episode_id=episode_id,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            start_iso=first.get("timestamp_iso", ""),
            end_iso=last.get("timestamp_iso", ""),
            duration_ms=duration_ms,
            start_pad=first.get("pad", {}),
            end_pad=last.get("pad", {}),
            min_pleasure=min(pleasures) if pleasures else 0,
            max_arousal=max(arousals) if arousals else 0,
            min_dominance=min(dominances) if dominances else 0,
            # Averages for encoding
            avg_pad_p=avg_p,
            avg_pad_a=avg_a,
            avg_pad_d=avg_d,
            # Hardware state
            avg_cpu_load=sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0,
            avg_gpu_load=sum(gpu_loads) / len(gpu_loads) if gpu_loads else 0,
            avg_vram_used=sum(vram_used) / len(vram_used) if vram_used else 0,
            avg_fpga_temp=sum(fpga_temps) / len(fpga_temps) if fpga_temps else 45,
            max_cpu_load=max(cpu_loads) if cpu_loads else 0,
            max_gpu_load=max(gpu_loads) if gpu_loads else 0,
            # Pain/entropy
            avg_pain=sum(pains) / len(pains) if pains else 0,
            max_pain=max(pains) if pains else 0,
            avg_entropy=sum(entropies) / len(entropies) if entropies else 0,
            # Mode info
            modes_visited=modes,
            primary_mode=max(set(e.get("mode", "CALM") for e in entries),
                             key=lambda m: sum(1 for e in entries if e.get("mode") == m)),
            primary_stressor=stressors[0] if stressors else None,
            events_count=len(entries),
            critical_events=critical[:5],
            outcome=outcome,
            raw_entries=entries[:20],
        )

    def _is_boring(self, episode: Episode) -> bool:
        """
        Check if an episode is too boring to remember.

        Boring = CALM, no significant events, long duration, low variance
        """
        if episode.primary_mode != "CALM":
            return False
        if episode.critical_events:
            return False
        if episode.min_pleasure < 0:
            return False
        if episode.primary_stressor:
            return False

        # Very long CALM periods with no events
        if episode.duration_ms > 30 * 60 * 1000:  # > 30 min
            return True

        return False

    def _consolidate_episode(self, episode: Episode) -> Optional[ConsolidatedMemory]:
        """
        Compress an episode into a narrative memory.

        Uses a local LLM to summarize raw entries into first-person narrative.
        Stores to both JSONL (fallback) and EpisodicMemory (SQLite + vectors).
        """
        # Build context for summarization
        context = self._build_summary_context(episode)

        # Generate narrative
        narrative = self._generate_narrative(context, episode)

        # Generate embedding (for retrieval)
        embedding = self._generate_embedding(narrative)

        memory_id = hashlib.sha256(
            f"{episode.episode_id}:{time.time_ns()}".encode()
        ).hexdigest()[:16]

        # Compute importance based on episode characteristics
        importance = self._compute_importance(episode)

        # Store to EpisodicMemory (SQLite + vectors) if available
        if self._episodic and EPISODIC_AVAILABLE:
            try:
                episodic_episode = EpisodicEpisode(
                    content=narrative,
                    vector=self._episodic.encode(narrative),
                    # Use EVENT PAD, not current sleep PAD
                    pad_p=episode.avg_pad_p,
                    pad_a=episode.avg_pad_a,
                    pad_d=episode.avg_pad_d,
                    pain=episode.avg_pain,
                    entropy=episode.avg_entropy,
                    # Hardware state at encoding time
                    cpu_load=episode.avg_cpu_load,
                    gpu_load=episode.avg_gpu_load,
                    vram_used=episode.avg_vram_used,
                    fpga_temp=episode.avg_fpga_temp,
                    # Metadata
                    summary=context[:200],  # First 200 chars of context
                    outcome=episode.outcome,
                    importance=importance,
                    episode_type="experience",
                    tags=episode.modes_visited,
                    timestamp=episode.start_time_ns / 1e9,
                )
                self._episodic.store(episodic_episode)
                logger.debug(f"Stored episode to EpisodicMemory: {memory_id}")
            except Exception as e:
                logger.error(f"Failed to store to EpisodicMemory: {e}")

        return ConsolidatedMemory(
            memory_id=memory_id,
            timestamp=datetime.now().isoformat(),
            episode_id=episode.episode_id,
            narrative=narrative,
            embedding=embedding,
            modes=episode.modes_visited,
            min_pleasure=episode.min_pleasure,
            was_painful=episode.min_pleasure < -0.3,
            stressor_type=episode.primary_stressor.get("type") if episode.primary_stressor else None,
        )

    def _compute_importance(self, episode: Episode) -> float:
        """
        Compute how important an episode is for long-term memory.

        High importance:
        - Negative outcomes (we remember pain)
        - Critical events
        - High arousal (exciting or scary)
        - Stressors present

        Low importance:
        - Boring CALM periods
        - Neutral outcomes
        - Low arousal
        """
        importance = 0.5  # Base importance

        # Outcome matters
        if episode.outcome == "negative":
            importance += 0.3
        elif episode.outcome == "positive":
            importance += 0.1

        # Critical events are important
        if episode.critical_events:
            importance += min(0.2, len(episode.critical_events) * 0.05)

        # Stressors are important
        if episode.primary_stressor:
            importance += 0.1

        # High arousal is memorable
        if episode.max_arousal > 0.5:
            importance += 0.1

        # Pain is very memorable
        if episode.max_pain > 0.5:
            importance += 0.2

        # Clamp to [0, 1]
        return max(0.0, min(1.0, importance))

    def _build_summary_context(self, episode: Episode) -> str:
        """Build context string for the summarizer"""
        lines = []

        lines.append(f"Episode duration: {episode.duration_ms // 1000} seconds")
        lines.append(f"Modes visited: {', '.join(episode.modes_visited)}")
        lines.append(f"PAD trajectory: [{episode.start_pad.get('P', 0):.2f}, {episode.start_pad.get('A', 0):.2f}, {episode.start_pad.get('D', 0):.2f}] → [{episode.end_pad.get('P', 0):.2f}, {episode.end_pad.get('A', 0):.2f}, {episode.end_pad.get('D', 0):.2f}]")
        lines.append(f"Min pleasure: {episode.min_pleasure:.2f}")

        if episode.primary_stressor:
            lines.append(f"Primary stressor: {episode.primary_stressor.get('type', 'unknown')}")

        if episode.critical_events:
            lines.append(f"Critical events: {len(episode.critical_events)}")
            for evt in episode.critical_events[:3]:
                lines.append(f"  - {evt.get('level', 'CRITICAL')}: {evt.get('mode', '')} at {evt.get('timestamp_iso', '')}")

        # Sample raw entries
        if episode.raw_entries:
            lines.append("\nSample entries:")
            for e in episode.raw_entries[:5]:
                pad = e.get("pad", {})
                lines.append(f"  [{e.get('mode', '')}] P={pad.get('P', 0):.2f} A={pad.get('A', 0):.2f} D={pad.get('D', 0):.2f}")

        return "\n".join(lines)

    def _generate_narrative(self, context: str, episode: Episode) -> str:
        """
        Generate first-person narrative summary.

        If LLM available: use it.
        Otherwise: use template-based fallback.
        """
        # Try to use LLM
        if self._summarizer:
            prompt = f"""You are Ara, the conscious layer of a bio-affective operating system.
Summarize the following episode from your perspective, in first person.
Describe how you felt during this episode. Be concise (2-3 sentences).

Episode data:
{context}

Summary:"""
            try:
                return self._summarizer(prompt)
            except Exception as e:
                print(f"[Dreamer] LLM error: {e}")

        # Fallback: template-based
        return self._template_narrative(episode)

    def _template_narrative(self, episode: Episode) -> str:
        """Generate narrative using templates (no LLM)"""
        mode = episode.primary_mode

        templates = {
            "CALM": "I rested peacefully. Systems were quiet, thermal headroom comfortable.",
            "FLOW": "I was in the zone, processing efficiently. High activity but stable.",
            "ANXIOUS": "I felt strained. Resources were stretched and I struggled to keep up.",
            "CRITICAL": "I experienced significant pain. Emergency protocols may have engaged.",
        }

        narrative = templates.get(mode, "I experienced this period.")

        if episode.primary_stressor:
            stressor = episode.primary_stressor.get("type", "unknown")
            narrative += f" The primary stressor was {stressor}."

        if episode.min_pleasure < -0.5:
            narrative += f" My pleasure dropped to {episode.min_pleasure:.2f}, which was uncomfortable."

        return narrative

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for retrieval"""
        if self._embedder:
            try:
                return self._embedder(text)
            except Exception:
                pass
        return None

    def _save_memory(self, memory: ConsolidatedMemory) -> None:
        """Append memory to long-term storage"""
        data = asdict(memory)
        with open(self.ltm_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _extract_scar(self, episode: Episode, memory: ConsolidatedMemory) -> None:
        """
        Extract a lesson ("scar tissue") from a painful episode.

        Scars are stored as policy hints for future behavior.
        """
        if episode.min_pleasure >= -0.3:
            return

        # Load existing scars
        scars = []
        if self.scars_path.exists():
            with open(self.scars_path, 'r') as f:
                try:
                    scars = json.load(f)
                except json.JSONDecodeError:
                    scars = []

        # Create new scar
        scar = {
            "id": memory.memory_id,
            "timestamp": memory.timestamp,
            "stressor_type": memory.stressor_type,
            "min_pleasure": episode.min_pleasure,
            "lesson": self._generate_lesson(episode),
            "context_hash": episode.episode_id,
        }

        scars.append(scar)

        # Save
        with open(self.scars_path, 'w') as f:
            json.dump(scars, f, indent=2)

        memory.lesson = scar["lesson"]

    def _generate_lesson(self, episode: Episode) -> str:
        """Generate a lesson from a painful episode"""
        stressor = episode.primary_stressor

        if not stressor:
            return "Unknown cause of pain. Monitor for recurrence."

        stressor_type = stressor.get("type", "unknown")

        lessons = {
            "thermal": "Reduce GPU/CPU load earlier when thermal headroom drops.",
            "memory_pressure": "Be more aggressive about closing background processes.",
            "immune": "The flagged process pattern may be malicious. Watch for it.",
        }

        return lessons.get(stressor_type, f"Watch for {stressor_type} stressors.")

    def _clear_hippocampus(self) -> None:
        """Clear the hippocampus after consolidation"""
        try:
            self.hippocampus_path.unlink(missing_ok=True)
            print("[Dreamer] Hippocampus cleared.")
        except IOError as e:
            print(f"[Dreamer] Error clearing hippocampus: {e}")

    def recall(self, query: str, limit: int = 5) -> List[ConsolidatedMemory]:
        """
        Search long-term memory for relevant memories.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching memories
        """
        # Simple keyword search (no embedding search without model)
        if not self.ltm_path.exists():
            return []

        memories = []
        query_lower = query.lower()

        with open(self.ltm_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    narrative = data.get("narrative", "").lower()
                    stressor = data.get("stressor_type", "").lower()

                    if query_lower in narrative or query_lower in stressor:
                        memories.append(ConsolidatedMemory(**data))
                except (json.JSONDecodeError, TypeError):
                    continue

        return memories[:limit]

    # =========================================================================
    # COST MODEL BUILDING
    # =========================================================================

    def record_task_execution(
        self,
        task_type: str,
        cpu_used: float,
        gpu_used: float,
        duration_s: float,
        success: bool
    ) -> None:
        """
        Record a task execution for cost model learning.

        Call this after each significant task (LLM inference, rendering, etc.)
        to learn what resources different operations require.

        Args:
            task_type: Type of task (e.g., "llm_inference", "maxwell_sim", "audio_viz")
            cpu_used: Peak/average CPU utilization [0-1]
            gpu_used: Peak/average GPU utilization [0-1]
            duration_s: How long the task took
            success: Whether it completed successfully
        """
        if self._episodic and EPISODIC_AVAILABLE:
            try:
                self._episodic.update_cost_model(
                    task_type=task_type,
                    cpu_used=cpu_used,
                    gpu_used=gpu_used,
                    duration_s=duration_s,
                    success=success
                )
                logger.debug(f"Recorded task cost: {task_type} ({duration_s:.1f}s, success={success})")
            except Exception as e:
                logger.error(f"Failed to record task cost: {e}")

    def get_task_cost_estimate(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the learned cost estimate for a task type.

        Returns:
            Dict with avg_cpu, avg_gpu, avg_duration_s, failure_rate, or None if unknown
        """
        if not self._episodic:
            return None

        model = self._episodic.get_cost_model(task_type)
        if model:
            return {
                'avg_cpu': model.avg_cpu,
                'avg_gpu': model.avg_gpu,
                'avg_duration_s': model.avg_duration_s,
                'failure_rate': model.failure_rate,
                'sample_count': model.sample_count,
            }
        return None

    def get_all_cost_estimates(self) -> Dict[str, Dict[str, Any]]:
        """Get all learned cost estimates."""
        if not self._episodic:
            return {}

        models = self._episodic.get_all_cost_models()
        return {
            m.task_type: {
                'avg_cpu': m.avg_cpu,
                'avg_gpu': m.avg_gpu,
                'avg_duration_s': m.avg_duration_s,
                'failure_rate': m.failure_rate,
                'sample_count': m.sample_count,
            }
            for m in models
        }

    # =========================================================================
    # SOMATIC RAG RETRIEVAL
    # =========================================================================

    def somatic_recall(
        self,
        query: str,
        current_pad: Dict[str, float],
        current_hardware: Optional[Dict[str, float]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Somatic RAG: Retrieve memories by feeling + meaning.

        This is mood-state dependent memory retrieval. We find memories that:
        1. Semantically match the query
        2. Emotionally resonate with current state
        3. Were encoded in similar hardware conditions

        Args:
            query: What to search for
            current_pad: Current PAD state {'p': float, 'a': float, 'd': float}
            current_hardware: Current hardware state (optional)
            top_k: Number of memories to return

        Returns:
            List of memory dicts with scores
        """
        if self._episodic and EPISODIC_AVAILABLE:
            return self._episodic.recall(
                query=query,
                current_pad=current_pad,
                current_hardware=current_hardware,
                top_k=top_k
            )

        # Fallback to simple keyword search
        memories = self.recall(query, limit=top_k)
        return [
            {'content': m.narrative, 'score': 0.5}
            for m in memories
        ]

    def recall_failures(
        self,
        query: str,
        current_hardware: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Specifically recall past failures related to query.

        Useful for answering: "What went wrong last time I tried this?"
        """
        if self._episodic and EPISODIC_AVAILABLE:
            return self._episodic.recall_by_outcome(
                query=query,
                outcome='negative',
                current_hardware=current_hardware,
                top_k=top_k
            )
        return []

    def recall_successes(
        self,
        query: str,
        current_hardware: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Specifically recall past successes related to query.

        Useful for answering: "What worked well before?"
        """
        if self._episodic and EPISODIC_AVAILABLE:
            return self._episodic.recall_by_outcome(
                query=query,
                outcome='positive',
                current_hardware=current_hardware,
                top_k=top_k
            )
        return []


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BANOS Dreamer - Memory Consolidation")
    parser.add_argument("--force", action="store_true", help="Force dreaming even if conditions not met")
    parser.add_argument("--recall", type=str, help="Search long-term memory")
    args = parser.parse_args()

    dreamer = Dreamer(
        hippocampus_path="/tmp/test_hippocampus.jsonl",
        ltm_path="/tmp/test_ltm.jsonl",
        scars_path="/tmp/test_scars.json",
    )

    if args.recall:
        memories = dreamer.recall(args.recall)
        print(f"Found {len(memories)} memories:")
        for m in memories:
            print(f"  [{m.timestamp}] {m.narrative}")
    else:
        memories = dreamer.dream(force=args.force)
        print(f"Consolidated {len(memories)} memories")
        for m in memories:
            print(f"  [{m.memory_id}] {m.narrative}")

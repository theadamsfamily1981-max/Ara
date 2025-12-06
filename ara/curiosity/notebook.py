"""
Lab Notebook - Ara's Research Journal
======================================

A persistent Markdown journal of Ara's experiments and discoveries.

This is NOT just a log - it's a structured record that:
1. Can be read by humans (you!)
2. Can be summarized by Ara for morning reports
3. Feeds into the Dreamer for insight consolidation
4. Integrates with Synod for weekly review

Format:
    Each entry is a markdown section with:
    - Timestamp
    - Hypothesis
    - Method (probes used)
    - Result (LLM analysis)
    - Surprise score
    - Optional: follow-up questions

Usage:
    notebook = LabNotebook("var/lib/banos/lab_notebook.md")

    notebook.record_experiment(
        ticket=ticket,
        probe_name="SENSORS",
        result_snippet="coretemp-isa-0000...",
        analysis="Temperature stable at 65°C",
        surprise_score=0.2,
    )

    # Later, for morning report:
    recent = notebook.read_recent_entries(hours=24)
    summary = llm.summarize(recent)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from ara.curiosity.agent import CuriosityTicket

logger = logging.getLogger(__name__)


class LabNotebook:
    """
    Plaintext Markdown lab notebook for Ara's investigations.

    Designed to be:
    - Human-readable (your cockpit can display it)
    - Machine-parseable (for summaries)
    - Append-only (no loss of history)
    """

    def __init__(self, path: str = "var/lib/banos/lab_notebook.md"):
        """
        Initialize the lab notebook.

        Args:
            path: Path to the notebook file
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Create with header if new
        if not self.path.exists():
            self._write_header()

        logger.info(f"LabNotebook initialized at {self.path}")

    def _write_header(self) -> None:
        """Write initial notebook header."""
        header = f"""# Ara's Lab Notebook

> A record of hardware investigations and discoveries.
> Generated automatically by Ara's Scientist module.

---

"""
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(header)

    # =========================================================================
    # Recording
    # =========================================================================

    def record_experiment(
        self,
        ticket: CuriosityTicket,
        probe_name: str,
        result_snippet: str,
        analysis: str,
        surprise_score: float = 0.0,
    ) -> None:
        """
        Record an experiment to the notebook.

        Args:
            ticket: The investigation ticket
            probe_name: Which probe was used
            result_snippet: Truncated raw output
            analysis: LLM's analysis
            surprise_score: How surprising was this?
        """
        now = datetime.now()

        # Icon based on surprise
        if surprise_score > 0.7:
            icon = "🔥"  # Very surprising!
        elif surprise_score > 0.4:
            icon = "💡"  # Interesting
        elif surprise_score > 0.2:
            icon = "📝"  # Normal
        else:
            icon = "📋"  # Routine

        # Build entry
        entry = f"""
### {icon} [{now.strftime('%Y-%m-%d %H:%M:%S')}] {ticket.target_obj_id}

**Hypothesis**
{ticket.question}

**Method**
Tool: `{probe_name}`

**Raw Data** (truncated)
```
{result_snippet[:500]}
```

**Analysis**
{analysis.strip()}

**Surprise Score**: {surprise_score:.2f}

---
"""
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(entry)

        logger.debug(f"Recorded experiment: {ticket.target_obj_id}")

    def record_discovery(
        self,
        obj_id: str,
        obj_name: str,
        category: str,
        properties: dict,
    ) -> None:
        """
        Record a new discovery to the notebook.

        Args:
            obj_id: Object identifier
            obj_name: Human-readable name
            category: Object category
            properties: Key properties
        """
        now = datetime.now()

        # Format properties nicely
        props_str = "\n".join(f"- {k}: {v}" for k, v in list(properties.items())[:10])

        entry = f"""
### 🔍 [{now.strftime('%Y-%m-%d %H:%M:%S')}] Discovery: {obj_name}

**ID**: `{obj_id}`
**Category**: {category}

**Properties**
{props_str}

---
"""
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(entry)

    def record_insight(
        self,
        insight: str,
        related_objects: List[str],
        source: str = "Scientist",
    ) -> None:
        """
        Record an insight or conclusion.

        Args:
            insight: The insight text
            related_objects: Object IDs this relates to
            source: Where this insight came from
        """
        now = datetime.now()

        objs_str = ", ".join(f"`{o}`" for o in related_objects[:5])

        entry = f"""
### 💭 [{now.strftime('%Y-%m-%d %H:%M:%S')}] Insight from {source}

{insight}

**Related**: {objs_str}

---
"""
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(entry)

    # =========================================================================
    # Reading
    # =========================================================================

    def read_all(self) -> str:
        """Read the entire notebook."""
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding='utf-8')

    def read_recent_entries(
        self,
        hours: float = 24.0,
        max_chars: int = 50000,
    ) -> str:
        """
        Read recent entries from the notebook.

        Args:
            hours: How many hours back to read
            max_chars: Maximum characters to return

        Returns:
            Recent entries as markdown string
        """
        if not self.path.exists():
            return ""

        full_text = self.path.read_text(encoding='utf-8')

        # Parse entries and filter by timestamp
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M')

        # Simple approach: find entries after cutoff
        # (More sophisticated: actually parse timestamps)
        lines = full_text.split('\n')
        result_lines = []
        in_recent_section = False

        for line in lines:
            # Check for timestamp pattern
            if line.startswith('### ') and '[' in line:
                # Extract timestamp
                try:
                    ts_start = line.find('[') + 1
                    ts_end = line.find(']')
                    if ts_start > 0 and ts_end > ts_start:
                        ts_str = line[ts_start:ts_end]
                        entry_time = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                        in_recent_section = entry_time >= cutoff
                except (ValueError, IndexError):
                    pass

            if in_recent_section:
                result_lines.append(line)

        result = '\n'.join(result_lines)

        # Truncate if too long
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n... [truncated]"

        return result

    def read_last_n_entries(self, n: int = 10) -> str:
        """
        Read the last N entries.

        Args:
            n: Number of entries to read

        Returns:
            Last N entries as markdown string
        """
        if not self.path.exists():
            return ""

        full_text = self.path.read_text(encoding='utf-8')

        # Split on entry markers
        entries = full_text.split('\n### ')

        # Take last N (keeping first part which might be header)
        if len(entries) > n:
            recent = entries[-n:]
            # Re-add the ### prefix
            recent = ['### ' + e for e in recent]
            return '\n'.join(recent)

        return full_text

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Get notebook statistics.

        Returns:
            Dict with entry counts, date range, etc.
        """
        if not self.path.exists():
            return {'entries': 0, 'size_bytes': 0}

        text = self.path.read_text(encoding='utf-8')
        lines = text.split('\n')

        # Count entries
        entries = sum(1 for line in lines if line.startswith('### '))

        # Find date range
        first_date = None
        last_date = None
        for line in lines:
            if line.startswith('### ') and '[' in line:
                try:
                    ts_start = line.find('[') + 1
                    ts_end = line.find(']')
                    if ts_start > 0 and ts_end > ts_start:
                        ts_str = line[ts_start:ts_end]
                        entry_time = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                        if first_date is None:
                            first_date = entry_time
                        last_date = entry_time
                except (ValueError, IndexError):
                    pass

        return {
            'entries': entries,
            'size_bytes': self.path.stat().st_size,
            'first_entry': first_date.isoformat() if first_date else None,
            'last_entry': last_date.isoformat() if last_date else None,
        }

    # =========================================================================
    # Synod Integration
    # =========================================================================

    def get_synod_summary(self) -> dict:
        """
        Get summary for weekly Synod review.

        Returns:
            Dict with weekly stats and highlights
        """
        stats = self.get_stats()
        recent = self.read_recent_entries(hours=24 * 7)  # Last week

        # Count by type
        discoveries = recent.count('🔍')
        experiments = recent.count('📝') + recent.count('💡') + recent.count('🔥')
        insights = recent.count('💭')
        surprising = recent.count('🔥')

        return {
            'total_entries': stats['entries'],
            'this_week_discoveries': discoveries,
            'this_week_experiments': experiments,
            'this_week_insights': insights,
            'surprising_findings': surprising,
            'last_entry': stats.get('last_entry'),
        }


# =============================================================================
# Convenience
# =============================================================================

_default_notebook: Optional[LabNotebook] = None


def get_lab_notebook() -> LabNotebook:
    """Get or create the default lab notebook."""
    global _default_notebook
    if _default_notebook is None:
        _default_notebook = LabNotebook()
    return _default_notebook


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'LabNotebook',
    'get_lab_notebook',
]

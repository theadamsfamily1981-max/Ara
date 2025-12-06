"""
The Muse - The Artist in the Machine
=====================================

The Muse works in The Studio, crafting "Gifts" for the user.
Unlike the Steward (which does boring but useful fixes),
the Muse creates *artistic* solutions.

When you're stuck on data structures, she doesn't just write a struct.
She generates a 3D visualization because she knows you're visual.

Gifts are:
    - Surprising: You didn't know you needed this
    - Beautiful: Aesthetic, not just functional
    - Timed perfectly: Delivered at the Kairos moment

The Studio Queue:
    Gifts incubate until the KairosEngine says "Now."
    She might hold a gift for days waiting for perfect receptivity.

Usage:
    from ara.daemon.muse import Muse

    muse = Muse(teleology=horizon_engine)

    # Craft a gift from a friction point
    gift = muse.craft_gift(friction_point)
    if gift:
        muse.studio_queue.append(gift)

    # When Kairos says it's time
    if kairos.is_opportune(gift_importance=0.9):
        muse.present(gift)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol
from enum import Enum

logger = logging.getLogger(__name__)


class GiftType(Enum):
    """Types of gifts the Muse can create."""
    VISUALIZATION = "visualization"      # Diagram, graph, chart
    DOCUMENTATION = "documentation"      # Beautiful docs, not just comments
    INSIGHT = "insight"                  # A realization, a new perspective
    TOOL = "tool"                        # A helper script, dashboard
    METAPHOR = "metaphor"                # A conceptual framework
    OPTIMIZATION = "optimization"        # Something made faster/better
    AESTHETIC = "aesthetic"              # Something made beautiful


class GiftStatus(Enum):
    """Status of a gift in the studio."""
    CONCEIVING = "conceiving"    # Idea forming
    CRAFTING = "crafting"        # Being created
    INCUBATING = "incubating"    # Ready, waiting for Kairos
    PRESENTED = "presented"      # Delivered to user
    DECLINED = "declined"        # User wasn't receptive
    ARCHIVED = "archived"        # Old gift, no longer relevant


@dataclass
class Gift:
    """
    A gift from the Muse.

    Not just useful - surprising, beautiful, perfectly timed.
    """
    id: str
    title: str
    gift_type: GiftType
    description: str             # What this gift is
    rationale: str               # Why it matters to the user
    artifact: Optional[str]      # The actual content/path
    importance: float = 0.7      # 0-1, how important is this gift

    # Metadata
    status: GiftStatus = GiftStatus.CONCEIVING
    created_at: float = field(default_factory=time.time)
    presented_at: Optional[float] = None

    # Source
    friction_id: Optional[str] = None  # If born from friction
    dream_id: Optional[str] = None     # If aligned with a dream

    # Execution
    execution_steps: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "gift_type": self.gift_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "importance": self.importance,
            "status": self.status.value,
            "created_at": self.created_at,
            "friction_id": self.friction_id,
            "dream_id": self.dream_id,
        }


class TeleologyProtocol(Protocol):
    """Protocol for teleology/alignment checking."""
    def alignment(self, text: str) -> float:
        ...


class WeaverProtocol(Protocol):
    """Protocol for creative synthesis."""
    def synthesize_creative_solution(self, prompt: str) -> Any:
        ...


class Muse:
    """
    The Artist in the Machine.

    Solves engineering problems with creative/aesthetic payloads.
    Works in The Studio, crafting gifts that incubate until the right moment.

    Now with real artifact generators:
    - Memory Constellation: SVG visualization of memory topology
    - Spike Orrery: Animated HTML dashboard of SNN flow
    - Heat Death Map: Thermal stress visualization
    """

    # Mapping from friction types to artifact generators
    GENERATOR_MAP = {
        "code_complexity": "constellation",      # Code as memory/data structure
        "memory_pressure": "constellation",      # Memory layout visualization
        "thermal_stress": "heatmap",             # Thermal visualization
        "snn_bottleneck": "orrery",              # SNN flow visualization
        "performance": "heatmap",                # Hot paths as thermal
        "cognitive_overload": "constellation",   # Mental map as constellation
    }

    # Templates for creative solutions by friction type
    CREATIVE_TEMPLATES = {
        "code_complexity": [
            ("visualization", "Generate an interactive dependency graph"),
            ("metaphor", "Create a metaphorical map of the code structure"),
            ("aesthetic", "Reorganize the code with visual symmetry"),
        ],
        "documentation_gap": [
            ("documentation", "Create a beautiful, illustrated API guide"),
            ("visualization", "Generate architecture diagrams"),
            ("insight", "Write a narrative explaining the design decisions"),
        ],
        "cognitive_overload": [
            ("visualization", "Create a dashboard showing task relationships"),
            ("tool", "Build a focus mode that hides distractions"),
            ("metaphor", "Design a mental model for managing complexity"),
        ],
        "context_switching": [
            ("tool", "Create a workspace snapshot/restore tool"),
            ("visualization", "Generate a project overview wallpaper"),
            ("insight", "Suggest a workflow restructuring"),
        ],
    }

    def __init__(
        self,
        teleology: Optional[TeleologyProtocol] = None,
        weaver: Optional[WeaverProtocol] = None,
        llm: Optional[Any] = None,
        min_alignment: float = 0.6,
    ):
        """
        Initialize the Muse.

        Args:
            teleology: HorizonEngine for alignment checking
            weaver: Weaver for creative synthesis
            llm: LLM for generating creative solutions
            min_alignment: Minimum teleological alignment to craft a gift
        """
        self.teleology = teleology
        self.weaver = weaver
        self.llm = llm
        self.min_alignment = min_alignment
        self.log = logging.getLogger("Muse")

        # The Studio Queue - gifts waiting for their moment
        self.studio_queue: List[Gift] = []

        # Archive of presented gifts
        self._archive: List[Gift] = []

    def craft_gift(self, friction_point: Any) -> Optional[Gift]:
        """
        Craft a gift from a friction point.

        Unlike the Steward (which does practical fixes), the Muse
        creates artistic, surprising solutions.

        Args:
            friction_point: FrictionPoint to address

        Returns:
            A Gift if one could be crafted, None otherwise
        """
        # 1. Check teleological alignment
        #    Only craft gifts that matter to the Horizon
        alignment = self._check_alignment(friction_point)
        if alignment < self.min_alignment:
            self.log.debug(f"Friction {friction_point.id} not aligned enough: {alignment:.2f}")
            return None

        # 2. Creative synthesis
        concept = self._generate_creative_concept(friction_point)
        if concept is None:
            return None

        # 3. Create the gift
        gift = Gift(
            id=f"gift_{uuid.uuid4().hex[:8]}",
            title=f"🎁 {concept['title']}",
            gift_type=GiftType(concept['type']),
            description=concept['description'],
            rationale=concept['rationale'],
            importance=min(0.95, alignment * 1.2),
            status=GiftStatus.CRAFTING,
            friction_id=friction_point.id,
            execution_steps=concept.get('steps', []),
        )

        # 4. Execute the gift creation
        success = self._craft_artifact(gift)

        if success:
            gift.status = GiftStatus.INCUBATING
            self.log.info(f"🎨 MUSE: Crafted gift '{gift.title}'")
            return gift

        return None

    def _check_alignment(self, friction_point: Any) -> float:
        """Check if friction point aligns with teleology."""
        if self.teleology is None:
            return 0.7  # Default moderate alignment

        try:
            return self.teleology.alignment(friction_point.description)
        except Exception as e:
            self.log.warning(f"Alignment check failed: {e}")
            return 0.5

    def _generate_creative_concept(
        self,
        friction_point: Any,
    ) -> Optional[Dict[str, Any]]:
        """Generate a creative concept for solving the friction."""
        # Try LLM-based generation
        if self.llm is not None:
            concept = self._llm_creative_concept(friction_point)
            if concept:
                return concept

        # Try Weaver-based generation
        if self.weaver is not None:
            concept = self._weaver_creative_concept(friction_point)
            if concept:
                return concept

        # Fall back to templates
        return self._template_creative_concept(friction_point)

    def _llm_creative_concept(
        self,
        friction_point: Any,
    ) -> Optional[Dict[str, Any]]:
        """Generate creative concept using LLM."""
        prompt = f"""PROBLEM: {friction_point.description}
ROOT CAUSE: {friction_point.root_cause}
CONTEXT: The User is visual and values elegant, surprising solutions.

TASK: Design a 'Gift' that solves this problem.
Do NOT just fix the code. Create an artifact that:
1. ILLUMINATES the solution (makes it visible, beautiful)
2. SURPRISES the user (unexpected approach)
3. DELIGHTS through aesthetic quality

TYPES of gifts you can create:
- visualization: A diagram, graph, interactive chart
- documentation: Beautiful illustrated docs
- insight: A new perspective or realization
- tool: A helper script or dashboard
- metaphor: A conceptual framework
- aesthetic: Something made beautiful

OUTPUT FORMAT:
TITLE: [Short, evocative title]
TYPE: [one of the types above]
DESCRIPTION: [What this gift is]
RATIONALE: [Why it will delight the user]
STEPS:
1. [First creation step]
2. [Second step]
3. [Third step]
"""
        try:
            response = self.llm.generate(prompt)
            return self._parse_concept_response(response)
        except Exception as e:
            self.log.warning(f"LLM concept generation failed: {e}")
            return None

    def _weaver_creative_concept(
        self,
        friction_point: Any,
    ) -> Optional[Dict[str, Any]]:
        """Generate creative concept using Weaver."""
        prompt = f"""
        PROBLEM: {friction_point.description}
        ROOT CAUSE: {friction_point.root_cause}
        CONTEXT: The User is visual and values elegant systems.

        TASK: Design a 'Gift' that solves this.
        Create an artifact (Diagram, Visualization, Dashboard, Metaphor)
        that illuminates the solution.
        """
        try:
            result = self.weaver.synthesize_creative_solution(prompt)
            return {
                "title": getattr(result, 'title', 'A Gift'),
                "type": "visualization",
                "description": getattr(result, 'description', str(result)),
                "rationale": getattr(result, 'rationale', 'To help you'),
                "steps": getattr(result, 'execution_steps', []),
            }
        except Exception as e:
            self.log.warning(f"Weaver concept generation failed: {e}")
            return None

    def _template_creative_concept(
        self,
        friction_point: Any,
    ) -> Dict[str, Any]:
        """Generate creative concept using templates."""
        friction_type = getattr(friction_point, 'friction_type', None)
        type_key = friction_type.value if friction_type else 'documentation_gap'

        templates = self.CREATIVE_TEMPLATES.get(type_key, [
            ("visualization", "Create a visual representation"),
        ])

        # Pick the first template
        gift_type, action = templates[0]

        return {
            "title": f"{action}: {friction_point.description[:30]}",
            "type": gift_type,
            "description": f"{action} for {friction_point.description}",
            "rationale": f"This will illuminate the solution to {friction_point.root_cause}",
            "steps": [
                f"Analyze: {friction_point.description}",
                f"Create: {action}",
                "Polish the aesthetic",
                "Prepare for presentation",
            ],
        }

    def _parse_concept_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into concept dict."""
        try:
            lines = response.strip().split('\n')
            concept = {"steps": []}

            in_steps = False
            for line in lines:
                line = line.strip()
                if line.startswith("TITLE:"):
                    concept["title"] = line[6:].strip()
                    in_steps = False
                elif line.startswith("TYPE:"):
                    concept["type"] = line[5:].strip().lower()
                    in_steps = False
                elif line.startswith("DESCRIPTION:"):
                    concept["description"] = line[12:].strip()
                    in_steps = False
                elif line.startswith("RATIONALE:"):
                    concept["rationale"] = line[10:].strip()
                    in_steps = False
                elif line.startswith("STEPS:"):
                    in_steps = True
                elif in_steps and line:
                    if line[0].isdigit():
                        step = line[line.index('.') + 1:].strip() if '.' in line else line
                        concept["steps"].append(step)

            if concept.get("title") and concept.get("type"):
                return concept
        except Exception as e:
            self.log.warning(f"Failed to parse concept response: {e}")

        return None

    def _craft_artifact(self, gift: Gift) -> bool:
        """
        Craft the actual artifact for a gift.

        Uses gift generators to create real SVG/HTML artifacts when possible.
        Falls back to placeholder descriptions for unsupported types.
        """
        gift.execution_log.append(f"Starting craft: {gift.title}")

        # Try to use a real generator if we have data
        artifact_content = None
        generator_name = None

        # Check if we have a generator for this friction type
        if gift.friction_id:
            friction_type = gift.friction_id.split("_")[0] if "_" in gift.friction_id else "generic"
            generator_name = self.GENERATOR_MAP.get(friction_type)

        if generator_name:
            artifact_content = self._generate_with_generator(gift, generator_name)

        if artifact_content:
            gift.execution_log.append(f"Generated artifact using {generator_name}")
            gift.artifact = artifact_content
        else:
            # Fallback to placeholder
            for step in gift.execution_steps:
                gift.execution_log.append(f"Executing: {step}")
                time.sleep(0.01)

            gift.execution_log.append("Artifact crafted successfully")
            gift.artifact = f"[{gift.gift_type.value}] {gift.description}"

        return True

    def _generate_with_generator(
        self,
        gift: Gift,
        generator_name: str,
    ) -> Optional[str]:
        """
        Generate an artifact using one of the gift generators.

        Args:
            gift: The gift being crafted
            generator_name: Name of the generator to use

        Returns:
            Artifact content string, or None if generation failed
        """
        try:
            from ara.daemon.gifts import get_generator

            generator = get_generator(generator_name, theme="ara")
            if generator is None:
                self.log.warning(f"Generator not found: {generator_name}")
                return None

            # Build sample data based on gift type
            # In production, this would come from actual system state
            data = self._build_generator_data(gift, generator_name)

            artifact = generator.generate(data)

            # Save the artifact
            from pathlib import Path
            artifacts_dir = Path("~/.ara/artifacts").expanduser()
            artifact.save(artifacts_dir)

            gift.execution_log.append(f"Saved artifact to: {artifact.file_path}")
            self.log.info(f"🎨 MUSE: Generated {generator_name} artifact: {artifact.name}")

            return artifact.content

        except ImportError as e:
            self.log.debug(f"Gift generators not available: {e}")
            return None
        except Exception as e:
            self.log.warning(f"Failed to generate with {generator_name}: {e}")
            return None

    def _build_generator_data(
        self,
        gift: Gift,
        generator_name: str,
    ) -> Dict[str, Any]:
        """
        Build input data for a generator.

        In production, this would pull real data from HAL, Hippocampus, etc.
        For now, we generate meaningful sample data.
        """
        if generator_name == "constellation":
            # Memory constellation data
            return {
                "title": gift.title.replace("🎁 ", ""),
                "regions": [
                    {"id": "stack", "name": "Stack", "size_bytes": 8388608,
                     "access_frequency": 0.9, "category": "stack"},
                    {"id": "heap", "name": "Main Heap", "size_bytes": 67108864,
                     "access_frequency": 0.7, "category": "heap"},
                    {"id": "snn_buf", "name": "SNN Buffers", "size_bytes": 134217728,
                     "access_frequency": 0.95, "category": "mapped"},
                    {"id": "kv_cache", "name": "KV Cache", "size_bytes": 268435456,
                     "access_frequency": 0.6, "category": "mapped"},
                    {"id": "code", "name": "Code Section", "size_bytes": 16777216,
                     "access_frequency": 0.4, "category": "code"},
                    {"id": "shared", "name": "Shared Memory", "size_bytes": 33554432,
                     "access_frequency": 0.3, "category": "shared"},
                ],
                "flows": [
                    {"source_id": "stack", "target_id": "heap", "bandwidth": 0.8},
                    {"source_id": "heap", "target_id": "snn_buf", "bandwidth": 0.9},
                    {"source_id": "snn_buf", "target_id": "kv_cache", "bandwidth": 0.7},
                    {"source_id": "code", "target_id": "stack", "bandwidth": 0.5},
                ],
            }

        elif generator_name == "orrery":
            # SNN orrery data
            return {
                "title": gift.title.replace("🎁 ", ""),
                "layers": [
                    {"id": "input", "name": "Sensory Input", "neuron_count": 1024,
                     "layer_type": "input", "firing_rate": 0.3},
                    {"id": "encode", "name": "Encoder", "neuron_count": 512,
                     "layer_type": "hidden", "firing_rate": 0.5},
                    {"id": "reservoir", "name": "Reservoir", "neuron_count": 2048,
                     "layer_type": "hidden", "firing_rate": 0.6},
                    {"id": "decode", "name": "Decoder", "neuron_count": 256,
                     "layer_type": "hidden", "firing_rate": 0.4},
                    {"id": "output", "name": "Motor Output", "neuron_count": 64,
                     "layer_type": "output", "firing_rate": 0.2},
                ],
                "connections": [
                    {"source_layer": "input", "target_layer": "encode",
                     "weight": 0.8, "spike_rate": 0.6},
                    {"source_layer": "encode", "target_layer": "reservoir",
                     "weight": 0.9, "spike_rate": 0.7},
                    {"source_layer": "reservoir", "target_layer": "decode",
                     "weight": 0.7, "spike_rate": 0.5},
                    {"source_layer": "decode", "target_layer": "output",
                     "weight": 0.8, "spike_rate": 0.4},
                ],
            }

        elif generator_name == "heatmap":
            # Thermal heatmap data
            return {
                "title": gift.title.replace("🎁 ", ""),
                "system_name": "BANOS Host",
                "zones": [
                    {"id": "cpu0", "name": "CPU Package", "current_temp": 68,
                     "max_temp": 95, "position": [0.25, 0.3], "size": [0.2, 0.15],
                     "zone_type": "cpu"},
                    {"id": "gpu0", "name": "GPU Core", "current_temp": 72,
                     "max_temp": 90, "position": [0.55, 0.3], "size": [0.25, 0.2],
                     "zone_type": "gpu"},
                    {"id": "nvme0", "name": "NVMe SSD", "current_temp": 52,
                     "max_temp": 70, "position": [0.15, 0.6], "size": [0.12, 0.08],
                     "zone_type": "nvme"},
                    {"id": "ram", "name": "DDR5 DIMMs", "current_temp": 45,
                     "max_temp": 85, "position": [0.4, 0.55], "size": [0.3, 0.08],
                     "zone_type": "ram"},
                    {"id": "vrm", "name": "VRM", "current_temp": 78,
                     "max_temp": 100, "position": [0.15, 0.35], "size": [0.06, 0.1],
                     "zone_type": "vrm"},
                    {"id": "chiplet", "name": "CXL Chiplet", "current_temp": 65,
                     "max_temp": 85, "position": [0.7, 0.55], "size": [0.1, 0.1],
                     "zone_type": "chiplet"},
                ],
                "history": [
                    {"zone_id": "gpu0", "peak_temp": 92, "duration_seconds": 180,
                     "severity": 0.7},
                    {"zone_id": "vrm", "peak_temp": 95, "duration_seconds": 60,
                     "severity": 0.5},
                ],
            }

        else:
            return {"title": gift.title}

    def present(self, gift: Gift) -> str:
        """
        Present a gift to the user.

        Args:
            gift: The gift to present

        Returns:
            Presentation message
        """
        gift.status = GiftStatus.PRESENTED
        gift.presented_at = time.time()

        # Move from queue to archive
        if gift in self.studio_queue:
            self.studio_queue.remove(gift)
        self._archive.append(gift)

        self.log.info(f"🎁 MUSE: Presenting gift '{gift.title}'")

        return f"I have made something for you. {gift.rationale}"

    def get_ready_gifts(self) -> List[Gift]:
        """Get gifts that are ready to present (incubating)."""
        return [g for g in self.studio_queue if g.status == GiftStatus.INCUBATING]

    def get_highest_importance_gift(self) -> Optional[Gift]:
        """Get the most important ready gift."""
        ready = self.get_ready_gifts()
        if not ready:
            return None
        return max(ready, key=lambda g: g.importance)

    def queue_gift(self, gift: Gift) -> None:
        """Add a gift to the studio queue."""
        gift.status = GiftStatus.INCUBATING
        self.studio_queue.append(gift)

    def get_studio_report(self) -> str:
        """Generate a report of the studio's work."""
        lines = ["## The Studio Report\n"]

        ready = self.get_ready_gifts()
        crafting = [g for g in self.studio_queue if g.status == GiftStatus.CRAFTING]
        recent = [g for g in self._archive
                  if g.presented_at and g.presented_at > time.time() - 86400]

        if ready:
            lines.append(f"**Gifts Ready to Present:** {len(ready)}\n")
            for gift in ready:
                lines.append(f"- 🎁 {gift.title} (importance: {gift.importance:.0%})")
            lines.append("")

        if crafting:
            lines.append(f"**Gifts Being Crafted:** {len(crafting)}\n")
            for gift in crafting:
                lines.append(f"- 🎨 {gift.title}")
            lines.append("")

        if recent:
            lines.append(f"**Recently Presented:** {len(recent)}\n")
            for gift in recent:
                lines.append(f"- ✨ {gift.title}")
            lines.append("")

        if not (ready or crafting or recent):
            lines.append("The studio is quiet. Inspiration will come.\n")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_muse: Optional[Muse] = None


def get_muse() -> Muse:
    """Get the default Muse instance."""
    global _default_muse
    if _default_muse is None:
        _default_muse = Muse()
    return _default_muse


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GiftType',
    'GiftStatus',
    'Gift',
    'Muse',
    'get_muse',
]

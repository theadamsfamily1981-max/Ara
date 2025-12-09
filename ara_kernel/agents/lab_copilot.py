"""
Lab Copilot Agent
==================

Hardware and lab assistant for physical projects.

Handles:
- 3D printing workflows
- Laser cutting coordination
- Hardware debugging
- Lab inventory tracking
- Project documentation
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ara_kernel.core.runtime import AraAgentRuntime

logger = logging.getLogger(__name__)


class LabTaskType(Enum):
    """Types of lab tasks."""
    PRINT_3D = "3d_print"
    LASER_CUT = "laser_cut"
    DEBUG_HARDWARE = "debug_hardware"
    INVENTORY = "inventory"
    DOCUMENT = "document"
    ANALYZE_IMAGE = "analyze_image"


@dataclass
class LabTask:
    """A lab/hardware task."""
    task_id: str
    task_type: LabTaskType
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    created_ts: float = field(default_factory=time.time)


class LabCopilotAgent:
    """
    Lab and hardware copilot.

    Assists with:
    - 3D print preparation and troubleshooting
    - Laser cutter workflows
    - Hardware debugging
    - Project documentation
    - Photo/image analysis of physical components
    """

    def __init__(
        self,
        kernel: AraAgentRuntime,
        domain: str = "lab",
    ) -> None:
        self.kernel = kernel
        self.domain = domain
        self._tasks_processed = 0

    async def process_task(self, task: LabTask) -> Dict[str, Any]:
        """Process a single lab task."""
        self._tasks_processed += 1
        logger.info(f"Processing lab task {task.task_id}: {task.task_type.value}")

        prompt = self._build_prompt(task)

        try:
            result = await self.kernel.process_input(
                user_input=prompt,
                mode="private",
                metadata={
                    "domain": self.domain,
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                },
            )
            logger.info(f"Lab task {task.task_id} completed")
            return result

        except Exception as e:
            logger.exception(f"Lab task {task.task_id} failed: {e}")
            return {"error": str(e)}

    def process_task_sync(self, task: LabTask) -> Dict[str, Any]:
        """Synchronous wrapper for process_task."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_task(task))
        finally:
            loop.close()

    def _build_prompt(self, task: LabTask) -> str:
        """Build prompt based on task type."""
        builders = {
            LabTaskType.PRINT_3D: self._print_3d_prompt,
            LabTaskType.LASER_CUT: self._laser_cut_prompt,
            LabTaskType.DEBUG_HARDWARE: self._debug_prompt,
            LabTaskType.INVENTORY: self._inventory_prompt,
            LabTaskType.DOCUMENT: self._document_prompt,
            LabTaskType.ANALYZE_IMAGE: self._analyze_image_prompt,
        }

        builder = builders.get(task.task_type, self._generic_prompt)
        return builder(task)

    def _print_3d_prompt(self, task: LabTask) -> str:
        model = task.payload.get("model", "")
        material = task.payload.get("material", "PLA")
        printer = task.payload.get("printer", "")
        issue = task.payload.get("issue", "")

        prompt = f"""[LAB - 3D PRINTING]

Task: {task.description}

Model: {model or "Not specified"}
Material: {material}
Printer: {printer or "Not specified"}
{"Issue: " + issue if issue else ""}

Please provide:
1. Recommended print settings (layer height, infill, supports)
2. Potential issues to watch for
3. Post-processing recommendations
4. Estimated print time (if calculable)
"""
        return prompt

    def _laser_cut_prompt(self, task: LabTask) -> str:
        material = task.payload.get("material", "")
        thickness = task.payload.get("thickness", "")
        design = task.payload.get("design", "")

        prompt = f"""[LAB - LASER CUTTING]

Task: {task.description}

Material: {material or "Not specified"}
Thickness: {thickness or "Not specified"}
Design: {design or "Not specified"}

Please provide:
1. Recommended power/speed settings
2. Safety considerations
3. Order of operations (cuts, scores, engravings)
4. Material-specific tips
"""
        return prompt

    def _debug_prompt(self, task: LabTask) -> str:
        component = task.payload.get("component", "")
        symptom = task.payload.get("symptom", "")
        tried = task.payload.get("already_tried", [])

        prompt = f"""[LAB - HARDWARE DEBUGGING]

Task: {task.description}

Component: {component or "Not specified"}
Symptom: {symptom or "Not specified"}
Already tried: {', '.join(tried) if tried else "Nothing yet"}

Please provide:
1. Diagnostic steps to isolate the issue
2. Most likely causes
3. Suggested fixes in order of likelihood
4. Safety warnings if applicable
"""
        return prompt

    def _inventory_prompt(self, task: LabTask) -> str:
        action = task.payload.get("action", "check")
        items = task.payload.get("items", [])

        prompt = f"""[LAB - INVENTORY]

Task: {task.description}

Action: {action}
Items: {', '.join(items) if items else "General inventory check"}

Please help with:
1. Current status/availability
2. Reorder recommendations
3. Storage/organization suggestions
4. Alternatives if items are unavailable
"""
        return prompt

    def _document_prompt(self, task: LabTask) -> str:
        project = task.payload.get("project", "")
        doc_type = task.payload.get("doc_type", "notes")

        prompt = f"""[LAB - DOCUMENTATION]

Task: {task.description}

Project: {project or "Not specified"}
Document type: {doc_type}

Please help create documentation including:
1. Project overview
2. Key specifications
3. Process/procedure notes
4. Lessons learned
"""
        return prompt

    def _analyze_image_prompt(self, task: LabTask) -> str:
        image_path = task.payload.get("image_path", "")
        context = task.payload.get("context", "")

        prompt = f"""[LAB - IMAGE ANALYSIS]

Task: {task.description}

Image: {image_path or "Not provided"}
Context: {context or "General analysis"}

Please analyze and provide:
1. What you observe
2. Potential issues or concerns
3. Recommendations
4. Follow-up questions if needed
"""
        return prompt

    def _generic_prompt(self, task: LabTask) -> str:
        return f"""[LAB - {task.task_type.value.upper()}]

Task: {task.description}

Details: {task.payload}

Please provide guidance and assistance with this lab task.
"""


# Convenience functions
def create_print_task(
    description: str,
    model: str = "",
    material: str = "PLA",
    issue: str = "",
) -> LabTask:
    """Create a 3D printing task."""
    return LabTask(
        task_id=f"print_{int(time.time())}",
        task_type=LabTaskType.PRINT_3D,
        description=description,
        payload={
            "model": model,
            "material": material,
            "issue": issue,
        },
    )


def create_debug_task(
    description: str,
    component: str = "",
    symptom: str = "",
    already_tried: Optional[List[str]] = None,
) -> LabTask:
    """Create a hardware debugging task."""
    return LabTask(
        task_id=f"debug_{int(time.time())}",
        task_type=LabTaskType.DEBUG_HARDWARE,
        description=description,
        payload={
            "component": component,
            "symptom": symptom,
            "already_tried": already_tried or [],
        },
    )

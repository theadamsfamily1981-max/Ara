"""
Publishing CEO Agent
=====================

The "CEO mode" agent for content creation and publishing.

Handles:
- Content ideation and drafting
- KDP/publishing pipeline
- Merch design coordination
- Content quality review
"""

from __future__ import annotations

import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ara_kernel.core.runtime import AraAgentRuntime

logger = logging.getLogger(__name__)


@dataclass
class PublishingTask:
    """A task in the publishing pipeline."""
    task_id: str
    task_type: str  # "ideate", "draft", "review", "publish"
    content_type: str  # "ebook", "blog", "merch", "music"
    payload: Dict[str, Any]
    priority: float = 0.5
    created_ts: float = 0

    def __post_init__(self):
        if self.created_ts == 0:
            self.created_ts = time.time()


class PublishingCEOAgent:
    """
    Publishing-focused agent wrapper.

    Processes publishing tasks through the kernel:
    - Ideation: Generate content ideas
    - Drafting: Create content outlines and drafts
    - Review: Quality check against brand guidelines
    - Publishing: Final approval workflow
    """

    def __init__(
        self,
        kernel: AraAgentRuntime,
        domain: str = "publishing",
    ) -> None:
        self.kernel = kernel
        self.domain = domain

        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._tasks_processed = 0

    def submit_task(self, task: PublishingTask) -> None:
        """Submit a task to the publishing queue."""
        # Priority queue uses (priority, item) - lower is higher priority
        self._task_queue.put((1.0 - task.priority, task))
        logger.info(f"Publishing task submitted: {task.task_id} ({task.task_type})")

    def start(self) -> None:
        """Start processing publishing tasks."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("PublishingCEOAgent already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("PublishingCEOAgent started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop processing."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(f"PublishingCEOAgent stopped ({self._tasks_processed} tasks processed)")

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._thread is not None and self._thread.is_alive()

    def queue_size(self) -> int:
        """Number of pending tasks."""
        return self._task_queue.qsize()

    def _loop(self) -> None:
        """Main processing loop."""
        while not self._stop.is_set():
            try:
                # Non-blocking get with timeout
                _, task = self._task_queue.get(timeout=1.0)
                self._process_task(task)
                self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error processing publishing task: {e}")

    def _process_task(self, task: PublishingTask) -> None:
        """Process a single publishing task."""
        self._tasks_processed += 1
        logger.info(f"Processing task {task.task_id}: {task.task_type} ({task.content_type})")

        # Build prompt based on task type
        prompt = self._build_prompt(task)

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.kernel.process_input(
                        user_input=prompt,
                        mode="private",
                        metadata={
                            "domain": self.domain,
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "content_type": task.content_type,
                        },
                    )
                )
                logger.info(f"Task {task.task_id} completed")
                logger.debug(f"Result: {result.get('text', '')[:200]}")
            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Task {task.task_id} failed: {e}")

    def _build_prompt(self, task: PublishingTask) -> str:
        """Build a prompt for the task."""
        prompts = {
            "ideate": self._ideation_prompt,
            "draft": self._drafting_prompt,
            "review": self._review_prompt,
            "publish": self._publish_prompt,
        }

        builder = prompts.get(task.task_type, self._generic_prompt)
        return builder(task)

    def _ideation_prompt(self, task: PublishingTask) -> str:
        topic = task.payload.get("topic", "")
        constraints = task.payload.get("constraints", [])

        prompt = f"""[PUBLISHING IDEATION - {task.content_type.upper()}]

Generate content ideas for: {topic}

Content type: {task.content_type}
Constraints: {', '.join(constraints) if constraints else 'None specified'}

Please provide:
1. 3-5 specific content ideas
2. Target audience for each
3. Estimated effort level
4. Unique angle or hook

Focus on ideas that align with our brand: curious, technically rigorous, and authentic.
"""
        return prompt

    def _drafting_prompt(self, task: PublishingTask) -> str:
        idea = task.payload.get("idea", "")
        outline = task.payload.get("outline", "")
        target_length = task.payload.get("target_length", "1000-2000")

        outline_section = f"Outline:\n{outline}" if outline else "No outline provided - start from scratch."

        prompt = f"""[PUBLISHING DRAFT - {task.content_type.upper()}]

Create a draft for: {idea}

{outline_section}

Requirements:
- Match our brand voice (geeky, articulate, emotionally grounded)
- Include practical, actionable content
- Be authentic - no corporate speak
- Target length: {target_length} words

Please draft the content.
"""
        return prompt

    def _review_prompt(self, task: PublishingTask) -> str:
        content = task.payload.get("content", "")

        prompt = f"""[PUBLISHING REVIEW - {task.content_type.upper()}]

Review this content for publication:

---
{content[:2000]}...
---

Check against:
1. Brand voice alignment
2. Technical accuracy
3. Engagement and clarity
4. Any red flags (controversial claims, missing attributions, etc.)

Provide:
- Overall rating (1-10)
- Specific issues found
- Suggested improvements
- Publication recommendation (approve/revise/reject)
"""
        return prompt

    def _publish_prompt(self, task: PublishingTask) -> str:
        content = task.payload.get("content", "")
        platform = task.payload.get("platform", "unknown")

        prompt = f"""[PUBLISHING FINAL - {platform.upper()}]

Prepare content for publication on: {platform}

Content preview:
---
{content[:1000]}...
---

Please:
1. Format for {platform} requirements
2. Generate metadata (title, description, tags)
3. Create any required assets list
4. Confirm publication checklist
"""
        return prompt

    def _generic_prompt(self, task: PublishingTask) -> str:
        return f"""[PUBLISHING TASK - {task.task_type.upper()}]

Task: {task.task_type}
Content type: {task.content_type}
Payload: {task.payload}

Please process this publishing task.
"""


# Convenience functions for common tasks
def create_ideation_task(
    topic: str,
    content_type: str = "blog",
    constraints: Optional[List[str]] = None,
    priority: float = 0.5,
) -> PublishingTask:
    """Create an ideation task."""
    return PublishingTask(
        task_id=f"ideate_{int(time.time())}",
        task_type="ideate",
        content_type=content_type,
        payload={
            "topic": topic,
            "constraints": constraints or [],
        },
        priority=priority,
    )


def create_draft_task(
    idea: str,
    content_type: str = "blog",
    outline: str = "",
    target_length: str = "1000-2000",
    priority: float = 0.5,
) -> PublishingTask:
    """Create a drafting task."""
    return PublishingTask(
        task_id=f"draft_{int(time.time())}",
        task_type="draft",
        content_type=content_type,
        payload={
            "idea": idea,
            "outline": outline,
            "target_length": target_length,
        },
        priority=priority,
    )

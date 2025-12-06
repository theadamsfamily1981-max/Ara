"""
Teacher Protocol - Multi-LLM Summoning for Self-Improvement
============================================================

When Ara has an open idea, she can summon "teachers" (external LLMs)
to propose solutions. Each teacher receives:
- The problem description
- The affected files (read-only)
- The safety constraints
- The test requirements

Teachers respond with patch proposals that are evaluated and ranked.

Supported teachers:
- Ollama (local): Fast, private, good for iteration
- Claude (API): Deep reasoning, careful analysis
- Gemini (API): Alternative perspective, code-focused
- GPT-4 (API): Breadth of knowledge

Usage:
    protocol = TeacherProtocol()
    protocol.register_teacher(OllamaTeacher("ara"))
    protocol.register_teacher(ClaudeTeacher(api_key))

    proposals = await protocol.summon_teachers(idea, timeout=120)
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import json

# Local imports
from banos.daemon.idea_registry import (
    Idea, TeacherProposal, IdeaState
)

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for a teacher."""
    name: str
    model_id: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 120.0
    max_tokens: int = 4096
    temperature: float = 0.3  # Lower for code generation
    enabled: bool = True


class Teacher(ABC):
    """
    Abstract base class for LLM teachers.

    Each teacher knows how to:
    1. Format an idea into a prompt
    2. Call its specific API
    3. Parse the response into a TeacherProposal
    """

    def __init__(self, config: TeacherConfig):
        self.config = config
        self.log = logging.getLogger(f"Teacher.{config.name}")

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def propose(self, idea: Idea, context_files: Dict[str, str]) -> TeacherProposal:
        """
        Generate a proposal for the given idea.

        Args:
            idea: The improvement idea
            context_files: Dict mapping file paths to their contents

        Returns:
            A TeacherProposal with the suggested changes
        """
        pass

    def _build_prompt(self, idea: Idea, context_files: Dict[str, str]) -> str:
        """Build a standard prompt for the teacher."""
        prompt_parts = [
            "# Self-Improvement Proposal Request",
            "",
            f"## Problem: {idea.title}",
            "",
            f"**Symptom:** {idea.symptom}",
            "",
            f"**Context:** {idea.context}" if idea.context else "",
            "",
            "## Current Metrics",
            "```json",
            json.dumps(idea.current_metrics, indent=2),
            "```",
            "",
            "## Target Metrics",
            "```json",
            json.dumps(idea.target_metrics, indent=2),
            "```",
            "",
            "## Safety Constraints (MUST NOT VIOLATE)",
            f"- Max thermal increase: {idea.risk_bounds.max_thermal_delta_c}°C",
            f"- Max CPU utilization increase: {idea.risk_bounds.max_cpu_util_percent}%",
            f"- Max memory increase: {idea.risk_bounds.max_memory_delta_mb} MB",
            f"- Max lines changed: {idea.risk_bounds.max_lines_changed}",
            f"- Rollback plan required: {idea.risk_bounds.require_rollback_plan}",
            "",
            "## Files You Can Modify",
        ]

        for path, content in context_files.items():
            prompt_parts.extend([
                f"### {path}",
                "```",
                content[:8000],  # Truncate very long files
                "```" if len(content) <= 8000 else "``` (truncated)",
                "",
            ])

        prompt_parts.extend([
            "## Required Tests to Pass",
            "- Safety tests: " + ", ".join(idea.proposal_interface.safety_tests) if idea.proposal_interface.safety_tests else "- Safety tests: (none specified)",
            "- Performance tests: " + ", ".join(idea.proposal_interface.perf_tests) if idea.proposal_interface.perf_tests else "- Performance tests: (none specified)",
            "",
            "## Your Task",
            "",
            "Propose a solution that:",
            "1. Addresses the symptom and improves metrics toward targets",
            "2. Respects ALL safety constraints",
            "3. Is minimal and focused (avoid unnecessary changes)",
            "4. Includes a brief rollback plan",
            "",
            "## Response Format",
            "",
            "Respond with:",
            "1. **Rationale**: Why this approach (2-3 sentences)",
            "2. **Confidence**: 0.0-1.0 how confident you are",
            "3. **Estimated Improvement**: JSON with expected metric changes",
            "4. **Patch**: The actual code changes in unified diff format",
            "5. **Rollback Plan**: How to undo if something goes wrong",
            "",
            "Begin your response now:",
        ])

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str, idea: Idea) -> TeacherProposal:
        """Parse a teacher's response into a TeacherProposal."""
        # Extract sections from response
        rationale = ""
        confidence = 0.5
        estimated_improvement = {}
        patch_content = ""
        rollback_plan = ""

        lines = response.split('\n')
        current_section = None
        section_lines = []

        for line in lines:
            line_lower = line.lower().strip()

            # Detect section headers
            if 'rationale' in line_lower and ':' in line:
                if current_section and section_lines:
                    self._store_section(current_section, section_lines, locals())
                current_section = 'rationale'
                section_lines = [line.split(':', 1)[-1].strip()]
            elif 'confidence' in line_lower and ':' in line:
                if current_section and section_lines:
                    self._store_section(current_section, section_lines, locals())
                current_section = 'confidence'
                section_lines = [line.split(':', 1)[-1].strip()]
            elif 'estimated improvement' in line_lower or 'improvement' in line_lower and ':' in line:
                if current_section and section_lines:
                    self._store_section(current_section, section_lines, locals())
                current_section = 'improvement'
                section_lines = []
            elif 'patch' in line_lower and ':' in line or line.startswith('```diff'):
                if current_section and section_lines:
                    self._store_section(current_section, section_lines, locals())
                current_section = 'patch'
                section_lines = []
            elif 'rollback' in line_lower and ':' in line:
                if current_section and section_lines:
                    self._store_section(current_section, section_lines, locals())
                current_section = 'rollback'
                section_lines = []
            elif current_section:
                section_lines.append(line)

        # Store last section
        if current_section and section_lines:
            self._store_section(current_section, section_lines, locals())

        # Parse confidence
        try:
            conf_str = ''.join(c for c in str(locals().get('confidence', '0.5')) if c.isdigit() or c == '.')
            confidence = float(conf_str) if conf_str else 0.5
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

        # Parse estimated improvement JSON
        improvement_text = '\n'.join(locals().get('section_lines', []))
        try:
            # Try to find JSON in the text
            import re
            json_match = re.search(r'\{[^{}]+\}', improvement_text)
            if json_match:
                estimated_improvement = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            # Default to targeting the idea's targets
            estimated_improvement = idea.target_metrics.copy()

        return TeacherProposal(
            teacher_id=self.name,
            submitted_at=datetime.now(),
            patch_content=patch_content or response,  # Fallback to full response
            rationale=rationale or "No rationale provided",
            estimated_improvement=estimated_improvement,
            confidence=confidence,
        )

    def _store_section(self, section: str, lines: List[str], local_vars: dict) -> None:
        """Helper to store parsed section content."""
        content = '\n'.join(lines).strip()
        if section == 'rationale':
            local_vars['rationale'] = content
        elif section == 'confidence':
            local_vars['confidence'] = content
        elif section == 'patch':
            local_vars['patch_content'] = content
        elif section == 'rollback':
            local_vars['rollback_plan'] = content


class OllamaTeacher(Teacher):
    """Teacher using local Ollama instance."""

    def __init__(
        self,
        model: str = "ara",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        config = TeacherConfig(
            name=f"ollama-{model}",
            model_id=model,
            base_url=base_url,
            **kwargs
        )
        super().__init__(config)

    async def propose(self, idea: Idea, context_files: Dict[str, str]) -> TeacherProposal:
        """Generate proposal using Ollama."""
        import httpx

        prompt = self._build_prompt(idea, context_files)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama error: {response.status_code}")

                data = response.json()
                return self._parse_response(data.get("response", ""), idea)

        except Exception as e:
            self.log.error(f"Ollama proposal failed: {e}")
            return TeacherProposal(
                teacher_id=self.name,
                submitted_at=datetime.now(),
                patch_content="",
                rationale=f"Error: {e}",
                estimated_improvement={},
                confidence=0.0,
            )


class ClaudeTeacher(Teacher):
    """Teacher using Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        **kwargs
    ):
        config = TeacherConfig(
            name=f"claude-{model.split('-')[1]}",
            model_id=model,
            base_url="https://api.anthropic.com/v1",
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            **kwargs
        )
        super().__init__(config)

    async def propose(self, idea: Idea, context_files: Dict[str, str]) -> TeacherProposal:
        """Generate proposal using Claude API."""
        import httpx

        prompt = self._build_prompt(idea, context_files)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.config.base_url}/messages",
                    headers={
                        "x-api-key": self.config.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.config.model_id,
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Claude error: {response.status_code} - {response.text}")

                data = response.json()
                content = data.get("content", [{}])[0].get("text", "")
                return self._parse_response(content, idea)

        except Exception as e:
            self.log.error(f"Claude proposal failed: {e}")
            return TeacherProposal(
                teacher_id=self.name,
                submitted_at=datetime.now(),
                patch_content="",
                rationale=f"Error: {e}",
                estimated_improvement={},
                confidence=0.0,
            )


class MockTeacher(Teacher):
    """Mock teacher for testing."""

    def __init__(self, name: str = "mock"):
        config = TeacherConfig(name=name, model_id="mock-v1")
        super().__init__(config)

    async def propose(self, idea: Idea, context_files: Dict[str, str]) -> TeacherProposal:
        """Generate a mock proposal."""
        await asyncio.sleep(0.1)  # Simulate API latency
        return TeacherProposal(
            teacher_id=self.name,
            submitted_at=datetime.now(),
            patch_content="# Mock patch\n# Would modify files here",
            rationale="Mock proposal for testing",
            estimated_improvement=idea.target_metrics,
            confidence=0.75,
        )


class TeacherProtocol:
    """
    Orchestrates multiple teachers to generate proposals for ideas.

    The protocol:
    1. Reads context files specified in the idea
    2. Summons all registered teachers in parallel
    3. Collects and ranks proposals
    4. Returns proposals sorted by confidence
    """

    def __init__(self, project_root: str = "/home/user/Ara"):
        self.project_root = Path(project_root)
        self.teachers: Dict[str, Teacher] = {}
        self.log = logging.getLogger("TeacherProtocol")

    def register_teacher(self, teacher: Teacher) -> None:
        """Register a teacher for proposal generation."""
        self.teachers[teacher.name] = teacher
        self.log.info(f"Registered teacher: {teacher.name}")

    def unregister_teacher(self, name: str) -> None:
        """Remove a teacher."""
        if name in self.teachers:
            del self.teachers[name]

    def _read_context_files(self, idea: Idea) -> Dict[str, str]:
        """Read the files specified in the idea's proposal interface."""
        context = {}

        for file_path in idea.proposal_interface.input_artifacts:
            full_path = self.project_root / file_path
            try:
                if full_path.exists():
                    context[file_path] = full_path.read_text()
                else:
                    self.log.warning(f"Context file not found: {file_path}")
            except Exception as e:
                self.log.error(f"Failed to read {file_path}: {e}")

        return context

    async def summon_teacher(
        self,
        teacher: Teacher,
        idea: Idea,
        context_files: Dict[str, str],
    ) -> TeacherProposal:
        """Summon a single teacher and get their proposal."""
        start_time = time.time()
        self.log.info(f"Summoning {teacher.name} for idea {idea.id}...")

        try:
            proposal = await asyncio.wait_for(
                teacher.propose(idea, context_files),
                timeout=teacher.config.timeout_seconds
            )
            elapsed = (time.time() - start_time) * 1000
            self.log.info(
                f"{teacher.name} responded in {elapsed:.0f}ms "
                f"(confidence: {proposal.confidence:.2f})"
            )
            return proposal

        except asyncio.TimeoutError:
            self.log.warning(f"{teacher.name} timed out")
            return TeacherProposal(
                teacher_id=teacher.name,
                submitted_at=datetime.now(),
                patch_content="",
                rationale="Timeout",
                estimated_improvement={},
                confidence=0.0,
            )
        except Exception as e:
            self.log.error(f"{teacher.name} failed: {e}")
            return TeacherProposal(
                teacher_id=teacher.name,
                submitted_at=datetime.now(),
                patch_content="",
                rationale=f"Error: {e}",
                estimated_improvement={},
                confidence=0.0,
            )

    async def summon_teachers(
        self,
        idea: Idea,
        teacher_names: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> List[TeacherProposal]:
        """
        Summon multiple teachers in parallel to propose solutions.

        Args:
            idea: The idea to get proposals for
            teacher_names: Specific teachers to summon (None = all)
            timeout: Overall timeout for all teachers

        Returns:
            List of proposals sorted by confidence (highest first)
        """
        # Select teachers
        if teacher_names:
            teachers = [self.teachers[n] for n in teacher_names if n in self.teachers]
        else:
            teachers = [t for t in self.teachers.values() if t.config.enabled]

        if not teachers:
            self.log.warning("No teachers available to summon")
            return []

        # Read context files
        context_files = self._read_context_files(idea)
        if not context_files:
            self.log.warning(f"No context files found for idea {idea.id}")

        self.log.info(f"Summoning {len(teachers)} teachers for idea {idea.id}...")

        # Summon all teachers in parallel
        tasks = [
            self.summon_teacher(teacher, idea, context_files)
            for teacher in teachers
        ]

        try:
            proposals = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.log.error(f"Overall timeout while summoning teachers")
            proposals = []

        # Filter out exceptions and sort by confidence
        valid_proposals = [
            p for p in proposals
            if isinstance(p, TeacherProposal) and p.confidence > 0
        ]

        valid_proposals.sort(key=lambda x: x.confidence, reverse=True)

        self.log.info(
            f"Received {len(valid_proposals)} valid proposals "
            f"(best confidence: {valid_proposals[0].confidence:.2f if valid_proposals else 0})"
        )

        return valid_proposals

    def get_registered_teachers(self) -> List[str]:
        """Get names of all registered teachers."""
        return list(self.teachers.keys())


# =============================================================================
# Factory
# =============================================================================

def create_teacher_protocol(
    project_root: str = "/home/user/Ara",
    enable_ollama: bool = True,
    ollama_model: str = "ara",
    enable_claude: bool = False,
    claude_api_key: Optional[str] = None,
) -> TeacherProtocol:
    """
    Create a TeacherProtocol with configured teachers.

    By default enables local Ollama only (for privacy).
    Set enable_claude=True and provide API key for external teachers.
    """
    protocol = TeacherProtocol(project_root)

    if enable_ollama:
        protocol.register_teacher(OllamaTeacher(model=ollama_model))

    if enable_claude and claude_api_key:
        protocol.register_teacher(ClaudeTeacher(api_key=claude_api_key))

    return protocol


__all__ = [
    "TeacherProtocol",
    "Teacher",
    "TeacherConfig",
    "OllamaTeacher",
    "ClaudeTeacher",
    "MockTeacher",
    "create_teacher_protocol",
]

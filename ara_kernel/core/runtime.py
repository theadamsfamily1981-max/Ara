"""
Ara Agent Runtime
==================

The main execution loop for Ara's agent kernel.

This is where everything comes together:
- Model inference (remote or local)
- Memory retrieval and storage
- Safety filtering
- Pheromone coordination
- Tool execution

The runtime is intentionally minimal - a thin coordination layer
that delegates to specialized subsystems.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from .config import KernelConfig, load_config
from .safety import SafetyCovenant, ActionPlan, FilteredPlan, ActionClass

logger = logging.getLogger(__name__)


class RuntimeState(Enum):
    """Agent runtime states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTING = "executing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    """Current runtime context."""
    conversation: List[Message] = field(default_factory=list)
    current_mode: str = "private"  # "private" or "public"
    active_tools: List[str] = field(default_factory=list)
    pending_approvals: List[Dict] = field(default_factory=list)
    session_id: str = ""
    user_id: str = ""


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0


class AraAgentRuntime:
    """
    The Ara Agent Runtime.

    This is the "tiny brain" that coordinates:
    1. Receiving input
    2. Enriching with memory
    3. Getting model response
    4. Filtering through safety
    5. Executing approved actions
    6. Storing experiences

    The runtime itself is stateless between sessions -
    all state lives in memory layers and pheromone store.
    """

    def __init__(
        self,
        config: KernelConfig,
        model_client: Optional[Any] = None,
        memory_core: Optional[Any] = None,
        pheromone_store: Optional[Any] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        self.config = config
        self.model_client = model_client
        self.memory_core = memory_core
        self.pheromone_store = pheromone_store
        self.tool_registry = tool_registry or {}

        # Safety covenant - always enforced
        self.safety = SafetyCovenant(
            allowed_domains=config.safety.allowed_domains,
            disallowed_domains=config.safety.disallowed_domains,
            disclosure_policy=config.safety.disclosure_policy,
            human_approval_actions=config.safety.human_approval_actions,
            max_autonomy_level=config.safety.max_autonomy_level,
        )

        self.state = RuntimeState.IDLE
        self.context = RuntimeContext()

        # Callbacks for external integration
        self._on_state_change: Optional[Callable] = None
        self._on_approval_needed: Optional[Callable] = None
        self._on_output: Optional[Callable] = None

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def process_input(
        self,
        user_input: str,
        mode: str = "private",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process user input through the full pipeline.

        Args:
            user_input: The user's message
            mode: "private" (full access) or "public" (filtered)
            metadata: Additional context

        Returns:
            Response dict with text, actions taken, and metadata
        """
        self._set_state(RuntimeState.PROCESSING)
        metadata = metadata or {}

        try:
            # 1. Add to conversation
            self.context.current_mode = mode
            self.context.conversation.append(Message(
                role="user",
                content=user_input,
                metadata=metadata,
            ))

            # 2. Enrich with memory (if available)
            enriched_context = await self._enrich_with_memory(user_input, mode)

            # 3. Check pheromones for current hive state
            hive_context = self._read_pheromones()

            # 4. Build prompt
            full_prompt = self._build_prompt(
                user_input,
                enriched_context,
                hive_context,
            )

            # 5. Get model response
            model_response = await self._call_model(full_prompt)

            # 6. Parse response for actions
            actions = self._parse_actions(model_response)

            # 7. Filter through safety
            if actions:
                plan = ActionPlan(actions=actions, context={"mode": mode})
                filtered = self.safety.filter(plan)

                # Handle blocked actions
                if filtered.blocked_actions:
                    logger.warning(
                        f"Blocked {len(filtered.blocked_actions)} actions: "
                        f"{[a.get('type') for a in filtered.blocked_actions]}"
                    )

                # Execute approved actions
                tool_results = []
                if filtered.allowed_actions:
                    self._set_state(RuntimeState.EXECUTING)
                    tool_results = await self._execute_actions(filtered.allowed_actions)
            else:
                filtered = FilteredPlan([], [], [], True)
                tool_results = []

            # 8. Check output safety
            output_safe, output_violations = self.safety.check_output(model_response)
            if not output_safe:
                model_response = self._sanitize_output(model_response, output_violations)

            # 9. Store experience (if memory available)
            await self._store_experience(
                user_input,
                model_response,
                tool_results,
                mode,
            )

            # 10. Build response
            response = {
                "text": model_response,
                "actions_executed": [r.tool_name for r in tool_results if r.success],
                "actions_blocked": [a.get("type") for a in filtered.blocked_actions],
                "mode": mode,
                "violations": [v.message for v in filtered.violations],
            }

            # Add to conversation
            self.context.conversation.append(Message(
                role="assistant",
                content=model_response,
                metadata={"actions": response["actions_executed"]},
            ))

            self._set_state(RuntimeState.IDLE)
            return response

        except Exception as e:
            logger.error(f"Runtime error: {e}")
            self._set_state(RuntimeState.ERROR)
            return {
                "text": "I encountered an error processing that request.",
                "error": str(e),
                "mode": mode,
            }

    # =========================================================================
    # Pipeline Steps
    # =========================================================================

    async def _enrich_with_memory(
        self,
        user_input: str,
        mode: str,
    ) -> Dict[str, Any]:
        """Retrieve relevant memories."""
        if not self.memory_core:
            return {}

        try:
            # Memory core handles visibility filtering based on mode
            from ara_memory.core import ContextFlags
            flags = ContextFlags(
                mode=mode,
                include_soul=True,
                include_skills=True,
                include_world=True,
            )
            enriched = self.memory_core.enrich_prompt(user_input, flags)
            return {
                "soul": enriched.soul_context,
                "skills": enriched.skill_context,
                "world": enriched.world_context,
            }
        except ImportError:
            return {}
        except Exception as e:
            logger.warning(f"Memory enrichment failed: {e}")
            return {}

    def _read_pheromones(self) -> Dict[str, Any]:
        """Read current hive state from pheromones."""
        if not self.pheromone_store:
            return {}

        try:
            from ara.hive import PheromoneKind

            # Get current global mode
            globals = self.pheromone_store.get_strongest(PheromoneKind.GLOBAL, 1)
            current_mode = globals[0].key if globals else "NORMAL"

            # Get active priorities
            priorities = self.pheromone_store.get_strongest(PheromoneKind.PRIORITY, 5)

            # Get active alarms
            alarms = self.pheromone_store.get_by_kind(PheromoneKind.ALARM)

            return {
                "global_mode": current_mode,
                "priorities": [p.key for p in priorities],
                "alarms": [a.key for a in alarms],
                "in_safe_mode": current_mode == "SAFE_MODE" or len(alarms) > 0,
            }
        except ImportError:
            return {}
        except Exception as e:
            logger.warning(f"Pheromone read failed: {e}")
            return {}

    def _build_prompt(
        self,
        user_input: str,
        memory_context: Dict,
        hive_context: Dict,
    ) -> str:
        """Build the full prompt for the model."""
        parts = []

        # System prompt (persona)
        parts.append(self._get_system_prompt())

        # Memory context
        if memory_context.get("soul"):
            parts.append(f"\n[Relevant memories]\n{memory_context['soul']}")

        if memory_context.get("skills"):
            parts.append(f"\n[Active skills]\n{memory_context['skills']}")

        if memory_context.get("world"):
            parts.append(f"\n[World knowledge]\n{memory_context['world']}")

        # Hive context
        if hive_context:
            if hive_context.get("in_safe_mode"):
                parts.append("\n[SAFE MODE ACTIVE - Avoid risky operations]")
            if hive_context.get("priorities"):
                parts.append(f"\n[Current priorities: {', '.join(hive_context['priorities'])}]")

        # Conversation history (recent)
        recent_msgs = self.context.conversation[-10:]
        if recent_msgs:
            history = "\n".join([
                f"{m.role}: {m.content[:500]}"
                for m in recent_msgs[:-1]  # Exclude the current message
            ])
            if history:
                parts.append(f"\n[Recent conversation]\n{history}")

        # Current input
        parts.append(f"\nUser: {user_input}")

        return "\n".join(parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on persona config."""
        persona = self.config.persona

        # Base prompt
        prompt = f"""You are {persona.name}, an AI assistant.

Voice: {persona.voice}

Core principles:
- Be helpful, harmless, and honest
- Acknowledge AI nature when appropriate
- Never pretend to be human
- Respect safety boundaries

Available tools: {list(self.tool_registry.keys())}

To use a tool, respond with:
[TOOL: tool_name]
{{"param": "value"}}
[/TOOL]
"""
        return prompt

    async def _call_model(self, prompt: str) -> str:
        """Call the model for inference."""
        if not self.model_client:
            # Fallback for testing
            return "I understand your request. [No model client configured]"

        try:
            # Generic model call - adapter pattern
            response = await self.model_client.complete(
                prompt=prompt,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
            )
            return response
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return f"I encountered an error: {e}"

    def _parse_actions(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model response."""
        import re
        import json

        actions = []
        pattern = r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]'
        matches = re.findall(pattern, response, re.DOTALL)

        for tool_name, params_str in matches:
            try:
                params = json.loads(params_str.strip()) if params_str.strip() else {}
                actions.append({
                    "type": tool_name,
                    "data": params,
                })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool params: {params_str}")

        return actions

    async def _execute_actions(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """Execute approved actions."""
        results = []

        for action in actions:
            tool_name = action.get("type")
            tool_data = action.get("data", {})

            if tool_name not in self.tool_registry:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error=f"Tool '{tool_name}' not found",
                ))
                continue

            tool_fn = self.tool_registry[tool_name]

            try:
                start = time.time()

                # Execute with timeout
                if asyncio.iscoroutinefunction(tool_fn):
                    output = await asyncio.wait_for(
                        tool_fn(**tool_data),
                        timeout=self.config.resources.tool_timeout_seconds,
                    )
                else:
                    output = tool_fn(**tool_data)

                duration = (time.time() - start) * 1000

                results.append(ToolResult(
                    tool_name=tool_name,
                    success=True,
                    output=output,
                    duration_ms=duration,
                ))
                logger.info(f"Tool '{tool_name}' executed in {duration:.0f}ms")

            except asyncio.TimeoutError:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error="Tool execution timed out",
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error=str(e),
                ))
                logger.error(f"Tool '{tool_name}' failed: {e}")

        return results

    def _sanitize_output(
        self,
        text: str,
        violations: List,
    ) -> str:
        """Sanitize output that failed safety checks."""
        # For now, just add a warning
        return text + "\n\n[Some content was filtered for safety]"

    async def _store_experience(
        self,
        user_input: str,
        response: str,
        tool_results: List[ToolResult],
        mode: str,
    ):
        """Store the experience in memory."""
        if not self.memory_core:
            return

        try:
            # Create episode
            episode = {
                "user_input": user_input,
                "response": response[:1000],  # Truncate for storage
                "tools_used": [r.tool_name for r in tool_results if r.success],
                "mode": mode,
                "timestamp": time.time(),
            }
            # Memory core will handle storage
            # await self.memory_core.soul.store_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to store experience: {e}")

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, new_state: RuntimeState):
        """Update runtime state and notify listeners."""
        old_state = self.state
        self.state = new_state

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

        logger.debug(f"Runtime state: {old_state.value} -> {new_state.value}")

    def on_state_change(self, callback: Callable):
        """Register state change callback."""
        self._on_state_change = callback

    def on_approval_needed(self, callback: Callable):
        """Register approval needed callback."""
        self._on_approval_needed = callback

    def on_output(self, callback: Callable):
        """Register output callback."""
        self._on_output = callback

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def register_tool(self, name: str, fn: Callable, action_class: str = "A"):
        """Register a tool for the agent to use."""
        self.tool_registry[name] = fn
        logger.info(f"Registered tool: {name} (class {action_class})")

    def register_tools(self, tools: Dict[str, Callable]):
        """Register multiple tools."""
        for name, fn in tools.items():
            self.register_tool(name, fn)

    # =========================================================================
    # Session Management
    # =========================================================================

    def new_session(self, session_id: str = "", user_id: str = ""):
        """Start a new conversation session."""
        self.context = RuntimeContext(
            session_id=session_id or f"session_{int(time.time())}",
            user_id=user_id,
        )
        self.state = RuntimeState.IDLE
        logger.info(f"New session: {self.context.session_id}")

    def get_conversation(self) -> List[Dict]:
        """Get current conversation as list of dicts."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in self.context.conversation
        ]

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def from_config_file(cls, config_path: str) -> AraAgentRuntime:
        """Create runtime from config file."""
        config = load_config(config_path)
        return cls(config)

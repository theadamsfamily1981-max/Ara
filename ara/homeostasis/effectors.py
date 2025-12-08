"""
Ara Effector Daemon - Motor Output to All Subsystems
====================================================

The EffectorDaemon is the organism's motor system - translating
sovereign commands into physical action across all subsystems.

Architecture:
                        [Sovereign Loop]
                               │
                               ▼
                     ┌─────────────────┐
                     │ EffectorDaemon  │
                     │   (500 Hz)      │
                     └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐        ┌──────────┐         ┌──────────┐
    │ Reflex  │        │  Spinal  │         │ Cortical │
    │ Layer   │        │  Layer   │         │  Layer   │
    │ (<1 µs) │        │ (~100 µs)│         │ (~1 ms)  │
    └────┬────┘        └────┬─────┘         └────┬─────┘
         │                  │                    │
         ▼                  ▼                    ▼
    [TCAM Rules]       [NodeAgents]         [Visual/UI]
    [Hash CAM]         [LAN Control]        [Cathedral]

Effector Layers:
1. Reflex (< 1 µs): Direct TCAM/CAM programming
2. Spinal (~100 µs): NodeAgent commands, flow control
3. Cortical (~1 ms): Visual output, cathedral writes
4. Visual (~10 ms): Compositor updates, user feedback
"""

from __future__ import annotations

import asyncio
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from queue import Queue, Empty
from enum import IntEnum
import logging

from .state import OperationalMode, ErrorVector, HomeostaticState
from .config import HomeostaticConfig, ModeConfig, MODES


logger = logging.getLogger(__name__)


# =============================================================================
# Effector Command Types
# =============================================================================

class CommandType(IntEnum):
    """Types of effector commands."""
    REFLEX_RULE = 0     # TCAM rule update
    FLOW_ENTRY = 1      # Hash CAM entry
    NODE_COMMAND = 2    # NodeAgent command
    VISUAL_UPDATE = 3   # Visual compositor
    CATHEDRAL_WRITE = 4 # Memory consolidation
    POWER_CONTROL = 5   # Power/thermal management
    MODE_CHANGE = 6     # Mode propagation


@dataclass
class EffectorCommand:
    """A command to an effector."""
    cmd_type: CommandType
    target: str = ""        # Target subsystem
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0       # Higher = more urgent
    timestamp: float = 0.0
    deadline_ms: float = 0.0  # 0 = no deadline


# =============================================================================
# Effector Layers
# =============================================================================

class ReflexLayer:
    """
    Reflex layer - sub-microsecond responses.

    Programs TCAM rules and Hash CAM entries for
    autonomous packet handling.
    """

    def __init__(self):
        self._reflex_search = None  # ReflexSearch instance
        self._command_count = 0

    def connect(self, reflex_search) -> None:
        """Connect to reflex search module."""
        self._reflex_search = reflex_search

    def execute(self, cmd: EffectorCommand) -> bool:
        """Execute a reflex command."""
        if self._reflex_search is None:
            return False

        try:
            if cmd.cmd_type == CommandType.REFLEX_RULE:
                # Program TCAM rule
                from ..core.lan.reflex_search import (
                    FlowSignature, ReflexAction
                )
                key = FlowSignature(**cmd.data.get('key', {}))
                mask = FlowSignature(**cmd.data.get('mask', {}))
                action = ReflexAction(cmd.data.get('action', 0))
                priority = cmd.data.get('priority', 0)

                self._reflex_search.add_rule(key, mask, action, priority)
                self._command_count += 1
                return True

            elif cmd.cmd_type == CommandType.FLOW_ENTRY:
                # Program Hash CAM entry
                from ..core.lan.reflex_search import FlowSignature
                sig = FlowSignature(**cmd.data.get('sig', {}))
                metadata = cmd.data.get('metadata', {})

                self._reflex_search.register_flow(sig, metadata)
                self._command_count += 1
                return True

        except Exception as e:
            logger.debug(f"Reflex command error: {e}")
            return False

        return False


class SpinalLayer:
    """
    Spinal layer - microsecond responses.

    Controls NodeAgents, flow priorities, and LAN behavior.
    """

    def __init__(self):
        self._node_agents: Dict[str, Any] = {}
        self._command_count = 0

    def register_node(self, node_id: str, agent) -> None:
        """Register a node agent."""
        self._node_agents[node_id] = agent

    def execute(self, cmd: EffectorCommand) -> bool:
        """Execute a spinal command."""
        try:
            if cmd.cmd_type == CommandType.NODE_COMMAND:
                target = cmd.target
                if target in self._node_agents:
                    agent = self._node_agents[target]
                    action = cmd.data.get('action', 'noop')

                    if action == 'throttle':
                        agent.set_rate_limit(cmd.data.get('rate', 1000))
                    elif action == 'priority':
                        agent.set_priority(cmd.data.get('level', 0))
                    elif action == 'isolate':
                        agent.isolate()
                    elif action == 'restore':
                        agent.restore()

                    self._command_count += 1
                    return True

            elif cmd.cmd_type == CommandType.POWER_CONTROL:
                # Power management commands
                action = cmd.data.get('action', 'noop')

                if action == 'throttle_fpga':
                    # Reduce FPGA clock or disable units
                    logger.info("FPGA throttling requested")
                elif action == 'enable_cooling':
                    # Activate active cooling
                    logger.info("Active cooling enabled")

                self._command_count += 1
                return True

        except Exception as e:
            logger.debug(f"Spinal command error: {e}")
            return False

        return False


class CorticalLayer:
    """
    Cortical layer - millisecond responses.

    Handles visual output, cathedral writes, and higher-level coordination.
    """

    def __init__(self):
        self._visual_compositor = None
        self._cathedral = None
        self._command_count = 0

    def connect(self, visual_compositor=None, cathedral=None) -> None:
        """Connect to cortical systems."""
        self._visual_compositor = visual_compositor
        self._cathedral = cathedral

    def execute(self, cmd: EffectorCommand) -> bool:
        """Execute a cortical command."""
        try:
            if cmd.cmd_type == CommandType.VISUAL_UPDATE:
                if self._visual_compositor:
                    update_type = cmd.data.get('type', 'status')

                    if update_type == 'status':
                        # Update status display
                        self._visual_compositor.update_status(cmd.data)
                    elif update_type == 'alert':
                        # Show alert
                        self._visual_compositor.show_alert(
                            cmd.data.get('message', ''),
                            cmd.data.get('level', 'info')
                        )
                    elif update_type == 'glitch':
                        # Trigger visual glitch (pain response)
                        self._visual_compositor.trigger_glitch(
                            cmd.data.get('intensity', 0.5)
                        )

                    self._command_count += 1
                    return True

            elif cmd.cmd_type == CommandType.CATHEDRAL_WRITE:
                if self._cathedral:
                    write_type = cmd.data.get('type', 'episode')

                    if write_type == 'episode':
                        # Write episode to cathedral
                        self._cathedral.write_episode(
                            cmd.data.get('h_moment'),
                            cmd.data.get('context', {})
                        )
                    elif write_type == 'attractor':
                        # Create new attractor
                        self._cathedral.create_attractor(
                            cmd.data.get('hv'),
                            cmd.data.get('label', '')
                        )

                    self._command_count += 1
                    return True

        except Exception as e:
            logger.debug(f"Cortical command error: {e}")
            return False

        return False


# =============================================================================
# Mode Effects
# =============================================================================

class ModeEffects:
    """
    Applies mode-specific effects across all layers.

    Each mode has different behaviors:
    - REST: Low activity, cathedral active
    - IDLE: Monitoring, ready to respond
    - ACTIVE: Normal operation
    - FLOW: Peak performance
    - EMERGENCY: Maximum responsiveness
    """

    def __init__(self, config: HomeostaticConfig):
        self.config = config

    def apply_mode(
        self,
        mode: OperationalMode,
        reflex: ReflexLayer,
        spinal: SpinalLayer,
        cortical: CorticalLayer,
    ) -> List[EffectorCommand]:
        """
        Generate commands to apply mode effects.

        Args:
            mode: Target mode
            reflex: Reflex layer
            spinal: Spinal layer
            cortical: Cortical layer

        Returns:
            List of commands to execute
        """
        commands = []
        mode_config = MODES.get(mode.name, MODES['IDLE'])

        if mode == OperationalMode.REST:
            # Minimize activity, maximize consolidation
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'low_power'},
            ))
            commands.append(EffectorCommand(
                cmd_type=CommandType.VISUAL_UPDATE,
                data={'type': 'status', 'mode': 'rest'},
            ))

        elif mode == OperationalMode.IDLE:
            # Light monitoring
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'normal_power'},
            ))

        elif mode == OperationalMode.ACTIVE:
            # Full operation
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'normal_power'},
            ))
            commands.append(EffectorCommand(
                cmd_type=CommandType.VISUAL_UPDATE,
                data={'type': 'status', 'mode': 'active'},
            ))

        elif mode == OperationalMode.FLOW:
            # Peak performance
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'high_performance'},
            ))
            commands.append(EffectorCommand(
                cmd_type=CommandType.VISUAL_UPDATE,
                data={'type': 'status', 'mode': 'flow'},
            ))

        elif mode == OperationalMode.EMERGENCY:
            # Emergency response
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'emergency'},
                priority=100,
            ))
            commands.append(EffectorCommand(
                cmd_type=CommandType.VISUAL_UPDATE,
                data={'type': 'alert', 'message': 'Emergency mode', 'level': 'critical'},
                priority=100,
            ))

        return commands


# =============================================================================
# Error Response
# =============================================================================

class ErrorResponse:
    """
    Generates effector commands in response to errors.

    Maps error types to corrective actions.
    """

    def __init__(self, config: HomeostaticConfig):
        self.config = config

    def respond_to_error(self, error: ErrorVector) -> List[EffectorCommand]:
        """
        Generate commands to respond to errors.

        Args:
            error: Current error vector

        Returns:
            List of corrective commands
        """
        commands = []

        # Thermal response
        if error.e_thermal > 0.5:
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'throttle_fpga'},
                priority=int(error.e_thermal * 100),
            ))
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'enable_cooling'},
                priority=int(error.e_thermal * 100),
            ))

        if error.e_thermal_critical:
            commands.append(EffectorCommand(
                cmd_type=CommandType.VISUAL_UPDATE,
                data={'type': 'glitch', 'intensity': 0.8},
                priority=100,
            ))

        # Cognitive load response
        if error.e_cognitive > 0.7:
            # Reduce processing
            commands.append(EffectorCommand(
                cmd_type=CommandType.POWER_CONTROL,
                data={'action': 'reduce_parallelism'},
                priority=int(error.e_cognitive * 100),
            ))

        # Network response
        if error.e_packet_loss > 0.5:
            # Flag problematic flows
            commands.append(EffectorCommand(
                cmd_type=CommandType.REFLEX_RULE,
                data={
                    'key': {},
                    'mask': {},
                    'action': 7,  # MARK_SUSPICIOUS
                    'priority': 50,
                },
            ))

        return commands


# =============================================================================
# Effector Daemon
# =============================================================================

class EffectorDaemon:
    """
    The effector daemon - translates sovereign commands into action.

    Runs at 500 Hz (2 ms period), executing commands from the
    sovereign loop across all effector layers.
    """

    def __init__(
        self,
        config: HomeostaticConfig,
        input_queue: Queue,
        target_hz: float = 500.0,
    ):
        """
        Initialize effector daemon.

        Args:
            config: Homeostatic configuration
            input_queue: Queue for command input (from sovereign)
            target_hz: Target loop rate
        """
        self.config = config
        self.input_queue = input_queue
        self.target_hz = target_hz
        self.period = 1.0 / target_hz

        # Layers
        self.reflex = ReflexLayer()
        self.spinal = SpinalLayer()
        self.cortical = CorticalLayer()

        # Effects generators
        self.mode_effects = ModeEffects(config)
        self.error_response = ErrorResponse(config)

        # State
        self._current_mode = OperationalMode.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Command queue (priority-sorted)
        self._pending_commands: List[EffectorCommand] = []

        # Statistics
        self._loop_count = 0
        self._commands_executed = 0
        self._reflex_commands = 0
        self._spinal_commands = 0
        self._cortical_commands = 0

    def connect_modules(
        self,
        reflex_search=None,
        visual_compositor=None,
        cathedral=None,
    ) -> None:
        """Connect effector layers to their targets."""
        if reflex_search:
            self.reflex.connect(reflex_search)
        if visual_compositor or cathedral:
            self.cortical.connect(visual_compositor, cathedral)

    def register_node(self, node_id: str, agent) -> None:
        """Register a node agent with the spinal layer."""
        self.spinal.register_node(node_id, agent)

    def start(self) -> None:
        """Start the effector daemon."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"EffectorDaemon started at {self.target_hz} Hz")

    def stop(self) -> None:
        """Stop the effector daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("EffectorDaemon stopped")

    def _run_loop(self) -> None:
        """Main effector loop."""
        next_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # 1. Receive sovereign commands
            self._receive_commands()

            # 2. Execute pending commands by priority
            self._execute_commands()

            # Statistics
            self._loop_count += 1

            # Timing control
            next_time += self.period
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def _receive_commands(self) -> None:
        """Receive commands from sovereign loop."""
        while True:
            try:
                cmd_data = self.input_queue.get_nowait()

                # Check for mode change
                new_mode = cmd_data.get('mode')
                if new_mode and new_mode != self._current_mode:
                    self._handle_mode_change(new_mode)
                    self._current_mode = new_mode

                # Generate error response commands
                error = cmd_data.get('error')
                if error:
                    error_cmds = self.error_response.respond_to_error(error)
                    self._pending_commands.extend(error_cmds)

            except Empty:
                break

    def _handle_mode_change(self, new_mode: OperationalMode) -> None:
        """Handle mode change by generating mode effect commands."""
        commands = self.mode_effects.apply_mode(
            new_mode,
            self.reflex,
            self.spinal,
            self.cortical,
        )
        self._pending_commands.extend(commands)

    def _execute_commands(self) -> None:
        """Execute pending commands by priority."""
        if not self._pending_commands:
            return

        # Sort by priority (higher first)
        self._pending_commands.sort(key=lambda c: c.priority, reverse=True)

        # Execute up to N commands per tick
        max_per_tick = 10
        executed = 0

        while self._pending_commands and executed < max_per_tick:
            cmd = self._pending_commands.pop(0)

            # Check deadline
            if cmd.deadline_ms > 0:
                age_ms = (time.time() - cmd.timestamp) * 1000
                if age_ms > cmd.deadline_ms:
                    continue  # Expired, skip

            # Route to appropriate layer
            success = False

            if cmd.cmd_type in (CommandType.REFLEX_RULE, CommandType.FLOW_ENTRY):
                success = self.reflex.execute(cmd)
                if success:
                    self._reflex_commands += 1

            elif cmd.cmd_type in (CommandType.NODE_COMMAND, CommandType.POWER_CONTROL):
                success = self.spinal.execute(cmd)
                if success:
                    self._spinal_commands += 1

            elif cmd.cmd_type in (CommandType.VISUAL_UPDATE, CommandType.CATHEDRAL_WRITE):
                success = self.cortical.execute(cmd)
                if success:
                    self._cortical_commands += 1

            if success:
                self._commands_executed += 1
            executed += 1

    def queue_command(self, cmd: EffectorCommand) -> None:
        """Manually queue a command for execution."""
        cmd.timestamp = time.time()
        self._pending_commands.append(cmd)

    def get_stats(self) -> Dict[str, Any]:
        """Get effector daemon statistics."""
        return {
            'loop_count': self._loop_count,
            'target_hz': self.target_hz,
            'commands_executed': self._commands_executed,
            'reflex_commands': self._reflex_commands,
            'spinal_commands': self._spinal_commands,
            'cortical_commands': self._cortical_commands,
            'pending_commands': len(self._pending_commands),
            'current_mode': self._current_mode.name,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CommandType',
    'EffectorCommand',
    'ReflexLayer',
    'SpinalLayer',
    'CorticalLayer',
    'ModeEffects',
    'ErrorResponse',
    'EffectorDaemon',
]

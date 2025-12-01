"""
Ara Avatar System API Service

Handles voice commands, system state, and bidirectional communication
with the Ara avatar system.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AraMode(str, Enum):
    """Ara personality modes."""
    WORK = "work"
    RELAX = "relax"


class AraService:
    """Service for Ara avatar system integration."""

    def __init__(self):
        """Initialize Ara service."""
        self.mode = AraMode.WORK
        self.avatar_config = {
            'profile': 'Default',
            'style': 'Realistic',
            'mood': 'Neutral'
        }
        self.command_history: List[Dict[str, Any]] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.connected_clients: List[Any] = []

        # System state
        self.system_state = {
            'workspace_mode': 'work',
            'current_view': 'dashboard',
            'training_active': False,
            'topology_visible': False,
            'fullscreen': False,
            'cockpit_active': False
        }

        # Voice command mappings
        self.command_mappings = self._build_command_mappings()

    def _build_command_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Build voice command → action mappings."""
        return {
            # View navigation
            'show dashboard': {
                'action': 'switch_view',
                'params': {'view': 'dashboard'},
                'response': "Opening dashboard"
            },
            'show pareto': {
                'action': 'switch_view',
                'params': {'view': 'pareto'},
                'response': "Opening Pareto optimization view"
            },
            'show training': {
                'action': 'switch_view',
                'params': {'view': 'training'},
                'response': "Opening training monitor"
            },
            'show topology': {
                'action': 'show_topology',
                'params': {'mode': 'landscape', 'fullscreen': False},
                'response': "Engaging topology visualization"
            },
            'show gpu': {
                'action': 'cockpit_view',
                'params': {'view': 'gpu'},
                'response': "Switching to GPU metrics"
            },
            'show cpu': {
                'action': 'cockpit_view',
                'params': {'view': 'cpu'},
                'response': "Switching to CPU/RAM metrics"
            },
            'show network': {
                'action': 'cockpit_view',
                'params': {'view': 'network'},
                'response': "Switching to network view"
            },
            'show storage': {
                'action': 'cockpit_view',
                'params': {'view': 'storage'},
                'response': "Switching to storage view"
            },
            'show overview': {
                'action': 'cockpit_view',
                'params': {'view': 'overview'},
                'response': "Switching to mission overview"
            },

            # Topology modes
            'topology barcode': {
                'action': 'set_topology_mode',
                'params': {'mode': 'barcode'},
                'response': "Switching to barcode nebula visualization"
            },
            'topology landscape': {
                'action': 'set_topology_mode',
                'params': {'mode': 'landscape'},
                'response': "Switching to landscape waterfall visualization"
            },
            'topology poincare': {
                'action': 'set_topology_mode',
                'params': {'mode': 'poincare'},
                'response': "Switching to Poincaré orbits visualization"
            },
            'topology pareto': {
                'action': 'set_topology_mode',
                'params': {'mode': 'pareto'},
                'response': "Switching to Pareto galaxy visualization"
            },
            'engage topology': {
                'action': 'show_topology',
                'params': {'fullscreen': True},
                'response': "Engaging full topology display"
            },
            'hide topology': {
                'action': 'hide_topology',
                'params': {},
                'response': "Hiding topology visualization"
            },

            # Workspace modes
            'work mode': {
                'action': 'set_workspace_mode',
                'params': {'mode': 'work'},
                'response': "Switching to work mode"
            },
            'relax mode': {
                'action': 'set_workspace_mode',
                'params': {'mode': 'relax'},
                'response': "Switching to relaxation mode"
            },

            # Training control
            'start training': {
                'action': 'start_training',
                'params': {'config': 'default'},
                'response': "Initiating training session"
            },
            'stop training': {
                'action': 'stop_training',
                'params': {},
                'response': "Stopping training session"
            },

            # Window control
            'fullscreen': {
                'action': 'toggle_fullscreen',
                'params': {},
                'response': "Toggling fullscreen"
            },
            'minimize': {
                'action': 'minimize_window',
                'params': {},
                'response': "Minimizing window"
            },
            'restore': {
                'action': 'restore_window',
                'params': {},
                'response': "Restoring window"
            },

            # Avatar control
            'avatar professional': {
                'action': 'set_avatar_profile',
                'params': {'profile': 'Professional'},
                'response': "Switching to professional avatar"
            },
            'avatar casual': {
                'action': 'set_avatar_profile',
                'params': {'profile': 'Casual'},
                'response': "Switching to casual avatar"
            },
            'avatar scientist': {
                'action': 'set_avatar_profile',
                'params': {'profile': 'Scientist'},
                'response': "Switching to scientist avatar"
            },

            # Status queries
            'status report': {
                'action': 'get_status_report',
                'params': {},
                'response': "Generating status report"
            },
            'metrics report': {
                'action': 'get_metrics_report',
                'params': {},
                'response': "Generating metrics report"
            },
        }

    def process_command(self, command_text: str) -> Dict[str, Any]:
        """
        Process a voice command from Ara.

        Args:
            command_text: Natural language command

        Returns:
            dict: Action to perform, params, and response text
        """
        command_lower = command_text.lower().strip()

        # Log command
        self.command_history.append({
            'timestamp': datetime.now().isoformat(),
            'command': command_text,
            'processed': True
        })

        # Try exact matches first
        for pattern, mapping in self.command_mappings.items():
            if pattern in command_lower:
                logger.info(f"[Ara] Matched command: {pattern} → {mapping['action']}")
                return {
                    'success': True,
                    'action': mapping['action'],
                    'params': mapping['params'],
                    'response': mapping['response'],
                    'original_command': command_text
                }

        # Try fuzzy matching for common patterns
        result = self._fuzzy_match_command(command_lower)
        if result:
            return result

        # Command not recognized
        return {
            'success': False,
            'action': 'unknown',
            'params': {},
            'response': f"I didn't understand: {command_text}",
            'original_command': command_text
        }

    def _fuzzy_match_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Fuzzy match common command patterns."""
        # Topology variations
        if 'topology' in command:
            if 'full' in command or 'screen' in command:
                return {
                    'success': True,
                    'action': 'show_topology',
                    'params': {'fullscreen': True},
                    'response': "Engaging full topology display"
                }
            return {
                'success': True,
                'action': 'show_topology',
                'params': {'fullscreen': False},
                'response': "Showing topology visualization"
            }

        # GPU/metrics variations
        if 'gpu' in command or 'graphics' in command:
            return {
                'success': True,
                'action': 'cockpit_view',
                'params': {'view': 'gpu'},
                'response': "Switching to GPU metrics"
            }

        # Status variations
        if 'status' in command or 'how are' in command or 'what' in command:
            return {
                'success': True,
                'action': 'get_status_report',
                'params': {},
                'response': "Generating status report"
            }

        return None

    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            **self.system_state,
            'mode': self.mode.value,
            'avatar': self.avatar_config,
            'timestamp': datetime.now().isoformat()
        }

    def update_system_state(self, key: str, value: Any):
        """Update a system state value."""
        self.system_state[key] = value
        logger.info(f"[Ara] System state updated: {key} = {value}")

    def set_workspace_mode(self, mode: str) -> bool:
        """Set workspace mode (work/relax)."""
        try:
            self.mode = AraMode(mode)
            self.system_state['workspace_mode'] = mode
            logger.info(f"[Ara] Workspace mode set to: {mode}")
            return True
        except ValueError:
            logger.error(f"[Ara] Invalid workspace mode: {mode}")
            return False

    def set_avatar_config(self, profile: str = None, style: str = None, mood: str = None) -> bool:
        """Update avatar configuration."""
        if profile:
            self.avatar_config['profile'] = profile
        if style:
            self.avatar_config['style'] = style
        if mood:
            self.avatar_config['mood'] = mood

        logger.info(f"[Ara] Avatar config updated: {self.avatar_config}")
        return True

    def get_command_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return self.command_history[-limit:]

    async def push_event(self, event_type: str, data: Dict[str, Any]):
        """Push an event to the event queue for WebSocket broadcast."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        await self.event_queue.put(event)

    def get_personality_config(self) -> Dict[str, Any]:
        """Get Ara's personality configuration based on current mode."""
        if self.mode == AraMode.WORK:
            return {
                'mode': 'professional',
                'speech_style': 'formal',
                'proactivity': 'high',
                'humor_level': 'low',
                'detail_level': 'technical'
            }
        else:
            return {
                'mode': 'conversational',
                'speech_style': 'casual',
                'proactivity': 'moderate',
                'humor_level': 'moderate',
                'detail_level': 'simplified'
            }

    def generate_status_report(self, metrics: Dict[str, Any] = None) -> str:
        """Generate a spoken status report for Ara."""
        state = self.get_system_state()

        if self.mode == AraMode.WORK:
            # Professional status report
            report = f"System status: "

            if state.get('training_active'):
                if metrics:
                    report += f"Training active at {metrics.get('accuracy', 0):.1%} accuracy. "
                else:
                    report += "Training in progress. "
            else:
                report += "Idle. "

            report += f"Currently in {state['workspace_mode']} mode, viewing {state['current_view']}."

        else:
            # Casual status report
            report = "Hey! "

            if state.get('training_active'):
                if metrics:
                    acc = metrics.get('accuracy', 0) * 100
                    report += f"Training's going well - we're at {acc:.0f}% accuracy. "
                else:
                    report += "Training's running smoothly. "
            else:
                report += "Everything's chill right now. "

            report += f"We're in {state['workspace_mode']} mode."

        return report

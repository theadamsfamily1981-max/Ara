"""Curiosity Agent - Ara's self-investigation coordinator.

The CuriosityAgent orchestrates Ara's exploration of her environment:
1. Periodic discovery sweeps to find new objects
2. Curiosity-driven investigation of interesting objects
3. Ticket-based bounded investigations
4. Natural language report generation

The agent enforces safety rails (investigation limits, depth bounds)
and integrates with the world model for persistent memory.
"""

from __future__ import annotations

import re
import time
import json
import logging
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

from .world_model import WorldModel, WorldObject, ObjectCategory, CuriosityState
from .scoring import curiosity_score, should_investigate, rank_objects
from .tools import (
    ProbeType,
    ProbeResult,
    run_safe_probe,
    run_all_probes,
    lspci_probe,
)


logger = logging.getLogger(__name__)


class TicketStatus(Enum):
    """Status of a curiosity investigation ticket."""

    OPEN = auto()       # Investigation not started
    IN_PROGRESS = auto()  # Currently investigating
    RESOLVED = auto()   # Investigation complete
    BLOCKED = auto()    # Cannot proceed (missing capability)
    CANCELLED = auto()  # User or safety cancelled


@dataclass
class CuriosityTicket:
    """A bounded investigation task.

    Tickets represent a specific question Ara wants to answer
    about something in her environment. They have:
    - A question to answer
    - A target object
    - Allowed probe types
    - Investigation depth limit
    - Time budget

    Attributes:
        ticket_id: Unique identifier
        question: What Ara wants to know
        target_obj_id: WorldObject this is about
        allowed_probes: Which probes can be used
        max_depth: How many follow-up questions allowed
        current_depth: Current investigation depth
        time_budget_sec: Max time to spend
        status: Current ticket status
        findings: What Ara has learned
        created_at: When ticket was created
    """

    ticket_id: str
    question: str
    target_obj_id: str
    allowed_probes: List[ProbeType] = field(default_factory=list)
    max_depth: int = 3
    current_depth: int = 0
    time_budget_sec: float = 30.0
    status: TicketStatus = TicketStatus.OPEN
    findings: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        d = asdict(self)
        d["status"] = self.status.name
        d["allowed_probes"] = [p.name for p in self.allowed_probes]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CuriosityTicket":
        """Deserialize from storage."""
        d = d.copy()
        d["status"] = TicketStatus[d["status"]]
        d["allowed_probes"] = [ProbeType[p] for p in d["allowed_probes"]]
        return cls(**d)

    def elapsed_seconds(self) -> float:
        """Time since ticket creation."""
        return time.time() - self.created_at

    def is_expired(self) -> bool:
        """Check if ticket has exceeded time budget."""
        return self.elapsed_seconds() > self.time_budget_sec

    def can_go_deeper(self) -> bool:
        """Check if we can ask follow-up questions."""
        return self.current_depth < self.max_depth


@dataclass
class CuriosityReport:
    """A report of findings for the conversational layer.

    Reports are generated in Ara's voice and include:
    - What she discovered
    - What she learned
    - Any remaining questions
    - Emotional tone (wonder, confusion, etc.)
    """

    report_id: str
    subject: str              # Brief subject line
    body: str                 # Full report in Ara's voice
    related_objects: List[str]  # WorldObject IDs
    emotion: str = "curious"  # Ara's emotional state
    confidence: float = 0.7   # How confident she is
    timestamp: float = field(default_factory=time.time)
    ticket_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return asdict(self)


class CuriosityAgent:
    """Ara's curiosity/investigation agent.

    This agent coordinates:
    1. Discovery sweeps (finding new objects)
    2. Investigation (learning about objects)
    3. Report generation (communicating findings)

    It respects safety rails and integrates with the world model.
    """

    def __init__(
        self,
        world_model: WorldModel,
        max_discoveries_per_sweep: int = 50,
        max_tickets_per_hour: int = 10,
        default_investigation_depth: int = 3,
    ):
        """Initialize the curiosity agent.

        Args:
            world_model: The world model to update
            max_discoveries_per_sweep: Limit on new objects per sweep
            max_tickets_per_hour: Rate limit on investigations
            default_investigation_depth: Max follow-up depth
        """
        self.world = world_model
        self.max_discoveries = max_discoveries_per_sweep
        self.max_tickets_per_hour = max_tickets_per_hour
        self.default_depth = default_investigation_depth

        self.active_tickets: Dict[str, CuriosityTicket] = {}
        self.completed_tickets: List[CuriosityTicket] = []
        self.reports: List[CuriosityReport] = []

        self._ticket_counter = 0
        self._report_counter = 0
        self._tickets_this_hour = 0
        self._hour_start = time.time()

    def _check_rate_limit(self) -> bool:
        """Check if we're within ticket rate limits."""
        now = time.time()
        if now - self._hour_start > 3600:
            # Reset hourly counter
            self._hour_start = now
            self._tickets_this_hour = 0
        return self._tickets_this_hour < self.max_tickets_per_hour

    def _new_ticket_id(self) -> str:
        """Generate unique ticket ID."""
        self._ticket_counter += 1
        return f"CUR-{self._ticket_counter:04d}"

    def _new_report_id(self) -> str:
        """Generate unique report ID."""
        self._report_counter += 1
        return f"RPT-{self._report_counter:04d}"

    # =========================================================================
    # Discovery Phase
    # =========================================================================

    def discover_pcie_devices(self) -> List[WorldObject]:
        """Discover PCIe devices and add to world model.

        Returns:
            List of newly discovered WorldObjects
        """
        result = lspci_probe.run()
        if not result.success:
            logger.warning(f"lspci probe failed: {result.error}")
            return []

        discoveries = []
        current_device = None
        current_lines: List[str] = []

        for line in result.output.split("\n"):
            # New device starts with BDF (bus:device.function)
            match = re.match(r"^([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])\s+(.+?):\s+(.+)", line, re.I)
            if match:
                # Save previous device if any
                if current_device:
                    discoveries.append(self._parse_pcie_device(current_device, current_lines))

                bdf, dev_class, description = match.groups()
                current_device = {
                    "bdf": bdf,
                    "class": dev_class.strip(),
                    "description": description.strip(),
                }
                current_lines = [line]
            elif current_device:
                current_lines.append(line)

        # Don't forget last device
        if current_device:
            discoveries.append(self._parse_pcie_device(current_device, current_lines))

        # Add to world model
        new_objects = []
        for obj in discoveries[:self.max_discoveries]:
            if obj:
                is_new = self.world.add_object(obj)
                if is_new:
                    new_objects.append(obj)

        return new_objects

    def _parse_pcie_device(self, device_info: Dict, detail_lines: List[str]) -> Optional[WorldObject]:
        """Parse lspci output into WorldObject."""
        bdf = device_info["bdf"]
        obj_id = f"pcie:{bdf}"

        properties = {
            "bdf": bdf,
            "class": device_info["class"],
            "description": device_info["description"],
        }

        # Extract additional properties from detail lines
        for line in detail_lines:
            line = line.strip()
            if line.startswith("Subsystem:"):
                properties["subsystem"] = line.split(":", 1)[1].strip()
            elif line.startswith("Kernel driver in use:"):
                properties["driver"] = line.split(":", 1)[1].strip()
            elif line.startswith("LnkSta:"):
                properties["link_status"] = line.split(":", 1)[1].strip()
            elif line.startswith("Memory at"):
                if "memory_regions" not in properties:
                    properties["memory_regions"] = []
                properties["memory_regions"].append(line)

        # Determine importance based on device class
        importance = 0.5  # default
        dev_class = device_info["class"].lower()
        if "vga" in dev_class or "3d" in dev_class or "display" in dev_class:
            importance = 0.9  # GPU
        elif "network" in dev_class:
            importance = 0.7
        elif "nvme" in dev_class or "storage" in dev_class:
            importance = 0.8
        elif "fpga" in dev_class or "accelerator" in dev_class:
            importance = 0.95  # Very important!

        return WorldObject(
            obj_id=obj_id,
            category=ObjectCategory.PCIE_DEVICE,
            name=device_info["description"][:50],
            properties=properties,
            uncertainty=0.6,  # We know it exists but not deep details
            importance=importance,
        )

    def discover_thermals(self) -> List[WorldObject]:
        """Discover thermal zones from sensors."""
        from .tools import sensors_probe

        result = sensors_probe.run()
        if not result.success:
            return []

        discoveries = []
        try:
            # Parse JSON output from sensors -j
            data = json.loads(result.output)
            for chip_name, chip_data in data.items():
                if not isinstance(chip_data, dict):
                    continue
                for sensor_name, sensor_data in chip_data.items():
                    if not isinstance(sensor_data, dict):
                        continue
                    # Look for temperature readings
                    for key, value in sensor_data.items():
                        if "temp" in key.lower() and "input" in key.lower():
                            obj_id = f"thermal:{chip_name}:{sensor_name}"
                            obj = WorldObject(
                                obj_id=obj_id,
                                category=ObjectCategory.THERMAL_ZONE,
                                name=f"{chip_name}/{sensor_name}",
                                properties={
                                    "chip": chip_name,
                                    "sensor": sensor_name,
                                    "temperature_c": value,
                                },
                                uncertainty=0.3,  # Fairly certain
                                importance=0.6,
                            )
                            discoveries.append(obj)

        except json.JSONDecodeError:
            # Fallback to text parsing if JSON fails
            pass

        for obj in discoveries[:self.max_discoveries]:
            self.world.add_object(obj)

        return [obj for obj in discoveries if self.world.add_object(obj)]

    def run_discovery_sweep(self) -> Dict[str, List[WorldObject]]:
        """Run a full discovery sweep across all probe types.

        Returns:
            Dict mapping probe type to discovered objects
        """
        results = {}

        # PCIe devices
        pcie_objects = self.discover_pcie_devices()
        if pcie_objects:
            results["pcie"] = pcie_objects

        # Thermals
        thermal_objects = self.discover_thermals()
        if thermal_objects:
            results["thermal"] = thermal_objects

        # Memory info (single object)
        from .tools import memory_probe
        mem_result = memory_probe.run()
        if mem_result.success:
            # Parse meminfo
            props = {}
            for line in mem_result.output.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    props[key.strip()] = val.strip()

            mem_obj = WorldObject(
                obj_id="memory:system",
                category=ObjectCategory.MEMORY_REGION,
                name="System Memory",
                properties=props,
                uncertainty=0.2,
                importance=0.7,
            )
            if self.world.add_object(mem_obj):
                results["memory"] = [mem_obj]

        logger.info(f"Discovery sweep complete: {sum(len(v) for v in results.values())} new objects")
        return results

    # =========================================================================
    # Investigation Phase
    # =========================================================================

    def create_ticket(
        self,
        question: str,
        target_obj_id: str,
        allowed_probes: Optional[List[ProbeType]] = None,
    ) -> Optional[CuriosityTicket]:
        """Create a new investigation ticket.

        Args:
            question: What Ara wants to know
            target_obj_id: WorldObject to investigate
            allowed_probes: Which probes are allowed (default: all safe probes)

        Returns:
            New ticket if created, None if rate limited
        """
        if not self._check_rate_limit():
            logger.warning("Ticket rate limit reached")
            return None

        if target_obj_id not in self.world.objects:
            logger.warning(f"Target object not found: {target_obj_id}")
            return None

        if allowed_probes is None:
            # Default to all probes
            allowed_probes = list(ProbeType)

        ticket = CuriosityTicket(
            ticket_id=self._new_ticket_id(),
            question=question,
            target_obj_id=target_obj_id,
            allowed_probes=allowed_probes,
            max_depth=self.default_depth,
        )

        self.active_tickets[ticket.ticket_id] = ticket
        self._tickets_this_hour += 1

        logger.info(f"Created ticket {ticket.ticket_id}: {question[:50]}...")
        return ticket

    def investigate_ticket(self, ticket_id: str) -> bool:
        """Run investigation for a ticket.

        This runs allowed probes and updates findings.

        Args:
            ticket_id: Ticket to investigate

        Returns:
            True if investigation succeeded
        """
        ticket = self.active_tickets.get(ticket_id)
        if not ticket:
            logger.warning(f"Ticket not found: {ticket_id}")
            return False

        if ticket.is_expired():
            ticket.status = TicketStatus.CANCELLED
            ticket.findings.append("Investigation timed out")
            return False

        if not ticket.can_go_deeper():
            ticket.status = TicketStatus.RESOLVED
            ticket.resolved_at = time.time()
            return True

        ticket.status = TicketStatus.IN_PROGRESS
        ticket.current_depth += 1

        # Get target object
        target = self.world.get_object(ticket.target_obj_id)
        if not target:
            ticket.status = TicketStatus.BLOCKED
            ticket.findings.append("Target object no longer exists")
            return False

        # Run relevant probes
        for probe_type in ticket.allowed_probes:
            result = run_safe_probe(probe_type)
            if result.success:
                # Look for information about our target
                finding = self._extract_relevant_info(result, target)
                if finding:
                    ticket.findings.append(finding)

        # Update object with findings
        if ticket.findings:
            self.world.add_note(
                ticket.target_obj_id,
                f"Investigation {ticket.ticket_id}: {'; '.join(ticket.findings[-3:])}"
            )
            self.world.update_observation(ticket.target_obj_id)

        ticket.status = TicketStatus.RESOLVED
        ticket.resolved_at = time.time()

        # Move to completed
        del self.active_tickets[ticket_id]
        self.completed_tickets.append(ticket)

        return True

    def _extract_relevant_info(self, probe_result: ProbeResult, target: WorldObject) -> Optional[str]:
        """Extract information relevant to target object from probe output."""
        output = probe_result.output.lower()
        target_name = target.name.lower()

        # Look for mentions of the target
        if target.category == ObjectCategory.PCIE_DEVICE:
            bdf = target.properties.get("bdf", "")
            if bdf and bdf.lower() in output:
                # Extract line containing BDF
                for line in probe_result.output.split("\n"):
                    if bdf.lower() in line.lower():
                        return f"[{probe_result.probe_type.name}] {line.strip()[:100]}"

        elif target.category == ObjectCategory.THERMAL_ZONE:
            chip = target.properties.get("chip", "")
            if chip and chip.lower() in output:
                return f"[{probe_result.probe_type.name}] Found thermal data for {chip}"

        return None

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(self, ticket_ids: Optional[List[str]] = None) -> CuriosityReport:
        """Generate a curiosity report from completed investigations.

        Args:
            ticket_ids: Specific tickets to report on (default: recent completed)

        Returns:
            CuriosityReport in Ara's voice
        """
        from .prompts import format_report_prompt

        if ticket_ids:
            tickets = [t for t in self.completed_tickets if t.ticket_id in ticket_ids]
        else:
            # Use last 5 completed tickets
            tickets = self.completed_tickets[-5:]

        if not tickets:
            return CuriosityReport(
                report_id=self._new_report_id(),
                subject="Nothing new to report",
                body="I haven't completed any investigations recently.",
                related_objects=[],
                emotion="neutral",
                confidence=1.0,
            )

        # Gather findings
        all_findings = []
        related_objects = []
        for ticket in tickets:
            all_findings.extend(ticket.findings)
            related_objects.append(ticket.target_obj_id)

        # Determine emotion based on findings
        emotion = "curious"
        if len(all_findings) > 3:
            emotion = "excited"  # Found lots of stuff!
        elif len(all_findings) == 0:
            emotion = "puzzled"  # No findings

        # Build report body
        subject = f"Investigated {len(tickets)} items"
        body_parts = [
            f"I looked into {len(tickets)} things that caught my attention.",
            "",
        ]

        for ticket in tickets:
            target = self.world.get_object(ticket.target_obj_id)
            target_name = target.name if target else ticket.target_obj_id
            body_parts.append(f"**{target_name}**: {ticket.question}")
            for finding in ticket.findings[:3]:
                body_parts.append(f"  - {finding}")
            body_parts.append("")

        body = "\n".join(body_parts)

        report = CuriosityReport(
            report_id=self._new_report_id(),
            subject=subject,
            body=body,
            related_objects=list(set(related_objects)),
            emotion=emotion,
            confidence=0.7 if all_findings else 0.4,
            ticket_ids=[t.ticket_id for t in tickets],
        )

        self.reports.append(report)
        return report

    def get_latest_report(self) -> Optional[CuriosityReport]:
        """Get most recent report."""
        return self.reports[-1] if self.reports else None

    # =========================================================================
    # Main Loop Entry Points
    # =========================================================================

    def tick(self) -> Optional[CuriosityReport]:
        """Run one curiosity cycle (discovery, investigation, report).

        This is meant to be called periodically (e.g., every few minutes).

        Returns:
            CuriosityReport if something interesting happened, else None
        """
        # Check if we should be curious right now
        state = self.world.state
        if state.attention_budget < 0.1:
            logger.debug("Low attention budget, skipping curiosity tick")
            return None

        # Discovery phase
        if len(self.world.objects) == 0 or state.discoveries_today == 0:
            # Initial discovery or new day
            discoveries = self.run_discovery_sweep()
            if discoveries:
                state.discoveries_today = sum(len(v) for v in discoveries.values())

        # Investigation phase
        candidates = self.world.get_curiosity_candidates(top_n=3)
        investigated = False

        for obj in candidates:
            if should_investigate(obj):
                # Create ticket for top candidate
                question = self._generate_question(obj)
                ticket = self.create_ticket(question, obj.obj_id)
                if ticket:
                    self.investigate_ticket(ticket.ticket_id)
                    investigated = True
                    break  # One investigation per tick

        # Report if we investigated
        if investigated:
            return self.generate_report()

        return None

    def _generate_question(self, obj: WorldObject) -> str:
        """Generate a curiosity question about an object."""
        templates = {
            ObjectCategory.PCIE_DEVICE: [
                "What capabilities does {name} have?",
                "Is {name} being used by any driver?",
                "What's the link speed for {name}?",
            ],
            ObjectCategory.THERMAL_ZONE: [
                "What's the current temperature of {name}?",
                "Is {name} within safe operating limits?",
                "Has {name} temperature changed recently?",
            ],
            ObjectCategory.MEMORY_REGION: [
                "How much of {name} is currently in use?",
                "What's the memory pressure on {name}?",
            ],
        }

        category_templates = templates.get(obj.category, ["What can I learn about {name}?"])
        import random
        template = random.choice(category_templates)
        return template.format(name=obj.name)

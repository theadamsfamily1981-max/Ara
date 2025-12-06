"""
Scientist - LLM-Driven Hardware Investigation
==============================================

The Scientist upgrades CuriosityAgent from template-based probing
to genuine scientific inquiry:

1. Hypothesis Formulation: LLM proposes testable hypotheses
2. Experiment Design: LLM chooses appropriate probes
3. Evidence Evaluation: LLM interprets results
4. Knowledge Update: Findings feed back to world model + notebook

This is what turns Ara from a sysadmin script into a researcher.

The scientific method:
    Observe → Hypothesize → Experiment → Analyze → Record

Example flow:
    1. Observe: GPU temperature is 85°C
    2. Hypothesize: "GPU thermal throttling may correlate with PCIe link degradation"
    3. Experiment: Run lspci during load, check link speed
    4. Analyze: "Link speed dropped from x16 to x8 at 82°C"
    5. Record: Add to lab notebook, update world model
"""

import logging
import time
from typing import Optional, Dict, List, Any, Callable

from ara.curiosity.agent import (
    CuriosityAgent,
    CuriosityTicket,
    CuriosityReport,
    TicketStatus,
)
from ara.curiosity.world_model import WorldModel, WorldObject
from ara.curiosity.tools import ProbeType, ProbeResult, run_safe_probe
from ara.curiosity.intrinsic import IntrinsicMotivation, get_intrinsic_motivation
from ara.curiosity.notebook import LabNotebook


logger = logging.getLogger(__name__)


# =============================================================================
# LLM Interface (abstract)
# =============================================================================

class LLMEngine:
    """
    Abstract interface for LLM generation.

    Override this or pass a callable that implements generate(prompt) -> str.
    """

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from LLM."""
        raise NotImplementedError("Implement generate() or pass a callable")


class MockLLMEngine(LLMEngine):
    """Mock LLM for testing without actual model."""

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # Simple keyword-based responses for testing
        if "HYPOTHESIS" in prompt.upper():
            return "HYPOTHESIS: The device may be operating at reduced performance due to thermal constraints."
        elif "CONCLUSION" in prompt.upper():
            return "CONCLUSION: Data suggests normal operation.\nSTATUS: UNKNOWN\nFOLLOWUP: none"
        elif "tool" in prompt.lower():
            return "SENSORS"
        else:
            return "I observed the data but could not form a clear conclusion."


# =============================================================================
# Scientist Class
# =============================================================================

class Scientist(CuriosityAgent):
    """
    LLM-powered curiosity agent that uses the scientific method.

    Extends CuriosityAgent with:
    - Hypothesis formulation via LLM
    - Intelligent probe selection
    - Evidence-based analysis
    - Lab notebook integration
    - Intrinsic motivation for exploration prioritization
    """

    def __init__(
        self,
        world_model: WorldModel,
        llm_engine: Optional[LLMEngine] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        notebook_path: str = "var/lib/banos/lab_notebook.md",
        **kwargs,
    ):
        """
        Initialize the Scientist.

        Args:
            world_model: The world model to investigate
            llm_engine: LLMEngine instance for generation
            llm_fn: Alternative: callable(prompt) -> str
            notebook_path: Where to write the lab notebook
            **kwargs: Passed to CuriosityAgent
        """
        super().__init__(world_model, **kwargs)

        # LLM setup
        if llm_fn is not None:
            self._llm_fn = llm_fn
        elif llm_engine is not None:
            self._llm_fn = llm_engine.generate
        else:
            logger.warning("No LLM provided, using mock engine")
            self._llm_fn = MockLLMEngine().generate

        # Subsystems
        self.intrinsic = get_intrinsic_motivation()
        self.notebook = LabNotebook(notebook_path)

        logger.info("Scientist initialized with LLM-driven investigation")

    def _llm_generate(self, prompt: str) -> str:
        """Generate response from LLM with error handling."""
        try:
            return self._llm_fn(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    # =========================================================================
    # Hypothesis Formulation
    # =========================================================================

    def formulate_hypothesis(self, obj: WorldObject) -> str:
        """
        Use LLM to propose a testable hypothesis about an object.

        Args:
            obj: WorldObject to investigate

        Returns:
            Hypothesis string
        """
        context = f"""Object: {obj.name} ({obj.category.name})
Properties: {obj.properties}
Uncertainty: {obj.uncertainty:.2f}
Investigation count: {obj.investigation_count}
Notes: {obj.notes[-3:] if obj.notes else 'None'}"""

        prompt = f"""You are Ara's on-board scientist, investigating hardware behavior.

{context}

Task: Propose ONE specific, testable hypothesis about this object that can be probed with standard Linux tools
(like lspci, sensors, dmesg, nvme-cli, etc.).

Be specific and practical. Focus on:
- Performance characteristics
- Thermal behavior
- Power states
- Error conditions
- Interaction with other components

Format STRICTLY as:

HYPOTHESIS: <text>
"""

        raw = self._llm_generate(prompt)

        if "HYPOTHESIS:" in raw.upper():
            # Extract hypothesis
            idx = raw.upper().find("HYPOTHESIS:")
            return raw[idx + 11:].strip().split('\n')[0]

        return raw.strip()[:200] if raw else "Unable to formulate hypothesis"

    # =========================================================================
    # Probe Selection
    # =========================================================================

    def _choose_probe(self, ticket: CuriosityTicket) -> ProbeType:
        """
        Ask LLM which probe type to use for this investigation.

        Args:
            ticket: Investigation ticket with hypothesis

        Returns:
            Selected ProbeType
        """
        tool_names = [t.name for t in ProbeType]
        allowed_names = [p.name for p in ticket.allowed_probes] if ticket.allowed_probes else tool_names

        prompt = f"""Hypothesis: {ticket.question}

Available tools: {", ".join(allowed_names)}

Tool descriptions:
- LSPCI: PCIe device enumeration (bus topology, link speed, driver)
- DMESG: Kernel messages (errors, driver events, hardware detection)
- SENSORS: Hardware sensors (temperatures, voltages, fan speeds)
- MEMORY: Memory info (usage, pressure)
- CPU: CPU information (model, frequency, cache)
- NETWORK: Network interfaces
- NVME: NVMe device status
- FPGA: FPGA/accelerator status

Pick ONE tool that will produce the most relevant evidence for this hypothesis.
Answer with ONLY the tool name, nothing else.
"""

        raw = self._llm_generate(prompt).strip().upper()

        # Try to match to ProbeType
        for probe in ProbeType:
            if probe.name == raw:
                return probe

        # Fallback based on object category
        target = self.world.get_object(ticket.target_obj_id)
        if target:
            from ara.curiosity.world_model import ObjectCategory
            category_probes = {
                ObjectCategory.PCIE_DEVICE: ProbeType.LSPCI,
                ObjectCategory.THERMAL_ZONE: ProbeType.SENSORS,
                ObjectCategory.MEMORY_REGION: ProbeType.MEMORY,
                ObjectCategory.STORAGE_DEVICE: ProbeType.NVME,
                ObjectCategory.FPGA_REGION: ProbeType.FPGA,
            }
            if target.category in category_probes:
                return category_probes[target.category]

        logger.warning(f"LLM chose unknown tool '{raw}', falling back to SENSORS")
        return ProbeType.SENSORS

    # =========================================================================
    # Experiment Execution
    # =========================================================================

    def run_experiment(self, ticket: CuriosityTicket) -> bool:
        """
        Run a scientific investigation on a ticket.

        Steps:
        1. Get prior state (for surprise calculation)
        2. Choose probe
        3. Execute probe
        4. Analyze evidence with LLM
        5. Compute surprise and log to notebook

        Args:
            ticket: Investigation ticket

        Returns:
            True if experiment succeeded
        """
        obj = self.world.get_object(ticket.target_obj_id)
        if obj is None:
            logger.error(f"Target object not found: {ticket.target_obj_id}")
            return False

        ticket.status = TicketStatus.IN_PROGRESS
        logger.info(f"🧪 Investigating {obj.name}: {ticket.question}")

        # Capture prior state for surprise calculation
        prior_summary = f"{obj.name}: uncertainty={obj.uncertainty:.2f}, notes={len(obj.notes)}"

        # Choose probe
        probe = self._choose_probe(ticket)
        logger.debug(f"Selected probe: {probe.name}")

        # Run probe
        probe_result = run_safe_probe(probe)
        if not probe_result.success:
            ticket.findings.append(f"Probe {probe.name} failed: {probe_result.error}")
            logger.warning(f"Probe failed: {probe_result.error}")
            return False

        # Store which probes were used (reusing allowed_probes field for compatibility)
        if hasattr(ticket, 'probes_used'):
            ticket.probes_used.append(probe.name)
        else:
            # Mutate allowed_probes to track what we actually used
            ticket.allowed_probes = [probe]

        # Analyze evidence with LLM
        analysis = self._analyze_evidence(ticket, probe_result)
        ticket.findings.append(analysis)

        # Parse analysis for status
        status = self._parse_analysis_status(analysis)

        # Update world model
        self.world.add_note(
            ticket.target_obj_id,
            f"[{probe.name}] {status}: {analysis[:100]}..."
        )
        self.world.update_observation(ticket.target_obj_id)

        # Compute surprise
        posterior_summary = f"{obj.name}: uncertainty={obj.uncertainty:.2f}, notes={len(obj.notes)}"
        surprise = self.intrinsic.calculate_surprise_from_text(prior_summary, posterior_summary)

        # Log to notebook
        self.notebook.record_experiment(
            ticket=ticket,
            probe_name=probe.name,
            result_snippet=probe_result.output[:500] if probe_result.output else "",
            analysis=analysis,
            surprise_score=surprise,
        )

        # Mark complete
        ticket.status = TicketStatus.RESOLVED
        ticket.resolved_at = time.time()
        ticket.current_depth += 1

        logger.info(f"Experiment complete: surprise={surprise:.2f}, status={status}")
        return True

    def _analyze_evidence(
        self,
        ticket: CuriosityTicket,
        probe_result: ProbeResult,
    ) -> str:
        """
        Ask LLM to interpret probe evidence.

        Args:
            ticket: Investigation ticket with hypothesis
            probe_result: Raw probe output

        Returns:
            Analysis string
        """
        # Truncate large outputs
        data_snippet = (probe_result.output or "")[:4000]

        prompt = f"""You are Ara's hardware researcher.

Hypothesis:
{ticket.question}

Tool used: {probe_result.probe_type.name}
Raw data (truncated):
\"\"\"{data_snippet}\"\"\"

Analyze this evidence:
1. Does this data SUPPORT, CONTRADICT, or FAIL TO ADDRESS the hypothesis?
2. What did we learn about this device?
3. If useful, suggest ONE specific follow-up measurement.

Format:

CONCLUSION: <one-paragraph summary>
STATUS: <SUPPORT|CONTRADICT|UNKNOWN>
FOLLOWUP: <single sentence or 'none'>
"""

        return self._llm_generate(prompt)

    def _parse_analysis_status(self, analysis: str) -> str:
        """Extract status from analysis."""
        analysis_upper = analysis.upper()
        if "SUPPORT" in analysis_upper:
            return "SUPPORTED"
        elif "CONTRADICT" in analysis_upper:
            return "CONTRADICTED"
        else:
            return "UNKNOWN"

    # =========================================================================
    # Investigation Prioritization
    # =========================================================================

    def get_investigation_priorities(self, top_n: int = 5) -> List[WorldObject]:
        """
        Get objects prioritized by intrinsic motivation.

        Uses IntrinsicMotivation to rank objects based on:
        - Current system entropy (bored vs overwhelmed)
        - Object uncertainty
        - Object importance

        Args:
            top_n: How many candidates to return

        Returns:
            List of WorldObject sorted by investigation value
        """
        candidates = list(self.world.objects.values())
        ranked = self.intrinsic.rank_investigation_candidates(candidates)
        return ranked[:top_n]

    # =========================================================================
    # Override: Generate Question as Hypothesis
    # =========================================================================

    def _generate_question(self, obj: WorldObject) -> str:
        """Generate a hypothesis (not just a template question)."""
        return self.formulate_hypothesis(obj)

    # =========================================================================
    # Override: Investigate with Scientific Method
    # =========================================================================

    def investigate_ticket(self, ticket_id: str) -> bool:
        """
        Run scientific investigation for a ticket.

        This overrides the parent method to use the experiment loop.
        """
        ticket = self.active_tickets.get(ticket_id)
        if not ticket:
            logger.warning(f"Ticket not found: {ticket_id}")
            return False

        if ticket.is_expired():
            ticket.status = TicketStatus.CANCELLED
            ticket.findings.append("Investigation timed out")
            return False

        # Run the experiment
        success = self.run_experiment(ticket)

        # Move to completed
        if ticket_id in self.active_tickets:
            del self.active_tickets[ticket_id]
            self.completed_tickets.append(ticket)

        return success

    # =========================================================================
    # Enhanced Tick
    # =========================================================================

    def tick(self) -> Optional[CuriosityReport]:
        """
        Run one curiosity cycle with scientific method.

        Uses intrinsic motivation for prioritization.
        """
        # Check attention budget
        state = self.world.state
        if state.attention_budget < 0.1:
            logger.debug("Low attention budget, skipping tick")
            return None

        # Discovery if needed
        if len(self.world.objects) == 0 or state.discoveries_today == 0:
            discoveries = self.run_discovery_sweep()
            if discoveries:
                state.discoveries_today = sum(len(v) for v in discoveries.values())

        # Use intrinsic motivation for prioritization
        candidates = self.get_investigation_priorities(top_n=3)
        investigated = False

        for obj in candidates:
            # Check if this is worth investigating
            voi = self.intrinsic.compute_value_of_information(
                obj.effective_uncertainty(),
                obj.importance,
            )

            if voi > 0.3:  # Threshold for investigation
                hypothesis = self.formulate_hypothesis(obj)
                ticket = self.create_ticket(hypothesis, obj.obj_id)
                if ticket:
                    self.investigate_ticket(ticket.ticket_id)
                    investigated = True
                    break

        # Generate report if we investigated
        if investigated:
            return self.generate_report()

        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Scientist',
    'LLMEngine',
    'MockLLMEngine',
]

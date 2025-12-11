# ara/jobs/grant.py
"""
Grant Job System
=================

First-class job type for grant discovery, qualification, writing, and tracking.

Grants are structured jobs with:
    - State machine: INTAKE → QUALIFY → DECOMPOSE → DRAFT → REVIEW → SUBMIT → TRACK
    - Fair-trade index: Only proceed if value/cost ratio meets threshold
    - IP exposure controls: Core IP never exposed, only "vertical skins"
    - Low overhead: Job is a text document (YAML spec)

Usage:
    from ara.jobs.grant import GrantJob, load_grant_spec

    # Load from YAML
    spec = load_grant_spec("grants/sbir_phase1.yaml")
    job = GrantJob(spec)

    # Check if worth pursuing
    if job.is_fair_trade():
        await job.run_pipeline(ara_kernel)

The fair-trade philosophy:
    - No jobs go forward if fairness_index < 1.0
    - No jobs expose IP beyond their configured max_exposure level
    - Value flows both ways: Ara learns, user gets output

IP Exposure Levels:
    - core_ip_protected: NIB/MEIS/QUANTA, recursive self-improvement, teleology
    - vertical_only: Industry-specific implementations (restaurant, music, etc.)
    - public: General methods, open frameworks
"""

from __future__ import annotations

import asyncio
import logging
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

log = logging.getLogger("Ara.Jobs.Grant")


# =============================================================================
# Enums and Constants
# =============================================================================

class GrantStage(str, Enum):
    """Grant pipeline stages."""
    INTAKE = "intake"           # Initial scan, parse opportunity
    QUALIFY = "qualify"         # Does this match our goals? Worth pursuing?
    DECOMPOSE = "decompose"     # Break into sections, assign agents
    DRAFT = "draft"             # Generate section content
    REVIEW = "review"           # Human review, agent cross-check
    SUBMIT = "submit"           # Package and submit
    TRACK = "track"             # Post-submission monitoring


class IPExposure(str, Enum):
    """IP exposure levels - what can be shared externally."""
    CORE_IP_PROTECTED = "core_ip_protected"  # Never expose (NIB/MEIS/QUANTA)
    VERTICAL_ONLY = "vertical_only"          # Industry-specific skins only
    PUBLIC = "public"                        # General methods, open


class GrantMode(str, Enum):
    """Operating modes for grant interaction."""
    DISCOVER = "discover"   # Find and qualify opportunities
    QUALIFY = "qualify"     # Evaluate a specific opportunity
    WRITE = "write"         # Draft grant content
    SUBMIT = "submit"       # Package and submit
    MANAGE = "manage"       # Track submitted grants


# Protected IP topics - never include in grants
PROTECTED_IP = frozenset([
    "nib",
    "meis",
    "quanta",
    "neuroimplicit_binding",
    "meta_experience_integration",
    "quantum_associative_network",
    "recursive_self_improvement",
    "teleology_engine",
    "ouroboros",
    "soul_driver",
    "fpga_soul",
])


# =============================================================================
# Fair Trade Calculation
# =============================================================================

@dataclass
class FairTradeMetrics:
    """Metrics for fair-trade index calculation."""

    # Value to Ara
    learning_value: float = 0.0      # What Ara learns (0-1)
    capability_growth: float = 0.0   # New capabilities gained (0-1)
    relationship_value: float = 0.0  # Strategic relationship (0-1)
    revenue_potential: float = 0.0   # Direct revenue (0-1)

    # Costs
    compute_cost: float = 0.0        # LLM tokens, GPU time (0-1)
    human_time_cost: float = 0.0     # Human review hours (0-1)
    opportunity_cost: float = 0.0    # What else could we do? (0-1)
    ip_exposure_cost: float = 0.0    # Risk of IP leakage (0-1)

    @property
    def value_score(self) -> float:
        """Total value (weighted sum)."""
        return (
            self.learning_value * 0.3 +
            self.capability_growth * 0.2 +
            self.relationship_value * 0.2 +
            self.revenue_potential * 0.3
        )

    @property
    def cost_score(self) -> float:
        """Total cost (weighted sum)."""
        return (
            self.compute_cost * 0.2 +
            self.human_time_cost * 0.3 +
            self.opportunity_cost * 0.2 +
            self.ip_exposure_cost * 0.3
        )

    @property
    def fair_trade_index(self) -> float:
        """
        Fair trade index: value / (cost + epsilon).

        >= 1.0: Fair trade, proceed
        < 1.0: Unfair, decline or renegotiate
        """
        epsilon = 0.01  # Avoid division by zero
        return self.value_score / (self.cost_score + epsilon)


@dataclass
class FairTradePolicy:
    """Policy constraints for fair-trade decisions."""

    min_fair_trade_index: float = 1.0       # Minimum FTI to proceed
    max_compute_tokens: int = 100_000       # Max LLM tokens
    max_human_hours: float = 4.0            # Max human review time
    max_ip_exposure: IPExposure = IPExposure.VERTICAL_ONLY

    # Learning requirements
    require_learning: bool = True           # Must learn something
    min_learning_value: float = 0.1         # Minimum learning score

    # Auto-decline triggers
    decline_if_no_attribution: bool = True  # Decline if can't attribute
    decline_competing_grants: bool = True   # Don't help competitors


def calculate_fair_trade(
    spec: Dict[str, Any],
    policy: Optional[FairTradePolicy] = None,
) -> FairTradeMetrics:
    """
    Calculate fair-trade metrics from grant spec.

    Args:
        spec: Grant specification dict
        policy: Fair trade policy constraints

    Returns:
        FairTradeMetrics with calculated values
    """
    policy = policy or FairTradePolicy()
    fair_trade_block = spec.get("fair_trade", {})

    metrics = FairTradeMetrics()

    # Value extraction
    value_block = fair_trade_block.get("value_to_ara", {})
    metrics.learning_value = value_block.get("learning", 0.5)
    metrics.capability_growth = value_block.get("capability", 0.3)
    metrics.relationship_value = value_block.get("relationship", 0.3)
    metrics.revenue_potential = fair_trade_block.get("revenue_share", 0) / 100

    # Cost extraction
    budget = fair_trade_block.get("budget", {})

    # Normalize compute cost (100k tokens = 1.0)
    compute_tokens = budget.get("max_compute_tokens", 50000)
    metrics.compute_cost = min(1.0, compute_tokens / 100000)

    # Normalize human time (8 hours = 1.0)
    human_hours = budget.get("max_human_hours", 2)
    metrics.human_time_cost = min(1.0, human_hours / 8)

    # Opportunity cost based on priority
    priority = spec.get("priority", "normal")
    metrics.opportunity_cost = {
        "urgent": 0.1,
        "high": 0.2,
        "normal": 0.3,
        "low": 0.5,
    }.get(priority, 0.3)

    # IP exposure cost
    max_exposure = fair_trade_block.get("max_exposure", "vertical_only")
    metrics.ip_exposure_cost = {
        "core_ip_protected": 0.0,
        "vertical_only": 0.2,
        "public": 0.5,
    }.get(max_exposure, 0.2)

    return metrics


# =============================================================================
# Grant Artifacts
# =============================================================================

@dataclass
class GrantArtifact:
    """Single artifact produced during grant pipeline."""
    name: str
    artifact_type: str  # "section", "budget", "appendix", "letter", etc.
    content: str
    stage: GrantStage
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed: bool = False
    reviewer_notes: str = ""


@dataclass
class GrantArtifacts:
    """Collection of artifacts for a grant."""
    items: Dict[str, GrantArtifact] = field(default_factory=dict)

    def add(self, artifact: GrantArtifact) -> None:
        """Add artifact."""
        self.items[artifact.name] = artifact

    def get(self, name: str) -> Optional[GrantArtifact]:
        """Get artifact by name."""
        return self.items.get(name)

    def by_stage(self, stage: GrantStage) -> List[GrantArtifact]:
        """Get all artifacts from a stage."""
        return [a for a in self.items.values() if a.stage == stage]

    def by_type(self, artifact_type: str) -> List[GrantArtifact]:
        """Get all artifacts of a type."""
        return [a for a in self.items.values() if a.artifact_type == artifact_type]

    def to_dict(self) -> Dict[str, Any]:
        """Export to dict."""
        return {
            name: {
                "type": a.artifact_type,
                "content": a.content[:500] + "..." if len(a.content) > 500 else a.content,
                "stage": a.stage.value,
                "reviewed": a.reviewed,
            }
            for name, a in self.items.items()
        }


# =============================================================================
# Grant Job
# =============================================================================

@dataclass
class GrantJob:
    """
    A grant job with full pipeline state machine.

    The job is driven by a YAML spec that defines:
        - Grant metadata (title, deadline, amount)
        - Input sources (prior art, data, etc.)
        - Deliverables (sections, budgets, appendices)
        - Fair trade constraints
        - IP exposure limits
    """

    spec: Dict[str, Any]
    stage: GrantStage = GrantStage.INTAKE
    artifacts: GrantArtifacts = field(default_factory=GrantArtifacts)

    # State tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stage_history: List[Dict[str, Any]] = field(default_factory=list)

    # Qualification results
    qualified: Optional[bool] = None
    qualification_notes: str = ""

    # Fair trade
    fair_trade_metrics: Optional[FairTradeMetrics] = None
    fair_trade_policy: FairTradePolicy = field(default_factory=FairTradePolicy)

    # Callbacks
    _stage_callbacks: Dict[GrantStage, List[Callable]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate spec and initialize metrics."""
        self._validate_spec()
        self.fair_trade_metrics = calculate_fair_trade(self.spec, self.fair_trade_policy)

    def _validate_spec(self) -> None:
        """Validate grant spec has required fields."""
        required = ["grant", "job_type"]
        for field_name in required:
            if field_name not in self.spec:
                raise ValueError(f"Grant spec missing required field: {field_name}")

        if self.spec.get("job_type") != "grant":
            raise ValueError(f"Expected job_type='grant', got '{self.spec.get('job_type')}'")

    # =========================================================================
    # Fair Trade
    # =========================================================================

    def is_fair_trade(self) -> bool:
        """Check if grant meets fair trade requirements."""
        if self.fair_trade_metrics is None:
            return False

        fti = self.fair_trade_metrics.fair_trade_index

        # Check minimum FTI
        if fti < self.fair_trade_policy.min_fair_trade_index:
            log.info("Grant fails fair trade: FTI=%.2f < %.2f",
                     fti, self.fair_trade_policy.min_fair_trade_index)
            return False

        # Check learning requirement
        if self.fair_trade_policy.require_learning:
            if self.fair_trade_metrics.learning_value < self.fair_trade_policy.min_learning_value:
                log.info("Grant fails fair trade: learning=%.2f < %.2f",
                         self.fair_trade_metrics.learning_value,
                         self.fair_trade_policy.min_learning_value)
                return False

        return True

    def get_fair_trade_report(self) -> Dict[str, Any]:
        """Get detailed fair trade report."""
        if self.fair_trade_metrics is None:
            return {"error": "No fair trade metrics calculated"}

        m = self.fair_trade_metrics
        return {
            "fair_trade_index": m.fair_trade_index,
            "is_fair": self.is_fair_trade(),
            "value": {
                "total": m.value_score,
                "learning": m.learning_value,
                "capability": m.capability_growth,
                "relationship": m.relationship_value,
                "revenue": m.revenue_potential,
            },
            "cost": {
                "total": m.cost_score,
                "compute": m.compute_cost,
                "human_time": m.human_time_cost,
                "opportunity": m.opportunity_cost,
                "ip_exposure": m.ip_exposure_cost,
            },
            "policy": {
                "min_fti": self.fair_trade_policy.min_fair_trade_index,
                "max_exposure": self.fair_trade_policy.max_ip_exposure.value,
            },
        }

    # =========================================================================
    # IP Protection
    # =========================================================================

    def check_ip_exposure(self, content: str) -> List[str]:
        """
        Check content for IP exposure violations.

        Returns list of protected terms found.
        """
        content_lower = content.lower()
        violations = []

        for term in PROTECTED_IP:
            if term in content_lower:
                violations.append(term)

        return violations

    def get_max_exposure(self) -> IPExposure:
        """Get maximum allowed IP exposure level."""
        exposure_str = self.spec.get("fair_trade", {}).get("max_exposure", "vertical_only")
        try:
            return IPExposure(exposure_str)
        except ValueError:
            return IPExposure.VERTICAL_ONLY

    def sanitize_for_exposure(self, content: str, exposure: IPExposure) -> str:
        """
        Sanitize content for given exposure level.

        Replaces protected terms with generic alternatives.
        """
        if exposure == IPExposure.CORE_IP_PROTECTED:
            # Most restrictive - remove all specific methods
            replacements = {
                "neuroimplicit binding": "neural association method",
                "meta-experience integration": "learning integration system",
                "quantum associative": "efficient associative",
                "recursive self-improvement": "iterative refinement",
                "teleology": "goal-directed",
                "ouroboros": "self-evolution",
            }
        elif exposure == IPExposure.VERTICAL_ONLY:
            # Only industry-specific, no core methods
            replacements = {
                "nib": "binding system",
                "meis": "integration layer",
                "quanta": "memory network",
            }
        else:
            # Public - minimal changes
            replacements = {}

        result = content
        for term, replacement in replacements.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            result = pattern.sub(replacement, result)

        return result

    # =========================================================================
    # Stage Machine
    # =========================================================================

    def advance_stage(self) -> GrantStage:
        """
        Advance to next stage in pipeline.

        Returns the new stage.
        """
        stage_order = list(GrantStage)
        current_idx = stage_order.index(self.stage)

        if current_idx < len(stage_order) - 1:
            old_stage = self.stage
            self.stage = stage_order[current_idx + 1]

            self.stage_history.append({
                "from": old_stage.value,
                "to": self.stage.value,
                "timestamp": datetime.utcnow().isoformat(),
            })

            log.info("Grant advanced: %s → %s", old_stage.value, self.stage.value)

        return self.stage

    def set_stage(self, stage: GrantStage) -> None:
        """Explicitly set stage (for rollback or skip)."""
        old_stage = self.stage
        self.stage = stage

        self.stage_history.append({
            "from": old_stage.value,
            "to": self.stage.value,
            "timestamp": datetime.utcnow().isoformat(),
            "explicit": True,
        })

    def on_stage(self, stage: GrantStage, callback: Callable) -> None:
        """Register callback for stage entry."""
        if stage not in self._stage_callbacks:
            self._stage_callbacks[stage] = []
        self._stage_callbacks[stage].append(callback)

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    async def run_step(self, kernel: Any = None) -> Dict[str, Any]:
        """
        Run one step of the grant pipeline.

        Args:
            kernel: Ara kernel with access to agents, memory, policies

        Returns:
            Step result dict with status and any outputs
        """
        if self.started_at is None:
            self.started_at = datetime.utcnow()

        # Fire stage callbacks
        for callback in self._stage_callbacks.get(self.stage, []):
            try:
                callback(self)
            except Exception as e:
                log.warning("Stage callback error: %s", e)

        # Dispatch to stage handler
        handlers = {
            GrantStage.INTAKE: self._run_intake,
            GrantStage.QUALIFY: self._run_qualify,
            GrantStage.DECOMPOSE: self._run_decompose,
            GrantStage.DRAFT: self._run_draft,
            GrantStage.REVIEW: self._run_review,
            GrantStage.SUBMIT: self._run_submit,
            GrantStage.TRACK: self._run_track,
        }

        handler = handlers.get(self.stage)
        if handler is None:
            return {"status": "error", "message": f"No handler for stage {self.stage}"}

        return await handler(kernel)

    async def run_pipeline(self, kernel: Any = None) -> Dict[str, Any]:
        """
        Run full pipeline from current stage to completion.

        Stops if:
            - Fair trade check fails
            - Stage handler returns error
            - Reached TRACK stage
        """
        results = []

        while self.stage != GrantStage.TRACK:
            # Fair trade gate
            if not self.is_fair_trade():
                return {
                    "status": "declined",
                    "reason": "fair_trade_check_failed",
                    "fair_trade_report": self.get_fair_trade_report(),
                    "results": results,
                }

            # Run current stage
            result = await self.run_step(kernel)
            results.append({
                "stage": self.stage.value,
                "result": result,
            })

            if result.get("status") == "error":
                return {
                    "status": "error",
                    "stage": self.stage.value,
                    "results": results,
                }

            if result.get("status") == "blocked":
                return {
                    "status": "blocked",
                    "stage": self.stage.value,
                    "reason": result.get("reason"),
                    "results": results,
                }

            # Advance stage
            self.advance_stage()

        # Run final TRACK step
        result = await self.run_step(kernel)
        results.append({
            "stage": self.stage.value,
            "result": result,
        })

        self.completed_at = datetime.utcnow()

        return {
            "status": "completed",
            "results": results,
            "artifacts": self.artifacts.to_dict(),
        }

    # =========================================================================
    # Stage Handlers
    # =========================================================================

    async def _run_intake(self, kernel: Any) -> Dict[str, Any]:
        """
        INTAKE stage: Parse and validate grant opportunity.

        Outputs:
            - Parsed grant metadata
            - Initial timeline
            - Required sections list
        """
        grant_info = self.spec.get("grant", {})

        # Extract key info
        title = grant_info.get("title", "Untitled Grant")
        deadline = grant_info.get("deadline")
        amount = grant_info.get("amount", {})

        # Create intake artifact
        intake_summary = f"""
Grant Intake Summary
====================
Title: {title}
Organization: {grant_info.get('organization', 'Unknown')}
Program: {grant_info.get('program', 'Unknown')}
Deadline: {deadline}
Amount: ${amount.get('min', 0):,} - ${amount.get('max', 0):,}

Eligibility:
{yaml.dump(grant_info.get('eligibility', {}), default_flow_style=False)}

Required Sections:
{yaml.dump(self.spec.get('deliverables', {}).get('sections', []), default_flow_style=False)}
        """.strip()

        self.artifacts.add(GrantArtifact(
            name="intake_summary",
            artifact_type="summary",
            content=intake_summary,
            stage=GrantStage.INTAKE,
        ))

        log.info("INTAKE: Parsed grant '%s'", title)

        return {
            "status": "ok",
            "title": title,
            "deadline": deadline,
            "sections_count": len(self.spec.get("deliverables", {}).get("sections", [])),
        }

    async def _run_qualify(self, kernel: Any) -> Dict[str, Any]:
        """
        QUALIFY stage: Evaluate if grant is worth pursuing.

        Checks:
            - Fair trade index
            - Deadline feasibility
            - Alignment with goals
            - IP exposure risk
        """
        # Fair trade is already calculated, but we do detailed checks here
        fti_report = self.get_fair_trade_report()

        # Check IP exposure
        max_exposure = self.get_max_exposure()
        if max_exposure == IPExposure.CORE_IP_PROTECTED:
            # Very restrictive - only if truly strategic
            if self.fair_trade_metrics.relationship_value < 0.7:
                self.qualified = False
                self.qualification_notes = "Core IP protection requires high relationship value"
                return {
                    "status": "blocked",
                    "reason": "ip_exposure_too_high",
                    "qualified": False,
                }

        # Check deadline feasibility (if kernel provides time estimation)
        deadline = self.spec.get("grant", {}).get("deadline")
        if deadline:
            # Simple check - real implementation would estimate actual work
            self.artifacts.add(GrantArtifact(
                name="deadline_check",
                artifact_type="analysis",
                content=f"Deadline: {deadline}\nFeasibility: Assumed OK (detailed estimation TBD)",
                stage=GrantStage.QUALIFY,
            ))

        # Check alignment
        goals = self.spec.get("alignment", {}).get("ara_goals", [])
        alignment_score = len(goals) * 0.2  # Simple scoring

        # Final qualification decision
        self.qualified = (
            fti_report["is_fair"] and
            alignment_score >= 0.2
        )

        self.qualification_notes = f"FTI={fti_report['fair_trade_index']:.2f}, Alignment={alignment_score:.2f}"

        self.artifacts.add(GrantArtifact(
            name="qualification_report",
            artifact_type="report",
            content=f"""
Qualification Report
====================
Fair Trade Index: {fti_report['fair_trade_index']:.2f}
Is Fair: {fti_report['is_fair']}
Alignment Score: {alignment_score:.2f}
Qualified: {self.qualified}

Notes: {self.qualification_notes}

Fair Trade Details:
{yaml.dump(fti_report, default_flow_style=False)}
            """.strip(),
            stage=GrantStage.QUALIFY,
        ))

        log.info("QUALIFY: %s (FTI=%.2f)", "PASS" if self.qualified else "FAIL",
                 fti_report["fair_trade_index"])

        if not self.qualified:
            return {
                "status": "blocked",
                "reason": "qualification_failed",
                "qualified": False,
                "fair_trade_index": fti_report["fair_trade_index"],
            }

        return {
            "status": "ok",
            "qualified": True,
            "fair_trade_index": fti_report["fair_trade_index"],
            "alignment_score": alignment_score,
        }

    async def _run_decompose(self, kernel: Any) -> Dict[str, Any]:
        """
        DECOMPOSE stage: Break grant into sections and assign agents.

        Creates work items for each deliverable section.
        """
        deliverables = self.spec.get("deliverables", {})
        sections = deliverables.get("sections", [])

        # Create decomposition plan
        work_items = []

        for section in sections:
            section_name = section if isinstance(section, str) else section.get("name", "unknown")
            word_count = 500 if isinstance(section, str) else section.get("word_count", 500)

            work_items.append({
                "section": section_name,
                "word_count": word_count,
                "agent": "grant_writer",  # Could be more specific
                "status": "pending",
            })

        # Add supporting documents
        for doc in deliverables.get("attachments", []):
            work_items.append({
                "section": doc,
                "word_count": 0,
                "agent": "document_generator",
                "status": "pending",
                "type": "attachment",
            })

        decomposition_plan = {
            "total_sections": len(sections),
            "total_attachments": len(deliverables.get("attachments", [])),
            "estimated_words": sum(w.get("word_count", 0) for w in work_items),
            "work_items": work_items,
        }

        self.artifacts.add(GrantArtifact(
            name="decomposition_plan",
            artifact_type="plan",
            content=yaml.dump(decomposition_plan, default_flow_style=False),
            stage=GrantStage.DECOMPOSE,
        ))

        log.info("DECOMPOSE: %d sections, %d attachments",
                 decomposition_plan["total_sections"],
                 decomposition_plan["total_attachments"])

        return {
            "status": "ok",
            "sections": len(sections),
            "work_items": len(work_items),
        }

    async def _run_draft(self, kernel: Any) -> Dict[str, Any]:
        """
        DRAFT stage: Generate content for each section.

        Uses kernel agents if available, otherwise creates placeholders.
        """
        plan_artifact = self.artifacts.get("decomposition_plan")
        if plan_artifact is None:
            return {"status": "error", "message": "No decomposition plan found"}

        plan = yaml.safe_load(plan_artifact.content)
        work_items = plan.get("work_items", [])

        drafted = 0
        max_exposure = self.get_max_exposure()

        for item in work_items:
            section_name = item.get("section", "unknown")
            word_count = item.get("word_count", 500)

            # Generate draft content
            # In real implementation, this would use kernel.agents.grant_writer
            if kernel is not None and hasattr(kernel, "draft_section"):
                content = await kernel.draft_section(
                    grant_spec=self.spec,
                    section=section_name,
                    word_count=word_count,
                )
            else:
                # Placeholder for development
                content = f"""
[DRAFT: {section_name}]

This section requires {word_count} words covering:
- Overview of the proposed work
- Technical approach
- Expected outcomes

[Content to be generated by grant_writer agent]
                """.strip()

            # Check and sanitize IP exposure
            violations = self.check_ip_exposure(content)
            if violations:
                log.warning("IP violations in %s: %s", section_name, violations)
                content = self.sanitize_for_exposure(content, max_exposure)

            self.artifacts.add(GrantArtifact(
                name=f"draft_{section_name}",
                artifact_type="section",
                content=content,
                stage=GrantStage.DRAFT,
            ))

            drafted += 1

        log.info("DRAFT: Generated %d section drafts", drafted)

        return {
            "status": "ok",
            "drafted": drafted,
            "total_items": len(work_items),
        }

    async def _run_review(self, kernel: Any) -> Dict[str, Any]:
        """
        REVIEW stage: Cross-check and human review.

        Validates:
            - IP exposure compliance
            - Section completeness
            - Consistency across sections
        """
        draft_artifacts = self.artifacts.by_stage(GrantStage.DRAFT)

        review_results = []
        all_violations = []

        for artifact in draft_artifacts:
            # IP check
            violations = self.check_ip_exposure(artifact.content)
            if violations:
                all_violations.extend(violations)

            # Completeness check (simple heuristic)
            word_count = len(artifact.content.split())
            is_complete = word_count >= 100 and "[DRAFT:" not in artifact.content

            review_results.append({
                "section": artifact.name,
                "ip_violations": violations,
                "word_count": word_count,
                "complete": is_complete,
                "needs_revision": bool(violations) or not is_complete,
            })

        # Create review report
        needs_revision = [r for r in review_results if r.get("needs_revision")]

        self.artifacts.add(GrantArtifact(
            name="review_report",
            artifact_type="report",
            content=yaml.dump({
                "total_sections": len(review_results),
                "sections_needing_revision": len(needs_revision),
                "ip_violations_found": len(all_violations),
                "details": review_results,
            }, default_flow_style=False),
            stage=GrantStage.REVIEW,
        ))

        log.info("REVIEW: %d/%d sections need revision",
                 len(needs_revision), len(review_results))

        if all_violations:
            log.warning("REVIEW: IP violations found: %s", all_violations)

        return {
            "status": "ok",
            "reviewed": len(review_results),
            "needs_revision": len(needs_revision),
            "ip_violations": len(all_violations),
        }

    async def _run_submit(self, kernel: Any) -> Dict[str, Any]:
        """
        SUBMIT stage: Package and submit grant.

        Creates final submission package.
        """
        # Gather all approved sections
        sections = self.artifacts.by_type("section")

        # Create submission manifest
        submission = {
            "grant_title": self.spec.get("grant", {}).get("title"),
            "submitted_at": datetime.utcnow().isoformat(),
            "sections": [s.name for s in sections],
            "section_count": len(sections),
            "total_words": sum(len(s.content.split()) for s in sections),
        }

        self.artifacts.add(GrantArtifact(
            name="submission_manifest",
            artifact_type="manifest",
            content=yaml.dump(submission, default_flow_style=False),
            stage=GrantStage.SUBMIT,
        ))

        log.info("SUBMIT: Packaged %d sections (%d words)",
                 submission["section_count"], submission["total_words"])

        # In real implementation, this would interface with submission portal
        return {
            "status": "ok",
            "sections": submission["section_count"],
            "words": submission["total_words"],
            "manifest": "submission_manifest",
        }

    async def _run_track(self, kernel: Any) -> Dict[str, Any]:
        """
        TRACK stage: Post-submission monitoring.

        Sets up tracking for grant status.
        """
        tracking_info = {
            "grant_title": self.spec.get("grant", {}).get("title"),
            "submitted_at": self.artifacts.get("submission_manifest").created_at.isoformat()
                if self.artifacts.get("submission_manifest") else None,
            "expected_response": self.spec.get("grant", {}).get("response_date"),
            "tracking_status": "awaiting_response",
            "follow_up_dates": [],
        }

        self.artifacts.add(GrantArtifact(
            name="tracking_record",
            artifact_type="tracking",
            content=yaml.dump(tracking_info, default_flow_style=False),
            stage=GrantStage.TRACK,
        ))

        log.info("TRACK: Grant submitted, awaiting response")

        return {
            "status": "ok",
            "tracking_status": "awaiting_response",
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export job state to dict."""
        return {
            "spec": self.spec,
            "stage": self.stage.value,
            "qualified": self.qualified,
            "qualification_notes": self.qualification_notes,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "stage_history": self.stage_history,
            "artifacts": self.artifacts.to_dict(),
            "fair_trade": self.get_fair_trade_report(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrantJob":
        """Load job from dict."""
        job = cls(spec=data["spec"])
        job.stage = GrantStage(data.get("stage", "intake"))
        job.qualified = data.get("qualified")
        job.qualification_notes = data.get("qualification_notes", "")
        job.stage_history = data.get("stage_history", [])

        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])

        return job


# =============================================================================
# YAML Schema and Loading
# =============================================================================

GRANT_SPEC_SCHEMA = """
# Grant Specification YAML Schema
# ================================
#
# job_type: grant  # Required, identifies this as a grant job
# mode: discover | qualify | write | submit | manage
#
# grant:
#   title: "Grant Title"
#   organization: "Funding Organization"
#   program: "Specific Program Name"
#   deadline: "YYYY-MM-DD"
#   response_date: "YYYY-MM-DD"  # Expected response
#   amount:
#     min: 50000
#     max: 150000
#     currency: USD
#   eligibility:
#     - Small business
#     - US-based
#
# inputs:
#   prior_work:
#     - path/to/paper.pdf
#     - path/to/patent.pdf
#   data_sources:
#     - internal_metrics
#     - public_datasets
#   context_docs:
#     - company_overview.md
#
# deliverables:
#   sections:
#     - name: executive_summary
#       word_count: 500
#     - name: technical_approach
#       word_count: 3000
#     - name: team_qualifications
#       word_count: 1000
#   attachments:
#     - budget_spreadsheet
#     - letters_of_support
#     - prior_publications
#
# fair_trade:
#   max_exposure: vertical_only  # core_ip_protected | vertical_only | public
#   budget:
#     max_compute_tokens: 100000
#     max_human_hours: 4
#   value_to_ara:
#     learning: 0.7        # What Ara learns (0-1)
#     capability: 0.5      # New capabilities (0-1)
#     relationship: 0.6    # Strategic value (0-1)
#   revenue_share: 10      # % of grant if awarded
#
# alignment:
#   ara_goals:
#     - sovereign_revenue
#     - capability_growth
#   vertical: restaurant    # Industry vertical if applicable
#   strategic_value: high   # low | medium | high
"""


def load_grant_spec(path: str) -> Dict[str, Any]:
    """
    Load grant specification from YAML file.

    Args:
        path: Path to YAML spec file

    Returns:
        Parsed spec dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Grant spec not found: {path}")

    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    # Validate job type
    if spec.get("job_type") != "grant":
        raise ValueError(f"Invalid job_type: expected 'grant', got '{spec.get('job_type')}'")

    return spec


def create_grant_spec(
    title: str,
    organization: str,
    deadline: str,
    amount_min: int,
    amount_max: int,
    sections: List[str],
    max_exposure: str = "vertical_only",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a grant spec programmatically.

    Args:
        title: Grant title
        organization: Funding organization
        deadline: Deadline (YYYY-MM-DD)
        amount_min: Minimum award
        amount_max: Maximum award
        sections: List of section names
        max_exposure: IP exposure level
        **kwargs: Additional spec fields

    Returns:
        Grant spec dict
    """
    spec = {
        "job_type": "grant",
        "mode": "write",
        "grant": {
            "title": title,
            "organization": organization,
            "deadline": deadline,
            "amount": {
                "min": amount_min,
                "max": amount_max,
                "currency": "USD",
            },
        },
        "deliverables": {
            "sections": [
                {"name": s, "word_count": 500} for s in sections
            ],
        },
        "fair_trade": {
            "max_exposure": max_exposure,
            "budget": {
                "max_compute_tokens": 100000,
                "max_human_hours": 4,
            },
            "value_to_ara": {
                "learning": 0.5,
                "capability": 0.3,
                "relationship": 0.3,
            },
        },
    }

    # Merge additional kwargs
    for key, value in kwargs.items():
        if key in spec:
            if isinstance(spec[key], dict) and isinstance(value, dict):
                spec[key].update(value)
            else:
                spec[key] = value
        else:
            spec[key] = value

    return spec


# =============================================================================
# Convenience
# =============================================================================

def quick_qualify(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick qualification check without running full pipeline.

    Returns fair trade report and qualification status.
    """
    job = GrantJob(spec)
    return {
        "is_fair_trade": job.is_fair_trade(),
        "fair_trade_report": job.get_fair_trade_report(),
        "max_exposure": job.get_max_exposure().value,
    }


__all__ = [
    # Enums
    'GrantStage',
    'IPExposure',
    'GrantMode',

    # Fair Trade
    'FairTradeMetrics',
    'FairTradePolicy',
    'calculate_fair_trade',

    # Artifacts
    'GrantArtifact',
    'GrantArtifacts',

    # Main Job
    'GrantJob',

    # Schema and Loading
    'GRANT_SPEC_SCHEMA',
    'load_grant_spec',
    'create_grant_spec',

    # Convenience
    'quick_qualify',
    'PROTECTED_IP',
]

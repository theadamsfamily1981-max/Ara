# ara/academy/pipeline.py
"""
Academy Pipeline - The Skill Lifecycle Orchestrator
====================================================

Wires together the full skill lifecycle:

    Raw Logs → Extractor → Pattern Miner → Curriculum → Dojo → Registry

This is where Ara goes from "I saw something interesting" to
"I now have a battle-tested skill I can use."

Lifecycle Stages:
    1. DISCOVER: Extractor finds patterns in session logs
    2. EVALUATE: Causal Miner determines if patterns are causal (not just correlated)
    3. DECIDE: Curriculum Manager decides if worth internalizing
    4. ARCHITECT: Architect generalizes patterns into robust skills
    5. HARDEN: Dojo stress-tests the skill implementation
    6. DEPLOY: Registry makes skill available for use

Each stage can reject candidates, preventing cargo-cult learning.

Usage:
    from ara.academy.pipeline import SkillPipeline, get_pipeline

    pipeline = get_pipeline()

    # Run full pipeline on recent logs
    result = pipeline.run(days=7)

    # Check what we learned
    print(f"Discovered: {len(result.discovered)}")
    print(f"Internalized: {len(result.internalized)}")
    print(f"Rejected: {len(result.rejected)}")

    # You can also run individual stages
    patterns = pipeline.discover(days=7)
    evaluated = pipeline.evaluate(patterns)
    decisions = pipeline.decide(evaluated)
    skills = pipeline.architect(decisions)
    hardened = pipeline.harden(skills)
    deployed = pipeline.deploy(hardened)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# Local academy imports
from .skills.extractor import (
    SkillExtractor,
    SessionPattern,
    SkillProposal,
    get_skill_extractor,
)
from .skills.architect import (
    Architect,
    Episode,
    SkillSpec,
    get_architect,
)
from .curriculum.internalization import (
    CurriculumManager,
    get_curriculum_manager,
)
from .dojo import (
    Dojo,
    SkillSpec as DojoSkillSpec,
    HardeningReport,
    HardeningResult,
    get_dojo,
)
from .skills.registry import (
    SkillRegistry,
    LearnedSkill,
    get_skill_registry,
)

# Meta imports
from ara.meta.causal_miner import (
    CausalPatternMiner,
    CausalEstimate,
    ToolOutcome,
    get_causal_miner,
)
from ara.meta.meta_logger import get_meta_logger, MetaLogger

log = logging.getLogger("Ara.Pipeline")


class PipelineStage(str, Enum):
    """Pipeline stages."""
    DISCOVER = "discover"
    EVALUATE = "evaluate"
    DECIDE = "decide"
    ARCHITECT = "architect"
    HARDEN = "harden"
    DEPLOY = "deploy"


class RejectionReason(str, Enum):
    """Reasons a skill candidate was rejected."""
    NO_CAUSAL_EFFECT = "no_causal_effect"
    NEGATIVE_EFFECT = "negative_effect"
    NOT_WORTH_INTERNALIZING = "not_worth_internalizing"
    FAILED_HARDENING = "failed_hardening"
    LOW_CONFIDENCE = "low_confidence"
    ALREADY_EXISTS = "already_exists"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class SkillCandidate:
    """A skill moving through the pipeline."""

    id: str
    name: str
    pattern: Optional[SessionPattern] = None
    proposal: Optional[SkillProposal] = None
    causal_estimate: Optional[CausalEstimate] = None
    internalize_score: float = 0.0
    skill_spec: Optional[SkillSpec] = None
    hardening_report: Optional[HardeningReport] = None
    learned_skill: Optional[LearnedSkill] = None

    # Stage tracking
    current_stage: PipelineStage = PipelineStage.DISCOVER
    rejected: bool = False
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: str = ""

    # Timestamps
    discovered_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "stage": self.current_stage.value,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "internalize_score": self.internalize_score,
            "causal_effect": self.causal_estimate.delta if self.causal_estimate else None,
        }


@dataclass
class PipelineResult:
    """Results from running the pipeline."""

    # Counts
    discovered: List[SkillCandidate] = field(default_factory=list)
    evaluated: List[SkillCandidate] = field(default_factory=list)
    decided: List[SkillCandidate] = field(default_factory=list)
    architected: List[SkillCandidate] = field(default_factory=list)
    hardened: List[SkillCandidate] = field(default_factory=list)
    deployed: List[SkillCandidate] = field(default_factory=list)
    rejected: List[SkillCandidate] = field(default_factory=list)

    # Stats
    duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Pipeline: {len(self.discovered)} discovered → "
            f"{len(self.evaluated)} evaluated → "
            f"{len(self.decided)} decided → "
            f"{len(self.architected)} architected → "
            f"{len(self.hardened)} hardened → "
            f"{len(self.deployed)} deployed | "
            f"{len(self.rejected)} rejected"
        )


class SkillPipeline:
    """
    The skill lifecycle orchestrator.

    Takes raw interaction logs and produces battle-tested skills.

    Configuration:
        - causal_threshold: Minimum causal effect (Δ) to consider
        - internalize_threshold: Minimum internalization score
        - hardening_threshold: Minimum success rate in dojo
        - require_hardening: Whether to skip dojo for low-risk skills
    """

    def __init__(
        self,
        extractor: Optional[SkillExtractor] = None,
        causal_miner: Optional[CausalPatternMiner] = None,
        curriculum: Optional[CurriculumManager] = None,
        architect: Optional[Architect] = None,
        dojo: Optional[Dojo] = None,
        registry: Optional[SkillRegistry] = None,
        meta_logger: Optional[MetaLogger] = None,
        # Thresholds
        causal_threshold: float = 0.1,
        internalize_threshold: float = 0.5,
        hardening_threshold: float = 0.6,
        require_hardening: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            extractor: Skill extractor (default: get_skill_extractor())
            causal_miner: Causal pattern miner (default: get_causal_miner())
            curriculum: Curriculum manager (default: get_curriculum_manager())
            architect: Skill architect (default: get_architect())
            dojo: Training dojo (default: get_dojo())
            registry: Skill registry (default: get_skill_registry())
            meta_logger: Meta logger for reading logs (default: get_meta_logger())
            causal_threshold: Minimum Δ to proceed
            internalize_threshold: Minimum internalization score
            hardening_threshold: Minimum dojo success rate
            require_hardening: Whether dojo is mandatory
        """
        self.extractor = extractor or get_skill_extractor()
        self.causal_miner = causal_miner or get_causal_miner()
        self.curriculum = curriculum or get_curriculum_manager()
        self.architect = architect or get_architect()
        self.dojo = dojo or get_dojo()
        self.registry = registry or get_skill_registry()
        self.meta_logger = meta_logger or get_meta_logger()

        self.causal_threshold = causal_threshold
        self.internalize_threshold = internalize_threshold
        self.hardening_threshold = hardening_threshold
        self.require_hardening = require_hardening

        log.info(
            "SkillPipeline initialized (causal_th=%.2f, intern_th=%.2f, hard_th=%.2f)",
            causal_threshold, internalize_threshold, hardening_threshold
        )

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def run(self, days: int = 7, dry_run: bool = False) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            days: Days of logs to analyze
            dry_run: If True, don't actually deploy skills

        Returns:
            PipelineResult with all candidates and their fates
        """
        start_time = time.time()
        result = PipelineResult(started_at=datetime.utcnow())

        log.info("🚀 PIPELINE: Starting skill learning pipeline (days=%d)", days)

        # Stage 1: Discover patterns
        candidates = self.discover(days)
        result.discovered = candidates.copy()
        log.info("📍 Stage 1 DISCOVER: Found %d pattern candidates", len(candidates))

        # Stage 2: Evaluate causally
        candidates = self.evaluate(candidates)
        active = [c for c in candidates if not c.rejected]
        result.evaluated = active.copy()
        result.rejected.extend([c for c in candidates if c.rejected])
        log.info("📍 Stage 2 EVALUATE: %d passed causal check", len(active))

        # Stage 3: Decide on internalization
        candidates = self.decide(active)
        active = [c for c in candidates if not c.rejected]
        result.decided = active.copy()
        result.rejected.extend([c for c in candidates if c.rejected])
        log.info("📍 Stage 3 DECIDE: %d worth internalizing", len(active))

        # Stage 4: Architect robust skills
        candidates = self.architect_skills(active)
        active = [c for c in candidates if not c.rejected]
        result.architected = active.copy()
        result.rejected.extend([c for c in candidates if c.rejected])
        log.info("📍 Stage 4 ARCHITECT: %d skills architected", len(active))

        # Stage 5: Harden in dojo
        candidates = self.harden(active)
        active = [c for c in candidates if not c.rejected]
        result.hardened = active.copy()
        result.rejected.extend([c for c in candidates if c.rejected])
        log.info("📍 Stage 5 HARDEN: %d passed hardening", len(active))

        # Stage 6: Deploy
        if not dry_run:
            candidates = self.deploy(active)
            result.deployed = [c for c in candidates if not c.rejected]
            result.rejected.extend([c for c in candidates if c.rejected])
            log.info("📍 Stage 6 DEPLOY: %d skills deployed", len(result.deployed))
        else:
            result.deployed = []
            log.info("📍 Stage 6 DEPLOY: Skipped (dry_run=True)")

        result.finished_at = datetime.utcnow()
        result.duration_ms = (time.time() - start_time) * 1000

        log.info("✅ PIPELINE: %s (%.0fms)", result.summary(), result.duration_ms)

        return result

    # =========================================================================
    # Individual Stages
    # =========================================================================

    def discover(self, days: int = 7) -> List[SkillCandidate]:
        """
        Stage 1: Discover patterns in logs.

        Uses the SkillExtractor to find recurring patterns.
        """
        candidates = []

        # Get patterns from extractor
        since = datetime.utcnow() - timedelta(days=days)
        records = self.meta_logger.query(since=since, limit=5000)

        # Convert to patterns
        patterns = self.extractor.extract_patterns(records)

        for i, pattern in enumerate(patterns):
            candidate = SkillCandidate(
                id=f"skill_{int(time.time())}_{i}",
                name=pattern.suggested_name or f"pattern_{i}",
                pattern=pattern,
                current_stage=PipelineStage.DISCOVER,
                discovered_at=datetime.utcnow(),
            )
            candidates.append(candidate)

        return candidates

    def evaluate(self, candidates: List[SkillCandidate]) -> List[SkillCandidate]:
        """
        Stage 2: Evaluate causal effectiveness.

        Uses the CausalPatternMiner to check if patterns are truly causal.
        """
        for candidate in candidates:
            if candidate.rejected:
                continue

            candidate.current_stage = PipelineStage.EVALUATE

            # Extract context hash from pattern
            context_hash = candidate.pattern.context_hash if candidate.pattern else None
            tool = candidate.pattern.primary_tool if candidate.pattern else None

            if not context_hash or not tool:
                candidate.rejected = True
                candidate.rejection_reason = RejectionReason.INSUFFICIENT_DATA
                candidate.rejection_details = "Missing context or tool information"
                continue

            # Get causal estimate
            estimate = self.causal_miner.estimate_causal_score(tool, context_hash)

            if estimate is None:
                candidate.rejected = True
                candidate.rejection_reason = RejectionReason.INSUFFICIENT_DATA
                candidate.rejection_details = "Not enough observations for causal estimate"
                continue

            candidate.causal_estimate = estimate

            # Check causal threshold
            if estimate.delta < self.causal_threshold:
                if estimate.delta < 0:
                    candidate.rejected = True
                    candidate.rejection_reason = RejectionReason.NEGATIVE_EFFECT
                    candidate.rejection_details = f"Pattern has negative effect (Δ={estimate.delta:.2f})"
                else:
                    candidate.rejected = True
                    candidate.rejection_reason = RejectionReason.NO_CAUSAL_EFFECT
                    candidate.rejection_details = f"Effect too small (Δ={estimate.delta:.2f} < {self.causal_threshold})"

        return candidates

    def decide(self, candidates: List[SkillCandidate]) -> List[SkillCandidate]:
        """
        Stage 3: Decide on internalization.

        Uses the CurriculumManager to decide if worth learning.
        """
        for candidate in candidates:
            if candidate.rejected:
                continue

            candidate.current_stage = PipelineStage.DECIDE

            # Build skill proposal for curriculum
            if candidate.pattern:
                proposal = SkillProposal(
                    name=candidate.name,
                    pattern=candidate.pattern,
                    frequency=candidate.pattern.frequency,
                    success_rate=candidate.pattern.success_rate,
                    causal_effect=candidate.causal_estimate.delta if candidate.causal_estimate else 0,
                )
                candidate.proposal = proposal

                # Ask curriculum manager
                should_intern, score, reason = self.curriculum.should_internalize_detailed(proposal)

                candidate.internalize_score = score

                if not should_intern:
                    candidate.rejected = True
                    candidate.rejection_reason = RejectionReason.NOT_WORTH_INTERNALIZING
                    candidate.rejection_details = reason or "Curriculum rejected"

        return candidates

    def architect_skills(self, candidates: List[SkillCandidate]) -> List[SkillCandidate]:
        """
        Stage 4: Architect robust skills.

        Uses the Architect to generalize patterns into robust skill specs.
        """
        for candidate in candidates:
            if candidate.rejected:
                continue

            candidate.current_stage = PipelineStage.ARCHITECT

            # Build episodes from pattern
            episodes = self._pattern_to_episodes(candidate.pattern)

            if len(episodes) < 2:
                candidate.rejected = True
                candidate.rejection_reason = RejectionReason.INSUFFICIENT_DATA
                candidate.rejection_details = "Need at least 2 episodes to generalize"
                continue

            try:
                skill_spec = self.architect.generalize(episodes)
                candidate.skill_spec = skill_spec
                candidate.name = skill_spec.name  # Update name from architect

                # Check confidence
                if skill_spec.confidence < 0.3:
                    candidate.rejected = True
                    candidate.rejection_reason = RejectionReason.LOW_CONFIDENCE
                    candidate.rejection_details = f"Low generalization confidence ({skill_spec.confidence:.2f})"

            except Exception as e:
                log.warning("Failed to architect skill %s: %s", candidate.name, e)
                candidate.rejected = True
                candidate.rejection_reason = RejectionReason.INSUFFICIENT_DATA
                candidate.rejection_details = str(e)

        return candidates

    def harden(self, candidates: List[SkillCandidate]) -> List[SkillCandidate]:
        """
        Stage 5: Harden in the dojo.

        Stress-tests skill implementations.
        """
        for candidate in candidates:
            if candidate.rejected:
                continue

            candidate.current_stage = PipelineStage.HARDEN

            # Low-risk skills can skip hardening
            if not self.require_hardening and candidate.skill_spec:
                if candidate.skill_spec.classification in ("secretary", "operational"):
                    log.debug("Skipping hardening for low-risk skill %s", candidate.name)
                    continue

            # Build dojo spec
            if not candidate.skill_spec:
                continue

            dojo_spec = DojoSkillSpec(
                name=candidate.skill_spec.name,
                entrypoint=f"ara.skills.{candidate.skill_spec.name}:run",
                tags=list(candidate.skill_spec.tags.keys()),
            )

            # Get seed examples from pattern
            seed_examples = self._pattern_to_seed_examples(candidate.pattern)

            try:
                report = self.dojo.harden_skill(dojo_spec, seed_examples)
                candidate.hardening_report = report

                if report.result == HardeningResult.FAILED:
                    candidate.rejected = True
                    candidate.rejection_reason = RejectionReason.FAILED_HARDENING
                    candidate.rejection_details = (
                        f"Dojo failure: {report.success_rate:.0%} success "
                        f"(need {self.hardening_threshold:.0%})"
                    )

            except Exception as e:
                log.warning("Dojo error for %s: %s", candidate.name, e)
                # Don't reject on dojo error - skill might not have implementation yet
                candidate.hardening_report = None

        return candidates

    def deploy(self, candidates: List[SkillCandidate]) -> List[SkillCandidate]:
        """
        Stage 6: Deploy to registry.

        Makes skills available for use.
        """
        for candidate in candidates:
            if candidate.rejected:
                continue

            candidate.current_stage = PipelineStage.DEPLOY

            # Check if already exists
            existing = self.registry.get(candidate.name)
            if existing:
                candidate.rejected = True
                candidate.rejection_reason = RejectionReason.ALREADY_EXISTS
                candidate.rejection_details = f"Skill '{candidate.name}' already in registry"
                continue

            # Build LearnedSkill
            if not candidate.skill_spec:
                continue

            learned = LearnedSkill(
                name=candidate.skill_spec.name,
                description=candidate.skill_spec.description,
                category=candidate.skill_spec.category,
                tags=list(candidate.skill_spec.tags.keys()),
                confidence=candidate.skill_spec.confidence,
                implementation_code=candidate.skill_spec.example_code,
                source_episodes=candidate.skill_spec.source_episodes,
                created_at=datetime.utcnow(),
            )

            # Register
            self.registry.register(learned)
            candidate.learned_skill = learned
            candidate.deployed_at = datetime.utcnow()

            log.info("✨ Deployed skill: %s (confidence=%.2f)", learned.name, learned.confidence)

        return candidates

    # =========================================================================
    # Helpers
    # =========================================================================

    def _pattern_to_episodes(self, pattern: Optional[SessionPattern]) -> List[Episode]:
        """Convert a SessionPattern to Episodes for the Architect."""
        if not pattern:
            return []

        episodes = []
        for i, session_id in enumerate(pattern.session_ids[:10]):  # Limit to 10
            episode = Episode(
                id=f"ep_{session_id}_{i}",
                timestamp=datetime.utcnow(),
                context=pattern.context or {},
                query=pattern.sample_queries[i] if i < len(pattern.sample_queries) else "",
                intent=pattern.intent or "",
                tools_used=pattern.tools or [],
                success=pattern.success_rate > 0.5,
                tags=pattern.tags or [],
            )
            episodes.append(episode)

        return episodes

    def _pattern_to_seed_examples(self, pattern: Optional[SessionPattern]) -> List[Dict[str, Any]]:
        """Convert a SessionPattern to seed examples for the Dojo."""
        if not pattern:
            return []

        examples = []
        for i, session_id in enumerate(pattern.session_ids[:5]):
            example = {
                "config": pattern.context or {},
                "data": pattern.sample_queries[i] if i < len(pattern.sample_queries) else "",
                "env": {},
            }
            examples.append(example)

        return examples


# =============================================================================
# Convenience Functions
# =============================================================================

_default_pipeline: Optional[SkillPipeline] = None


def get_pipeline() -> SkillPipeline:
    """Get the default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SkillPipeline()
    return _default_pipeline


def run_pipeline(days: int = 7, dry_run: bool = False) -> PipelineResult:
    """
    Run the skill learning pipeline.

    Args:
        days: Days of logs to analyze
        dry_run: Don't actually deploy skills

    Returns:
        PipelineResult
    """
    return get_pipeline().run(days=days, dry_run=dry_run)


def discover_skills(days: int = 7) -> List[SkillCandidate]:
    """
    Discover potential skills from logs.

    Args:
        days: Days of logs to analyze

    Returns:
        List of skill candidates
    """
    return get_pipeline().discover(days)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'PipelineStage',
    'RejectionReason',
    'SkillCandidate',
    'PipelineResult',
    'SkillPipeline',
    'get_pipeline',
    'run_pipeline',
    'discover_skills',
]

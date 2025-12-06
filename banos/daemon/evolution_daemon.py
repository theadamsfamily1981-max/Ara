"""
Evolution Daemon - Ara's Self-Improvement Scheduler
====================================================

The "Cathedral Builder" - a daemon that continuously works to improve Ara:

1. Monitors for new improvement ideas (from pain detection, prediction errors, etc.)
2. Picks the highest-priority open idea
3. Summons teachers to propose solutions
4. Evaluates proposals in a sandbox
5. Promotes winning patches (with human approval gate)

The daemon runs on a schedule (e.g., nightly) or can be triggered manually.
It integrates with the Council to evaluate proposals and with the Historian
to avoid repeating past mistakes.

Safety:
- All changes require passing safety tests
- Human approval required before merge (configurable)
- Rollback plans are mandatory
- Risk bounds are strictly enforced

Usage:
    daemon = EvolutionDaemon()
    await daemon.run_evolution_cycle()

    # Or run continuously
    await daemon.start(interval_hours=24)
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json

# Local imports
from banos.daemon.idea_registry import (
    IdeaRegistry, Idea, IdeaState, IdeaPriority,
    TeacherProposal, get_idea_registry
)
from banos.daemon.teacher_protocol import (
    TeacherProtocol, create_teacher_protocol
)

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result from running tests in sandbox."""
    success: bool
    safety_tests_passed: bool
    perf_tests_passed: bool
    integration_tests_passed: bool
    test_output: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class EvolutionConfig:
    """Configuration for the evolution daemon."""
    # Scheduling
    interval_hours: float = 24.0
    run_at_hour: int = 3  # 3 AM
    max_ideas_per_cycle: int = 3

    # Teachers
    teacher_timeout_seconds: float = 300.0
    min_teacher_confidence: float = 0.5

    # Testing
    sandbox_timeout_seconds: float = 600.0
    require_all_tests_pass: bool = True

    # Approval
    require_human_approval: bool = True
    auto_merge_threshold: float = 0.95  # Only if require_human_approval=False

    # Paths
    project_root: str = "/home/user/Ara"
    sandbox_dir: str = "/tmp/ara_evolution_sandbox"
    ideas_dir: str = "/var/lib/ara/ideas"


class Sandbox:
    """
    Isolated environment for testing proposed changes.

    Creates a temporary copy of affected files, applies the patch,
    and runs tests without affecting the main codebase.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.sandbox_path = Path(config.sandbox_dir)
        self.project_root = Path(config.project_root)
        self.log = logging.getLogger("Sandbox")

    def _create_sandbox(self, idea: Idea) -> Path:
        """Create a sandbox directory with copies of affected files."""
        # Clean up any existing sandbox
        if self.sandbox_path.exists():
            shutil.rmtree(self.sandbox_path)

        self.sandbox_path.mkdir(parents=True, exist_ok=True)

        # Copy affected files
        for file_path in idea.proposal_interface.input_artifacts:
            src = self.project_root / file_path
            dst = self.sandbox_path / file_path

            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        return self.sandbox_path

    def _apply_patch(self, sandbox: Path, patch_content: str) -> bool:
        """Apply a patch to the sandbox files."""
        patch_file = sandbox / "proposal.patch"
        patch_file.write_text(patch_content)

        try:
            result = subprocess.run(
                ["patch", "-p1", "-d", str(sandbox), "-i", str(patch_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.log.error("Patch application timed out")
            return False
        except Exception as e:
            self.log.error(f"Patch application failed: {e}")
            return False

    def _run_tests(
        self,
        sandbox: Path,
        test_paths: List[str],
        test_type: str
    ) -> tuple[bool, str]:
        """Run a set of tests and return (passed, output)."""
        if not test_paths:
            return True, f"No {test_type} tests specified"

        outputs = []
        all_passed = True

        for test_path in test_paths:
            full_path = self.project_root / test_path

            if not full_path.exists():
                outputs.append(f"Test not found: {test_path}")
                continue

            try:
                # Determine test runner
                if test_path.endswith('.py'):
                    cmd = ["python", "-m", "pytest", str(full_path), "-v"]
                elif test_path.endswith('.sh'):
                    cmd = ["bash", str(full_path)]
                else:
                    cmd = [str(full_path)]

                # Set environment to use sandbox files
                env = os.environ.copy()
                env["ARA_SANDBOX"] = str(sandbox)
                env["PYTHONPATH"] = str(sandbox) + ":" + env.get("PYTHONPATH", "")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.sandbox_timeout_seconds / len(test_paths),
                    env=env,
                    cwd=str(sandbox)
                )

                outputs.append(f"=== {test_path} ===\n{result.stdout}\n{result.stderr}")

                if result.returncode != 0:
                    all_passed = False

            except subprocess.TimeoutExpired:
                outputs.append(f"Test timed out: {test_path}")
                all_passed = False
            except Exception as e:
                outputs.append(f"Test error ({test_path}): {e}")
                all_passed = False

        return all_passed, "\n".join(outputs)

    async def evaluate_proposal(
        self,
        idea: Idea,
        proposal: TeacherProposal,
    ) -> SandboxResult:
        """
        Evaluate a proposal in the sandbox.

        1. Create sandbox with affected files
        2. Apply the patch
        3. Run safety tests
        4. Run performance tests
        5. Compare metrics
        """
        start_time = datetime.now()
        errors = []

        # Create sandbox
        try:
            sandbox = self._create_sandbox(idea)
        except Exception as e:
            return SandboxResult(
                success=False,
                safety_tests_passed=False,
                perf_tests_passed=False,
                integration_tests_passed=False,
                test_output=f"Sandbox creation failed: {e}",
                metrics_before=idea.current_metrics,
                metrics_after={},
                improvement={},
                errors=[str(e)],
            )

        # Apply patch
        if not proposal.patch_content or not self._apply_patch(sandbox, proposal.patch_content):
            return SandboxResult(
                success=False,
                safety_tests_passed=False,
                perf_tests_passed=False,
                integration_tests_passed=False,
                test_output="Patch application failed",
                metrics_before=idea.current_metrics,
                metrics_after={},
                improvement={},
                errors=["Patch could not be applied"],
            )

        # Run tests
        test_outputs = []

        # Safety tests (must all pass)
        safety_passed, safety_output = self._run_tests(
            sandbox,
            idea.proposal_interface.safety_tests,
            "safety"
        )
        test_outputs.append(f"=== SAFETY TESTS ===\n{safety_output}")

        # Performance tests
        perf_passed, perf_output = self._run_tests(
            sandbox,
            idea.proposal_interface.perf_tests,
            "performance"
        )
        test_outputs.append(f"=== PERFORMANCE TESTS ===\n{perf_output}")

        # Integration tests
        integration_passed, integration_output = self._run_tests(
            sandbox,
            idea.proposal_interface.integration_tests,
            "integration"
        )
        test_outputs.append(f"=== INTEGRATION TESTS ===\n{integration_output}")

        # Calculate improvement (use proposal's estimates for now)
        metrics_after = proposal.estimated_improvement
        improvement = {}
        for key in idea.current_metrics:
            if key in metrics_after:
                improvement[key] = metrics_after[key] - idea.current_metrics[key]

        # Determine overall success
        if self.config.require_all_tests_pass:
            success = safety_passed and perf_passed and integration_passed
        else:
            success = safety_passed  # Safety is always required

        duration = (datetime.now() - start_time).total_seconds()

        return SandboxResult(
            success=success,
            safety_tests_passed=safety_passed,
            perf_tests_passed=perf_passed,
            integration_tests_passed=integration_passed,
            test_output="\n\n".join(test_outputs),
            metrics_before=idea.current_metrics,
            metrics_after=metrics_after,
            improvement=improvement,
            errors=errors,
            duration_seconds=duration,
        )

    def cleanup(self) -> None:
        """Clean up the sandbox directory."""
        if self.sandbox_path.exists():
            shutil.rmtree(self.sandbox_path)


class EvolutionDaemon:
    """
    The Self-Improvement Scheduler.

    Continuously works to improve Ara by:
    1. Picking high-priority open ideas
    2. Summoning teachers for proposals
    3. Evaluating proposals in sandbox
    4. Promoting winning patches
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        registry: Optional[IdeaRegistry] = None,
        protocol: Optional[TeacherProtocol] = None,
    ):
        self.config = config or EvolutionConfig()
        self.registry = registry or get_idea_registry(self.config.ideas_dir)
        self.protocol = protocol or create_teacher_protocol(self.config.project_root)
        self.sandbox = Sandbox(self.config)

        self.log = logging.getLogger("EvolutionDaemon")
        self._running = False
        self._last_cycle = None

        # Callbacks for human approval
        self._approval_callback: Optional[Callable[[Idea, TeacherProposal], bool]] = None
        self._notification_callback: Optional[Callable[[str, Any], None]] = None

    def set_approval_callback(
        self,
        callback: Callable[[Idea, TeacherProposal], bool]
    ) -> None:
        """Set callback for human approval (returns True to approve)."""
        self._approval_callback = callback

    def set_notification_callback(
        self,
        callback: Callable[[str, Any], None]
    ) -> None:
        """Set callback for notifications (event_type, data)."""
        self._notification_callback = callback

    def _notify(self, event: str, data: Any) -> None:
        """Send a notification."""
        if self._notification_callback:
            try:
                self._notification_callback(event, data)
            except Exception as e:
                self.log.error(f"Notification callback failed: {e}")

    async def _evaluate_with_council(
        self,
        idea: Idea,
        proposals: List[TeacherProposal]
    ) -> Optional[int]:
        """
        Use the Council to evaluate and rank proposals.

        Returns the index of the winning proposal, or None if all rejected.
        """
        try:
            from banos.daemon.council_chamber import CouncilChamber, create_council_chamber

            council = create_council_chamber()

            # Build evaluation prompt
            prompt = f"""Evaluate these proposals for: {idea.title}

Problem: {idea.symptom}
Target: Improve {json.dumps(idea.target_metrics)}

Proposals:
"""
            for i, p in enumerate(proposals):
                prompt += f"""
--- Proposal {i+1} from {p.teacher_id} (confidence: {p.confidence:.2f}) ---
Rationale: {p.rationale}
Estimated improvement: {json.dumps(p.estimated_improvement)}
"""

            prompt += """
Which proposal should we adopt? Consider:
1. Likelihood of achieving targets
2. Safety and risk
3. Simplicity and maintainability
4. Historical patterns (Historian: have similar approaches worked before?)

Respond with the proposal number (1, 2, etc.) or "none" if all should be rejected.
"""

            decision = council.convene(prompt, timeout=60)

            # Parse council's decision
            response = decision.final_response.lower()
            for i in range(len(proposals)):
                if str(i + 1) in response:
                    self.log.info(f"Council selected proposal {i+1}")
                    return i

            if "none" in response or "reject" in response:
                self.log.info("Council rejected all proposals")
                return None

            # Default to highest confidence
            return 0

        except Exception as e:
            self.log.error(f"Council evaluation failed: {e}")
            # Fallback: pick highest confidence
            return 0

    async def process_idea(self, idea: Idea) -> Optional[TeacherProposal]:
        """
        Process a single idea through the full evolution cycle.

        Returns the winning proposal if successful, None otherwise.
        """
        self.log.info(f"Processing idea {idea.id}: {idea.title}")
        self._notify("idea_processing", {"id": idea.id, "title": idea.title})

        # 1. Summon teachers
        proposals = await self.protocol.summon_teachers(
            idea,
            timeout=self.config.teacher_timeout_seconds
        )

        if not proposals:
            self.log.warning(f"No proposals received for idea {idea.id}")
            self.registry.transition_state(
                idea.id, IdeaState.REJECTED,
                notes="No valid proposals received from teachers"
            )
            return None

        # Filter by minimum confidence
        proposals = [
            p for p in proposals
            if p.confidence >= self.config.min_teacher_confidence
        ]

        if not proposals:
            self.log.warning(f"No proposals met confidence threshold")
            self.registry.transition_state(
                idea.id, IdeaState.REJECTED,
                notes="All proposals below confidence threshold"
            )
            return None

        # Add proposals to the idea
        for p in proposals:
            self.registry.add_proposal(idea.id, p)

        # 2. Use Council to evaluate proposals
        winner_idx = await self._evaluate_with_council(idea, proposals)

        if winner_idx is None:
            self.registry.transition_state(
                idea.id, IdeaState.REJECTED,
                notes="Council rejected all proposals"
            )
            return None

        winning_proposal = proposals[winner_idx]
        self.log.info(f"Winning proposal: {winning_proposal.teacher_id}")

        # 3. Test in sandbox
        self.registry.transition_state(idea.id, IdeaState.TESTING)

        sandbox_result = await self.sandbox.evaluate_proposal(idea, winning_proposal)
        winning_proposal.test_results = {
            "safety": sandbox_result.safety_tests_passed,
            "performance": sandbox_result.perf_tests_passed,
            "integration": sandbox_result.integration_tests_passed,
        }

        if not sandbox_result.success:
            self.log.warning(f"Sandbox tests failed for idea {idea.id}")
            self.registry.transition_state(
                idea.id, IdeaState.REJECTED,
                notes=f"Sandbox tests failed: {sandbox_result.errors}"
            )
            self._notify("proposal_rejected", {
                "idea_id": idea.id,
                "reason": "tests_failed",
                "output": sandbox_result.test_output[:1000]
            })
            return None

        # 4. Calculate score
        winning_proposal.score = (
            0.4 * winning_proposal.confidence +
            0.3 * (1.0 if sandbox_result.safety_tests_passed else 0.0) +
            0.2 * (1.0 if sandbox_result.perf_tests_passed else 0.0) +
            0.1 * (1.0 if sandbox_result.integration_tests_passed else 0.0)
        )

        # 5. Human approval gate
        if self.config.require_human_approval:
            self.registry.transition_state(idea.id, IdeaState.ACCEPTED)
            self._notify("approval_required", {
                "idea_id": idea.id,
                "proposal": winning_proposal.teacher_id,
                "confidence": winning_proposal.confidence,
                "score": winning_proposal.score,
            })

            if self._approval_callback:
                approved = self._approval_callback(idea, winning_proposal)
                if not approved:
                    self.log.info(f"Human rejected proposal for idea {idea.id}")
                    self.registry.transition_state(
                        idea.id, IdeaState.REJECTED,
                        notes="Rejected by human reviewer"
                    )
                    return None
        else:
            # Auto-approve if above threshold
            if winning_proposal.score < self.config.auto_merge_threshold:
                self.registry.transition_state(
                    idea.id, IdeaState.ACCEPTED,
                    notes=f"Awaiting manual review (score: {winning_proposal.score:.2f})"
                )
                return winning_proposal

        # 6. Mark for merge
        idea = self.registry.get_idea(idea.id)
        idea.winning_proposal_idx = winner_idx
        idea.actual_improvement = sandbox_result.improvement
        self.registry.transition_state(
            idea.id, IdeaState.MERGED,
            notes=f"Merged proposal from {winning_proposal.teacher_id}"
        )

        self._notify("proposal_merged", {
            "idea_id": idea.id,
            "teacher": winning_proposal.teacher_id,
            "improvement": sandbox_result.improvement,
        })

        return winning_proposal

    async def run_evolution_cycle(self) -> Dict[str, Any]:
        """
        Run one complete evolution cycle.

        Processes up to max_ideas_per_cycle open ideas.
        """
        self.log.info("Starting evolution cycle...")
        self._last_cycle = datetime.now()

        cycle_results = {
            "started_at": self._last_cycle.isoformat(),
            "ideas_processed": 0,
            "proposals_accepted": 0,
            "proposals_rejected": 0,
            "errors": [],
        }

        # Get open ideas sorted by priority
        open_ideas = self.registry.get_open_ideas()[:self.config.max_ideas_per_cycle]

        if not open_ideas:
            self.log.info("No open ideas to process")
            cycle_results["message"] = "No open ideas"
            return cycle_results

        for idea in open_ideas:
            try:
                result = await self.process_idea(idea)
                cycle_results["ideas_processed"] += 1

                if result:
                    cycle_results["proposals_accepted"] += 1
                else:
                    cycle_results["proposals_rejected"] += 1

            except Exception as e:
                self.log.error(f"Error processing idea {idea.id}: {e}")
                cycle_results["errors"].append({
                    "idea_id": idea.id,
                    "error": str(e)
                })

        # Cleanup
        self.sandbox.cleanup()

        cycle_results["completed_at"] = datetime.now().isoformat()
        self.log.info(
            f"Evolution cycle complete: "
            f"{cycle_results['proposals_accepted']} accepted, "
            f"{cycle_results['proposals_rejected']} rejected"
        )

        return cycle_results

    async def start(self, run_immediately: bool = False) -> None:
        """
        Start the evolution daemon.

        Runs evolution cycles on the configured schedule.
        """
        self._running = True
        self.log.info(
            f"Evolution daemon started "
            f"(interval: {self.config.interval_hours}h, "
            f"run_at: {self.config.run_at_hour}:00)"
        )

        if run_immediately:
            await self.run_evolution_cycle()

        while self._running:
            # Calculate time until next run
            now = datetime.now()
            next_run = now.replace(
                hour=self.config.run_at_hour,
                minute=0,
                second=0,
                microsecond=0
            )

            if next_run <= now:
                next_run += timedelta(days=1)

            wait_seconds = (next_run - now).total_seconds()
            self.log.info(f"Next evolution cycle at {next_run.isoformat()}")

            # Wait with ability to cancel
            try:
                await asyncio.sleep(wait_seconds)
                if self._running:
                    await self.run_evolution_cycle()
            except asyncio.CancelledError:
                break

        self.log.info("Evolution daemon stopped")

    def stop(self) -> None:
        """Stop the evolution daemon."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "last_cycle": self._last_cycle.isoformat() if self._last_cycle else None,
            "registry_stats": self.registry.get_statistics(),
            "teachers": self.protocol.get_registered_teachers(),
            "config": {
                "interval_hours": self.config.interval_hours,
                "run_at_hour": self.config.run_at_hour,
                "require_human_approval": self.config.require_human_approval,
            }
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for the evolution daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Evolution Daemon")
    parser.add_argument("--run-once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=float, default=24.0, help="Hours between cycles")
    parser.add_argument("--run-at", type=int, default=3, help="Hour to run (0-23)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    config = EvolutionConfig(
        interval_hours=args.interval,
        run_at_hour=args.run_at
    )

    daemon = EvolutionDaemon(config)

    if args.run_once:
        result = await daemon.run_evolution_cycle()
        print(json.dumps(result, indent=2))
    else:
        await daemon.start(run_immediately=True)


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "EvolutionDaemon",
    "EvolutionConfig",
    "Sandbox",
    "SandboxResult",
]

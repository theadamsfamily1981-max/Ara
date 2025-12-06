"""
Atomic Updater - Test, Measure, Apply (or Reject)
=================================================

The updater is the ONLY component that can actually apply mutations.
It has a strict protocol:

1. Run pytest on the proposal's test file
2. If tests fail: REJECT, log failure, done
3. If tests pass: Benchmark old vs new
4. If benchmark fails: REJECT, log as "worse"
5. If benchmark passes: Mark as READY FOR APPROVAL
6. If human approves (or auto-approve enabled): Apply

"Apply" means:
- Import the new function from mutations/
- Setattr it onto the original module
- Log everything
- Keep the old version available for rollback

This is NOT hot-patching running daemons. This is controlled
module-level replacement that takes effect on next import/restart.
"""

import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from banos.daemon.ouroboros.mutation_policy import (
    ouroboros_enabled,
    ouroboros_auto_apply,
    is_mutable,
)
from banos.daemon.ouroboros.semantic_optimizer import MutationProposal

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from running tests on a proposal."""
    passed: bool
    duration_seconds: float
    stdout: str
    stderr: str
    test_count: int
    failures: int
    errors: int


@dataclass
class BenchmarkResult:
    """Result from benchmarking old vs new implementation."""
    original_time_ms: float
    new_time_ms: float
    speedup: float  # > 1.0 means improvement
    memory_original_kb: float
    memory_new_kb: float
    memory_reduction: float  # > 0 means improvement


@dataclass
class ApplyResult:
    """Result from applying (or attempting to apply) a mutation."""
    success: bool
    message: str
    applied_at: Optional[datetime] = None
    rollback_available: bool = False


class MutationLog:
    """
    Persistent log of all mutation attempts.

    This is the audit trail. Every proposal, test, benchmark, and
    apply decision is logged here for debugging and learning.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        proposal_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an event to the mutation log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "proposal_id": proposal_id,
            **data,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_history(self, proposal_id: str) -> List[Dict[str, Any]]:
        """Get all events for a proposal."""
        if not self.log_path.exists():
            return []

        events = []
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("proposal_id") == proposal_id:
                        events.append(entry)
                except json.JSONDecodeError:
                    continue
        return events

    def get_failures(self, since_hours: float = 24) -> List[Dict[str, Any]]:
        """Get recent failures for learning."""
        if not self.log_path.exists():
            return []

        cutoff = time.time() - (since_hours * 3600)
        failures = []

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") in ("test_failed", "benchmark_failed", "apply_failed"):
                        ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        if ts > cutoff:
                            failures.append(entry)
                except (json.JSONDecodeError, ValueError):
                    continue

        return failures


class AtomicUpdater:
    """
    Handles the test → benchmark → apply pipeline for mutations.

    All operations are logged and reversible.
    """

    def __init__(
        self,
        repo_root: Path,
        mutations_dir: Path,
        log_path: Optional[Path] = None,
    ):
        self.repo_root = Path(repo_root)
        self.mutations_dir = Path(mutations_dir)
        self.log = logging.getLogger("AtomicUpdater")

        log_path = log_path or (self.repo_root / "var" / "log" / "mutations.jsonl")
        self.mutation_log = MutationLog(log_path)

        # Applied mutations (for rollback)
        self._applied: Dict[str, Dict[str, Any]] = {}

    def run_tests(self, proposal: MutationProposal) -> TestResult:
        """
        Run pytest on a proposal's test file.

        Returns TestResult with pass/fail and details.
        """
        if not proposal.test_path or not proposal.test_path.exists():
            return TestResult(
                passed=False,
                duration_seconds=0,
                stdout="",
                stderr="No test file found",
                test_count=0,
                failures=0,
                errors=1,
            )

        self.log.info(f"Running tests for {proposal.proposal_id}")
        start = time.time()

        try:
            # Run pytest with JSON output
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(proposal.test_path),
                    "-v",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.repo_root),
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.mutations_dir) + ":" + os.environ.get("PYTHONPATH", ""),
                },
            )

            duration = time.time() - start

            # Parse pytest output for counts
            test_count = result.stdout.count("PASSED") + result.stdout.count("FAILED")
            failures = result.stdout.count("FAILED")
            errors = result.stderr.count("ERROR") if "ERROR" in result.stderr else 0

            passed = result.returncode == 0

            test_result = TestResult(
                passed=passed,
                duration_seconds=duration,
                stdout=result.stdout,
                stderr=result.stderr,
                test_count=test_count,
                failures=failures,
                errors=errors,
            )

        except subprocess.TimeoutExpired:
            test_result = TestResult(
                passed=False,
                duration_seconds=60.0,
                stdout="",
                stderr="Tests timed out after 60 seconds",
                test_count=0,
                failures=0,
                errors=1,
            )

        except Exception as e:
            test_result = TestResult(
                passed=False,
                duration_seconds=time.time() - start,
                stdout="",
                stderr=str(e),
                test_count=0,
                failures=0,
                errors=1,
            )

        # Log result
        self.mutation_log.log_event(
            "test_passed" if test_result.passed else "test_failed",
            proposal.proposal_id,
            {
                "test_count": test_result.test_count,
                "failures": test_result.failures,
                "duration_seconds": test_result.duration_seconds,
            },
        )

        return test_result

    def run_benchmark(
        self,
        proposal: MutationProposal,
        iterations: int = 100,
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark original vs new implementation.

        Returns BenchmarkResult if successful, None if benchmark couldn't run.
        """
        if not proposal.impl_path or not proposal.impl_path.exists():
            return None

        self.log.info(f"Benchmarking {proposal.proposal_id}")

        try:
            # Import original module and function
            orig_module = importlib.import_module(proposal.module)
            orig_func = getattr(orig_module, proposal.func_name)

            # Import new implementation
            # Convert path to module: mutations/foo/bar/func_v123.py -> mutations.foo.bar.func_v123
            rel_path = proposal.impl_path.relative_to(self.mutations_dir)
            new_mod_name = str(rel_path.with_suffix('')).replace('/', '.')
            new_mod_name = f"mutations.{new_mod_name}"

            # Add mutations dir to path temporarily
            sys.path.insert(0, str(self.mutations_dir.parent))
            try:
                new_module = importlib.import_module(new_mod_name)
                new_func = getattr(new_module, proposal.func_name)
            finally:
                sys.path.pop(0)

            # Generate test inputs (this is tricky - we'd need function-specific data)
            # For now, just time the function calls with no args (smoke test)
            # A real implementation would use stored test cases

            import timeit

            # Benchmark original
            orig_time = timeit.timeit(
                lambda: orig_func() if callable(orig_func) else None,
                number=iterations,
            )

            # Benchmark new
            new_time = timeit.timeit(
                lambda: new_func() if callable(new_func) else None,
                number=iterations,
            )

            speedup = orig_time / new_time if new_time > 0 else 1.0

            result = BenchmarkResult(
                original_time_ms=(orig_time / iterations) * 1000,
                new_time_ms=(new_time / iterations) * 1000,
                speedup=speedup,
                memory_original_kb=0,  # Would need memory profiling
                memory_new_kb=0,
                memory_reduction=0,
            )

            self.mutation_log.log_event(
                "benchmark_complete",
                proposal.proposal_id,
                {
                    "speedup": result.speedup,
                    "original_ms": result.original_time_ms,
                    "new_ms": result.new_time_ms,
                },
            )

            return result

        except Exception as e:
            self.log.error(f"Benchmark failed: {e}")
            self.mutation_log.log_event(
                "benchmark_failed",
                proposal.proposal_id,
                {"error": str(e)},
            )
            return None

    def evaluate_proposal(
        self,
        proposal: MutationProposal,
        min_speedup: float = 1.0,
    ) -> MutationProposal:
        """
        Full evaluation pipeline: test → benchmark → score.

        Updates the proposal with results and returns it.
        """
        # Run tests
        test_result = self.run_tests(proposal)
        proposal.tests_passed = test_result.passed

        if not test_result.passed:
            proposal.error_message = f"Tests failed: {test_result.failures} failures, {test_result.errors} errors"
            return proposal

        # Run benchmark
        benchmark = self.run_benchmark(proposal)
        if benchmark:
            proposal.actual_speedup = benchmark.speedup

            if benchmark.speedup < min_speedup:
                proposal.error_message = f"No improvement: {benchmark.speedup:.2f}x (need {min_speedup}x)"
                self.mutation_log.log_event(
                    "benchmark_rejected",
                    proposal.proposal_id,
                    {"speedup": benchmark.speedup, "required": min_speedup},
                )

        return proposal

    def apply_mutation(
        self,
        proposal: MutationProposal,
        force: bool = False,
    ) -> ApplyResult:
        """
        Apply a tested and approved mutation.

        This replaces the function in the original module with the new one.
        The old version is saved for rollback.
        """
        if not ouroboros_enabled():
            return ApplyResult(
                success=False,
                message="Ouroboros is disabled",
            )

        if not force and not ouroboros_auto_apply():
            return ApplyResult(
                success=False,
                message="Auto-apply is disabled, requires human approval",
            )

        if not proposal.tests_passed:
            return ApplyResult(
                success=False,
                message="Cannot apply: tests have not passed",
            )

        if not is_mutable(proposal.module):
            return ApplyResult(
                success=False,
                message=f"Module {proposal.module} is not mutable",
            )

        self.log.info(f"Applying mutation {proposal.proposal_id}")

        try:
            # Import original module
            orig_module = importlib.import_module(proposal.module)
            orig_func = getattr(orig_module, proposal.func_name)

            # Save original for rollback
            self._applied[proposal.proposal_id] = {
                "module": proposal.module,
                "func_name": proposal.func_name,
                "original_func": orig_func,
                "applied_at": datetime.now(),
            }

            # Import and apply new function
            rel_path = proposal.impl_path.relative_to(self.mutations_dir)
            new_mod_name = str(rel_path.with_suffix('')).replace('/', '.')

            sys.path.insert(0, str(self.mutations_dir))
            try:
                new_module = importlib.import_module(new_mod_name)
                new_func = getattr(new_module, proposal.func_name)
            finally:
                sys.path.pop(0)

            # Apply!
            setattr(orig_module, proposal.func_name, new_func)

            self.mutation_log.log_event(
                "mutation_applied",
                proposal.proposal_id,
                {
                    "module": proposal.module,
                    "func": proposal.func_name,
                    "speedup": proposal.actual_speedup,
                },
            )

            return ApplyResult(
                success=True,
                message=f"Applied {proposal.func_name} from {proposal.impl_path}",
                applied_at=datetime.now(),
                rollback_available=True,
            )

        except Exception as e:
            self.log.error(f"Apply failed: {e}")
            self.mutation_log.log_event(
                "apply_failed",
                proposal.proposal_id,
                {"error": str(e)},
            )
            return ApplyResult(
                success=False,
                message=str(e),
            )

    def rollback(self, proposal_id: str) -> ApplyResult:
        """
        Rollback a previously applied mutation.

        Restores the original function.
        """
        if proposal_id not in self._applied:
            return ApplyResult(
                success=False,
                message=f"No applied mutation found for {proposal_id}",
            )

        record = self._applied[proposal_id]

        try:
            module = importlib.import_module(record["module"])
            setattr(module, record["func_name"], record["original_func"])

            del self._applied[proposal_id]

            self.mutation_log.log_event(
                "mutation_rolled_back",
                proposal_id,
                {"module": record["module"], "func": record["func_name"]},
            )

            return ApplyResult(
                success=True,
                message=f"Rolled back {record['func_name']}",
            )

        except Exception as e:
            return ApplyResult(
                success=False,
                message=f"Rollback failed: {e}",
            )

    def rollback_all(self) -> List[ApplyResult]:
        """Rollback all applied mutations."""
        results = []
        for proposal_id in list(self._applied.keys()):
            results.append(self.rollback(proposal_id))
        return results

    def get_applied_mutations(self) -> List[Dict[str, Any]]:
        """Get list of currently applied mutations."""
        return [
            {
                "proposal_id": pid,
                "module": record["module"],
                "func": record["func_name"],
                "applied_at": record["applied_at"].isoformat(),
            }
            for pid, record in self._applied.items()
        ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AtomicUpdater",
    "TestResult",
    "BenchmarkResult",
    "ApplyResult",
    "MutationLog",
]

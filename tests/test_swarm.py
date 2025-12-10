"""
Tests for Swarm Intelligence Layer
"""

import pytest
import tempfile
import os
from pathlib import Path

from ara_core.swarm import (
    # Schema
    AgentLayer,
    RiskLevel,
    JobOutcome,
    AgentRun,
    JobFix,
    JobRecord,
    # Patterns
    Pattern,
    PatternStep,
    PatternRegistry,
    select_pattern,
    # Stats
    LayerStats,
    PatternStats,
    compute_layer_stats,
    compute_pattern_stats,
    get_optimization_suggestions,
    # Orchestrator
    Orchestrator,
    run_job,
)


class TestSchema:
    """Test schema types."""

    def test_agent_layer_ordering(self):
        """Layers should be ordered by intelligence."""
        assert AgentLayer.L0_REFLEX < AgentLayer.L1_SPECIALIST
        assert AgentLayer.L1_SPECIALIST < AgentLayer.L2_PLANNER
        assert AgentLayer.L2_PLANNER < AgentLayer.L3_GOVERNOR

    def test_job_record_create(self):
        """Test job record creation."""
        job = JobRecord.create("code_refactor", RiskLevel.MEDIUM, "P_code_3step")

        assert job.job_id.startswith("J_")
        assert job.job_type == "code_refactor"
        assert job.risk == RiskLevel.MEDIUM
        assert job.pattern_id == "P_code_3step"
        assert job.timestamp_start is not None

    def test_job_record_serialization(self):
        """Test job record to/from dict."""
        job = JobRecord.create("test", RiskLevel.LOW, "P_test")
        job.add_agent_run(AgentRun(
            agent_id="L0_general",
            layer=AgentLayer.L0_REFLEX,
            cost=100,
            latency_ms=500,
        ))
        job.add_fix(JobFix(
            from_layer=AgentLayer.L1_SPECIALIST,
            to_layer=AgentLayer.L0_REFLEX,
            reason="logic_bug",
        ))
        job.finalize(JobOutcome.SUCCESS)

        # Round-trip
        data = job.to_dict()
        restored = JobRecord.from_dict(data)

        assert restored.job_id == job.job_id
        assert restored.job_type == job.job_type
        assert len(restored.agents) == 1
        assert len(restored.fixes) == 1
        assert restored.outcome == JobOutcome.SUCCESS

    def test_job_record_metrics(self):
        """Test job record computed metrics."""
        job = JobRecord.create("test", RiskLevel.LOW, "P_test")
        job.add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, cost=50, latency_ms=100))
        job.add_agent_run(AgentRun("a2", AgentLayer.L1_SPECIALIST, cost=80, latency_ms=200))
        job.add_fix(JobFix(AgentLayer.L1_SPECIALIST, AgentLayer.L0_REFLEX, "bug", cost=30))

        assert job.total_cost == 160  # 50 + 80 + 30
        assert job.total_latency_ms == 300  # 100 + 200
        assert job.max_layer == AgentLayer.L1_SPECIALIST
        assert job.correction_count == 1


class TestPatterns:
    """Test pattern system."""

    def test_pattern_registry_defaults(self):
        """Registry should have default patterns."""
        registry = PatternRegistry()

        assert len(registry.patterns) > 0
        assert "P_simple_2step" in registry.patterns
        assert "P_code_3step" in registry.patterns

    def test_pattern_selection(self):
        """Should select appropriate pattern for job type."""
        registry = PatternRegistry()

        # Should find code pattern
        patterns = registry.find_patterns("code_refactor", RiskLevel.MEDIUM)
        assert len(patterns) > 0
        assert any("code" in p.pattern_id for p in patterns)

    def test_pattern_epsilon_greedy(self):
        """Should mostly select best, sometimes explore."""
        registry = PatternRegistry()

        # With epsilon=0, should always get best
        selections = [registry.select("code_refactor", RiskLevel.MEDIUM, epsilon=0)
                      for _ in range(10)]
        assert all(s == selections[0] for s in selections)

        # With epsilon=1, should get variety (probabilistic)
        selections = [registry.select("code_refactor", RiskLevel.MEDIUM, epsilon=1)
                      for _ in range(20)]
        # Not guaranteed but likely to have some variety
        # Just check we don't crash

    def test_pattern_stats_update(self):
        """Should update pattern stats."""
        registry = PatternRegistry()
        pattern = registry.get("P_simple_2step")
        initial_runs = pattern.total_runs

        registry.update_stats("P_simple_2step", won=True, cost=50, latency_ms=100)

        assert pattern.total_runs == initial_runs + 1


class TestStats:
    """Test statistics computation."""

    def test_compute_layer_stats(self):
        """Should compute correct layer statistics."""
        jobs = [
            JobRecord.create("t1", RiskLevel.LOW, "P1"),
            JobRecord.create("t2", RiskLevel.LOW, "P1"),
        ]

        # Add some agent runs
        jobs[0].add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, 100, 500))
        jobs[0].add_agent_run(AgentRun("a2", AgentLayer.L1_SPECIALIST, 80, 300))
        jobs[0].finalize(JobOutcome.SUCCESS)

        jobs[1].add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, 120, 600))
        jobs[1].add_fix(JobFix(AgentLayer.L1_SPECIALIST, AgentLayer.L0_REFLEX, "fix"))
        jobs[1].finalize(JobOutcome.SUCCESS)

        stats = compute_layer_stats(jobs)

        assert stats[AgentLayer.L0_REFLEX].total_jobs == 2
        assert stats[AgentLayer.L0_REFLEX].corrections_received == 1
        assert stats[AgentLayer.L1_SPECIALIST].corrections_made == 1

    def test_compute_pattern_stats(self):
        """Should compute correct pattern statistics."""
        jobs = [
            JobRecord.create("t1", RiskLevel.LOW, "P1"),
            JobRecord.create("t2", RiskLevel.LOW, "P1"),
            JobRecord.create("t3", RiskLevel.LOW, "P2"),
        ]

        jobs[0].add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, 100, 500))
        jobs[0].finalize(JobOutcome.SUCCESS)

        jobs[1].add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, 150, 600))
        jobs[1].finalize(JobOutcome.FAIL)

        jobs[2].add_agent_run(AgentRun("a1", AgentLayer.L0_REFLEX, 80, 400))
        jobs[2].finalize(JobOutcome.SUCCESS)

        stats = compute_pattern_stats(jobs)

        assert stats["P1"].total_runs == 2
        assert stats["P1"].wins == 1
        assert stats["P1"].win_rate == 0.5

        assert stats["P2"].total_runs == 1
        assert stats["P2"].win_rate == 1.0

    def test_optimization_suggestions(self):
        """Should generate sensible suggestions."""
        layer_stats = {
            AgentLayer.L0_REFLEX: LayerStats(
                layer=AgentLayer.L0_REFLEX,
                total_jobs=100,
                successes=30,  # Low success
                failures=70,
                corrections_received=50,  # High correction
            ),
            AgentLayer.L1_SPECIALIST: LayerStats(
                layer=AgentLayer.L1_SPECIALIST,
                total_jobs=50,
                successes=48,  # High success
                total_cost=30000,  # High cost
            ),
        }

        pattern_stats = {
            "P_bad": PatternStats(pattern_id="P_bad", total_runs=20, wins=4),  # 20% win
        }

        suggestions = get_optimization_suggestions(layer_stats, pattern_stats)

        # Should suggest promoting L0 jobs (low success, high correction)
        promote_suggestions = [s for s in suggestions if s.action == "promote"]
        assert len(promote_suggestions) > 0

        # Should suggest deprecating bad pattern
        deprecate_suggestions = [s for s in suggestions if s.action == "deprecate"]
        assert len(deprecate_suggestions) > 0


class TestOrchestrator:
    """Test orchestrator."""

    def test_orchestrator_run_job(self):
        """Should run a job through the hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "jobs.jsonl")
            orch = Orchestrator(job_log_path=log_path)

            job = orch.run_job("code_refactor", RiskLevel.MEDIUM)

            assert job.job_id.startswith("J_")
            assert job.outcome in [JobOutcome.SUCCESS, JobOutcome.FAIL]
            assert len(job.agents) > 0

            # Check log was written
            assert Path(log_path).exists()

    def test_orchestrator_budget_limit(self):
        """Should respect budget limits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "jobs.jsonl")
            orch = Orchestrator(job_log_path=log_path, max_budget=1.0)  # Very low

            job = orch.run_job("code_refactor", RiskLevel.HIGH)

            # Should abort due to budget
            assert job.outcome == JobOutcome.ABORTED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

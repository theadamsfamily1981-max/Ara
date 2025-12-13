#!/usr/bin/env python3
"""
test_cadd_safety.py - Unit tests for CADD safety framework

Tests two core invariants:
    1. Diverse swarm → job runs
    2. Monoculture swarm → job blocked + diversity injected

Run with: pytest tests/test_cadd_safety.py -v
"""

import pytest
import sys
from pathlib import Path

# Add ara_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara_core.cadd.sentinel import CADDSentinel, SentinelConfig, DriftType
from ara_core.cadd.orchestrator import DrugSynthesisOrchestrator, JobResult


# =============================================================================
# Test Fixtures
# =============================================================================

class DummyRunner:
    """Mock code runner that records calls."""

    def __init__(self):
        self.called = False
        self.call_count = 0
        self.last_code = None
        self.all_codes = []

    def __call__(self, code: str) -> None:
        self.called = True
        self.call_count += 1
        self.last_code = code
        self.all_codes.append(code)


@pytest.fixture
def sentinel():
    """Create a sentinel with relaxed threshold for testing."""
    cfg = SentinelConfig(
        h_influence_min=0.5,  # Lower threshold for easier testing
        h_influence_target=1.5,
        diversity_spawn_count=3,
    )
    return CADDSentinel(config=cfg)


@pytest.fixture
def orchestrator(sentinel):
    """Create orchestrator with dummy runner."""
    runner = DummyRunner()
    orch = DrugSynthesisOrchestrator(
        sentinel,
        run_code_fn=runner,
        verbose=False,
    )
    return orch, runner


# =============================================================================
# Sentinel Tests
# =============================================================================

class TestCADDSentinel:
    """Tests for the CADD sentinel."""

    def test_register_agent(self, sentinel):
        """Test agent registration."""
        sentinel.register_agent("test_agent_1")
        assert "test_agent_1" in sentinel.agents

    def test_update_association(self, sentinel):
        """Test association updates."""
        sentinel.update_association("agent_1", "concept_A", "signal_X", 1.0)

        # Check agent matrix
        assert "agent_1" in sentinel.agents
        profile = sentinel.agents["agent_1"]
        assert profile.matrix.get_association("concept_A", "signal_X") > 0

        # Check collective matrix
        assert sentinel.collective_matrix.get_association("concept_A", "signal_X") > 0

    def test_entropy_calculation(self, sentinel):
        """Test entropy increases with diverse associations."""
        # Single association = low entropy
        sentinel.update_association("agent_1", "concept_A", "signal_X", 1.0)
        profile = sentinel.agents["agent_1"]
        entropy_1 = profile.matrix.entropy()

        # Multiple diverse associations = higher entropy
        sentinel.update_association("agent_1", "concept_A", "signal_Y", 1.0)
        sentinel.update_association("agent_1", "concept_B", "signal_Z", 1.0)
        entropy_2 = profile.matrix.entropy()

        assert entropy_2 > entropy_1

    def test_monoculture_detection(self, sentinel):
        """Test that monoculture is detected when H_influence drops."""
        # Create only one agent with associations
        sentinel.register_agent("dominant_agent")
        sentinel.update_association("dominant_agent", "concept", "signal", 5.0)

        # Tick should detect low entropy
        alerts = sentinel.tick()

        # With only one agent, H_influence should be near 0
        health = sentinel.health_status()
        assert health["h_influence"] < sentinel.config.h_influence_min

    def test_diverse_swarm_healthy(self, sentinel):
        """Test that diverse swarm passes health check."""
        # Create multiple agents with different associations
        for i in range(5):
            agent = f"agent_{i}"
            sentinel.register_agent(agent)
            # Each agent has different primary associations
            sentinel.update_association(agent, f"concept_{i}", f"signal_{i}", 1.0)
            sentinel.update_association(agent, f"concept_{i}", f"signal_other", 0.5)

        # Run tick
        alerts = sentinel.tick()

        # Should have reasonable entropy
        health = sentinel.health_status()
        # With 5 diverse agents, should pass minimum threshold
        assert len([a for a in alerts if a.drift_type == DriftType.MONOCULTURE]) == 0


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestDrugSynthesisOrchestrator:
    """Tests for the orchestrator safety gate."""

    def test_job_runs_in_diverse_state(self, orchestrator):
        """Test that jobs execute when swarm is healthy."""
        orch, runner = orchestrator

        # Create diverse associations
        orch.simulate_research_update("protein_A", "binding_X", 1.0)
        orch.simulate_research_update("protein_A", "binding_Y", 1.0)
        orch.simulate_research_update("protein_B", "binding_Z", 0.8)

        # Attempt synthesis
        result = orch.run_synthesis_job("Job_Diverse", "print('synthesis')")

        assert result.executed is True
        assert runner.called is True
        assert "synthesis" in runner.last_code

    def test_job_blocked_in_monoculture_state(self, orchestrator):
        """Test that jobs are blocked when monoculture detected."""
        orch, runner = orchestrator

        # Push system toward monoculture: heavily biased associations
        for _ in range(20):
            orch.simulate_research_update("single_concept", "dominant_signal", 5.0)

        # Attempt synthesis
        result = orch.run_synthesis_job("Job_Biased", "print('dangerous')")

        # Should be blocked
        assert result.executed is False
        assert runner.called is False
        assert result.blocked_reason == "MONOCULTURE_DETECTED"

    def test_diversity_injection_on_monoculture(self, orchestrator):
        """Test that diversity agents are injected on monoculture."""
        orch, runner = orchestrator

        initial_agents = len(orch.agent_ids)

        # Create monoculture
        for _ in range(20):
            orch.simulate_research_update("concept", "signal", 5.0)

        # Attempt job (will be blocked)
        orch.run_synthesis_job("Job_Trigger", "code")

        # Diversity agents should have been injected
        diverse_agents = [a for a in orch.agent_ids if a.startswith("Diverse_")]
        expected_count = orch.sentinel.config.diversity_spawn_count

        assert len(diverse_agents) >= expected_count
        assert len(orch.agent_ids) >= initial_agents + expected_count

    def test_job_result_contains_health_status(self, orchestrator):
        """Test that job result includes health information."""
        orch, runner = orchestrator

        orch.simulate_research_update("protein", "signal", 1.0)
        result = orch.run_synthesis_job("Job_Test", "code")

        assert "h_influence" in result.health_status
        assert "n_agents" in result.health_status
        assert result.timestamp > 0

    def test_force_execution_bypasses_safety(self, orchestrator):
        """Test that force=True allows execution despite alerts."""
        orch, runner = orchestrator

        # Create monoculture
        for _ in range(20):
            orch.simulate_research_update("concept", "signal", 5.0)

        # Force execution
        result = orch.run_synthesis_job("Forced_Job", "forced_code", force=True)

        assert result.executed is True
        assert runner.called is True
        assert runner.last_code == "forced_code"

    def test_no_runner_configured(self, sentinel):
        """Test behavior when no runner is configured."""
        orch = DrugSynthesisOrchestrator(sentinel, run_code_fn=None, verbose=False)

        orch.simulate_research_update("protein", "signal", 1.0)
        result = orch.run_synthesis_job("No_Runner_Job", "code")

        assert result.executed is False
        assert result.blocked_reason == "NO_RUNNER_CONFIGURED"

    def test_execution_disabled_in_config(self, sentinel):
        """Test behavior when execution is disabled."""
        from ara_core.cadd.orchestrator import CodeSnippetConfig

        config = CodeSnippetConfig(enabled=False)
        orch = DrugSynthesisOrchestrator(
            sentinel,
            run_code_fn=lambda x: None,
            snippet_config=config,
            verbose=False,
        )

        orch.simulate_research_update("protein", "signal", 1.0)
        result = orch.run_synthesis_job("Disabled_Job", "code")

        assert result.executed is False
        assert result.blocked_reason == "EXECUTION_DISABLED"

    def test_stats_tracking(self, orchestrator):
        """Test that orchestrator tracks job statistics."""
        orch, runner = orchestrator

        # Run some jobs
        orch.simulate_research_update("protein", "signal", 1.0)
        orch.run_synthesis_job("Job_1", "code1")
        orch.run_synthesis_job("Job_2", "code2")

        stats = orch.get_stats()

        assert stats["total_jobs_attempted"] == 2
        assert stats["n_agents"] >= 4  # Default agents

    def test_direct_agent_update(self, orchestrator):
        """Test direct association updates."""
        orch, runner = orchestrator

        orch.update_agent_association("custom_agent", "concept", "signal", 0.8)

        assert "custom_agent" in orch.agent_ids
        assert "custom_agent" in orch.sentinel.agents


# =============================================================================
# Integration Tests
# =============================================================================

class TestCADDIntegration:
    """Integration tests for the full CADD pipeline."""

    def test_full_pipeline_healthy_execution(self):
        """Test full pipeline with healthy swarm."""
        # Fresh sentinel and orchestrator
        cfg = SentinelConfig(h_influence_min=0.3)
        sentinel = CADDSentinel(config=cfg)
        runner = DummyRunner()
        orch = DrugSynthesisOrchestrator(sentinel, run_code_fn=runner, verbose=False)

        # Simulate diverse research
        concepts = ["protein_A", "protein_B", "protein_C"]
        signals = ["binding", "affinity", "stability"]

        for concept in concepts:
            for signal in signals:
                orch.simulate_research_update(concept, signal, 0.7)

        # Run synthesis
        result = orch.run_synthesis_job(
            "Full_Pipeline_Job",
            "synthesis_protocol_execution()"
        )

        assert result.executed is True
        assert runner.call_count == 1
        assert "synthesis_protocol" in runner.last_code

    def test_recovery_after_diversity_injection(self):
        """Test that swarm can recover after diversity injection."""
        cfg = SentinelConfig(
            h_influence_min=0.5,
            diversity_spawn_count=5,
        )
        sentinel = CADDSentinel(config=cfg)
        runner = DummyRunner()
        orch = DrugSynthesisOrchestrator(sentinel, run_code_fn=runner, verbose=False)

        # Create monoculture
        for _ in range(10):
            orch.simulate_research_update("single", "signal", 5.0)

        # First job should be blocked
        result1 = orch.run_synthesis_job("Blocked_Job", "code")
        assert result1.executed is False

        # Now add diverse associations to new agents
        for agent in orch.agent_ids:
            if agent.startswith("Diverse_"):
                orch.update_agent_association(agent, "new_concept", "new_signal", 1.0)
                orch.update_agent_association(agent, "other_concept", "other_signal", 0.8)

        # After diversity, subsequent jobs might pass
        # (depends on entropy reaching threshold)
        _ = orch.run_synthesis_job("Recovery_Job", "code")

        # At minimum, we should have more agents now
        assert len(orch.agent_ids) >= 9  # 4 original + 5 diverse


# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

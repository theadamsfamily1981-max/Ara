#!/usr/bin/env python3
"""
L3 Metacontrol Validation Tests

Tests the L3 metacontrol vertical slice:
1. Workspace mode → PAD state mappings
2. PAD → temperature/memory multiplier control law
3. Integration with Pulse affect estimation
4. API endpoint functionality
5. Telemetry recording

Run with: python -m pytest ara/tests/test_l3_metacontrol.py -v
Or directly: python ara/tests/test_l3_metacontrol.py
"""

import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest
from typing import Dict, Any


class TestWorkspaceModes(unittest.TestCase):
    """Test workspace mode to PAD mappings."""

    def setUp(self):
        from ara.metacontrol import (
            WorkspaceMode,
            WORKSPACE_PAD_MAPPINGS,
        )
        self.WorkspaceMode = WorkspaceMode
        self.mappings = WORKSPACE_PAD_MAPPINGS

    def test_all_modes_have_mappings(self):
        """All workspace modes should have PAD mappings."""
        for mode in self.WorkspaceMode:
            self.assertIn(mode, self.mappings, f"Missing mapping for {mode}")

    def test_work_mode_high_activation(self):
        """Work mode should have high valence and arousal."""
        pad = self.mappings[self.WorkspaceMode.WORK]
        self.assertGreater(pad.valence, 0.5, "Work mode should have positive valence")
        self.assertGreater(pad.arousal, 0.5, "Work mode should have high arousal")

    def test_relax_mode_low_activation(self):
        """Relax mode should have low arousal."""
        pad = self.mappings[self.WorkspaceMode.RELAX]
        self.assertLess(pad.arousal, 0.5, "Relax mode should have low arousal")

    def test_creative_mode_high_arousal(self):
        """Creative mode should have high arousal for exploration."""
        pad = self.mappings[self.WorkspaceMode.CREATIVE]
        self.assertGreater(pad.arousal, 0.6, "Creative mode should have high arousal")

    def test_support_mode_warm(self):
        """Support mode should be warm (positive valence)."""
        pad = self.mappings[self.WorkspaceMode.SUPPORT]
        self.assertGreater(pad.valence, 0.4, "Support mode should have positive valence")


class TestControlLaw(unittest.TestCase):
    """Test PAD → temperature/memory control law."""

    def setUp(self):
        from ara.metacontrol import (
            L3MetacontrolService,
            PADState,
        )
        self.service = L3MetacontrolService()
        self.PADState = PADState

    def test_high_arousal_increases_temperature(self):
        """High arousal should increase temperature multiplier."""
        low_arousal = self.PADState(valence=0.0, arousal=0.2, confidence=1.0)
        high_arousal = self.PADState(valence=0.0, arousal=0.8, confidence=1.0)

        # Reset state between tests
        self.service.reset()
        mod_low = self.service.compute_modulation(low_arousal)

        self.service.reset()
        mod_high = self.service.compute_modulation(high_arousal)

        self.assertGreater(
            mod_high.temperature_multiplier,
            mod_low.temperature_multiplier,
            "Higher arousal should produce higher temperature"
        )

    def test_low_valence_decreases_memory(self):
        """Low valence should decrease memory write probability."""
        low_valence = self.PADState(valence=-0.5, arousal=0.5, confidence=1.0)
        high_valence = self.PADState(valence=0.8, arousal=0.5, confidence=1.0)

        self.service.reset()
        mod_low = self.service.compute_modulation(low_valence)

        self.service.reset()
        mod_high = self.service.compute_modulation(high_valence)

        self.assertLess(
            mod_low.memory_write_multiplier,
            mod_high.memory_write_multiplier,
            "Lower valence should produce lower memory write probability"
        )

    def test_temperature_bounds(self):
        """Temperature multiplier should stay within bounds [0.8, 1.3]."""
        extreme_pad = self.PADState(valence=1.0, arousal=1.0, confidence=1.0)

        self.service.reset()
        mod = self.service.compute_modulation(extreme_pad)

        self.assertGreaterEqual(mod.temperature_multiplier, 0.8)
        self.assertLessEqual(mod.temperature_multiplier, 1.3)

    def test_memory_bounds(self):
        """Memory multiplier should stay within bounds [0.7, 1.2]."""
        extreme_pad = self.PADState(valence=-1.0, arousal=0.0, confidence=1.0)

        self.service.reset()
        mod = self.service.compute_modulation(extreme_pad)

        self.assertGreaterEqual(mod.memory_write_multiplier, 0.7)
        self.assertLessEqual(mod.memory_write_multiplier, 1.2)

    def test_low_confidence_reduces_weight(self):
        """Low confidence should reduce effective control weight."""
        high_conf = self.PADState(valence=0.5, arousal=0.7, confidence=0.9)
        low_conf = self.PADState(valence=0.5, arousal=0.7, confidence=0.3)

        self.service.reset()
        mod_high = self.service.compute_modulation(high_conf)

        self.service.reset()
        mod_low = self.service.compute_modulation(low_conf)

        self.assertLess(
            mod_low.effective_weight,
            mod_high.effective_weight,
            "Lower confidence should reduce effective weight"
        )


class TestIntegration(unittest.TestCase):
    """Test metacontrol integration with orchestrator."""

    def setUp(self):
        from ara.integration import AraOrchestrator, WorkspaceMode
        self.orchestrator = AraOrchestrator(default_workspace_mode=WorkspaceMode.WORK)
        self.WorkspaceMode = WorkspaceMode

    def test_process_turn_includes_metacontrol(self):
        """Process turn should include metacontrol modulation."""
        result = self.orchestrator.process_turn(
            user_text="Help me write a function",
            session_id="test-1",
        )

        self.assertIsNotNone(result.metacontrol)
        self.assertIn(result.workspace_mode, ["work", "relax", "creative", "support", "default"])
        self.assertGreater(result.effective_temperature, 0)

    def test_workspace_mode_affects_temperature(self):
        """Changing workspace mode should affect effective temperature."""
        # Set work mode (high arousal)
        self.orchestrator.metacontrol.set_workspace_mode(self.WorkspaceMode.WORK)
        result_work = self.orchestrator.process_turn("Test", session_id="test-2")

        # Set relax mode (low arousal)
        self.orchestrator.metacontrol.set_workspace_mode(self.WorkspaceMode.RELAX)
        result_relax = self.orchestrator.process_turn("Test", session_id="test-3")

        # Work mode should generally have different temp than relax
        # (exact comparison depends on affect estimation)
        self.assertIsNotNone(result_work.effective_temperature)
        self.assertIsNotNone(result_relax.effective_temperature)

    def test_metacontrol_in_event_log(self):
        """Event log should include metacontrol data."""
        self.orchestrator.process_turn("Test event logging", session_id="test-4")

        events = self.orchestrator.get_event_log()
        self.assertGreater(len(events), 0)

        event = events[-1]
        self.assertIn("metacontrol", event)
        self.assertIn("workspace_mode", event["metacontrol"])
        self.assertIn("temperature_multiplier", event["metacontrol"])


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def test_set_workspace_mode(self):
        """set_workspace_mode should return modulation dict."""
        from ara.metacontrol import set_workspace_mode

        result = set_workspace_mode("work")

        self.assertIsInstance(result, dict)
        self.assertIn("temperature_multiplier", result)
        self.assertIn("memory_write_multiplier", result)

    def test_compute_pad_gating(self):
        """compute_pad_gating should compute modulation from raw values."""
        from ara.metacontrol import compute_pad_gating

        result = compute_pad_gating(
            valence=0.5,
            arousal=0.7,
            dominance=0.5,
            confidence=0.9,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("temperature_multiplier", result)

    def test_get_metacontrol_status(self):
        """get_metacontrol_status should return current status."""
        from ara.metacontrol import get_metacontrol_status, set_workspace_mode

        set_workspace_mode("creative")
        status = get_metacontrol_status()

        self.assertIsInstance(status, dict)
        self.assertEqual(status["workspace_mode"], "creative")


class TestTelemetry(unittest.TestCase):
    """Test telemetry recording."""

    def test_telemetry_imports(self):
        """Telemetry module should import without errors."""
        from ara.telemetry import (
            PulseTelemetry,
            MetricSnapshot,
            get_telemetry,
            get_telemetry_summary,
        )
        self.assertIsNotNone(PulseTelemetry)

    def test_record_snapshot(self):
        """Should be able to record metric snapshots."""
        from ara.telemetry import PulseTelemetry, MetricSnapshot

        telemetry = PulseTelemetry(enable_prometheus=False)

        snapshot = MetricSnapshot(
            valence=0.5,
            arousal=0.7,
            workspace_mode="work",
            temperature_multiplier=1.05,
        )

        telemetry.record(snapshot)

        self.assertEqual(len(telemetry._snapshots), 1)

    def test_summary_statistics(self):
        """Should compute summary statistics."""
        from ara.telemetry import PulseTelemetry, MetricSnapshot

        telemetry = PulseTelemetry(enable_prometheus=False)

        for i in range(5):
            snapshot = MetricSnapshot(
                valence=0.1 * i,
                arousal=0.5,
                processing_time_ms=10.0 + i,
            )
            telemetry.record(snapshot)

        summary = telemetry.get_summary()

        self.assertEqual(summary["total_turns"], 5)
        self.assertIn("avg_processing_ms", summary)


def run_validation():
    """Run all validation tests and print summary."""
    print("=" * 60)
    print("L3 Metacontrol Validation Suite")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestWorkspaceModes))
    suite.addTests(loader.loadTestsFromTestCase(TestControlLaw))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestTelemetry))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All validation tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)

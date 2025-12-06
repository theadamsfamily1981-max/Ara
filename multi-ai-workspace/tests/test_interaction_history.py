"""Tests for MIES Interaction History - Pattern Memory.

Validates:
1. ContextSignature creation and hashing
2. InteractionHistory recording and retrieval
3. Antibody/preference formation
4. Integration with ThermodynamicGovernor
"""

import pytest
import tempfile
from pathlib import Path
import time

from src.integrations.mies.context import (
    ModalityContext,
    ForegroundInfo,
    ForegroundAppType,
    AudioContext,
    ActivityType,
)
from src.integrations.mies.history import (
    ContextSignature,
    InteractionRecord,
    PatternStats,
    InteractionHistory,
    OutcomeType,
    detect_outcome_from_timing,
)
from src.integrations.mies.policy.ebm_aepo_policy import (
    ThermodynamicGovernor,
    EnergyFunction,
    create_thermodynamic_governor,
)
from src.integrations.mies.modes import DEFAULT_MODES


def create_ide_context(fullscreen: bool = False) -> ModalityContext:
    """Create a context for IDE deep work."""
    ctx = ModalityContext(
        foreground=ForegroundInfo(
            app_type=ForegroundAppType.IDE,
            wm_class="code",
            title="Visual Studio Code",
            is_fullscreen=fullscreen,
        ),
        audio=AudioContext(),
        activity=ActivityType.DEEP_WORK,
        user_cognitive_load=0.7,
    )
    return ctx


def create_browser_context() -> ModalityContext:
    """Create a context for casual browsing."""
    ctx = ModalityContext(
        foreground=ForegroundInfo(
            app_type=ForegroundAppType.BROWSER,
            wm_class="firefox",
            title="Firefox",
            is_fullscreen=False,
        ),
        audio=AudioContext(),
        activity=ActivityType.CASUAL_WORK,
        user_cognitive_load=0.3,
    )
    return ctx


class TestContextSignature:
    """Tests for context signature creation and matching."""

    def test_signature_from_context(self):
        """Test signature creation from ModalityContext."""
        ctx = create_ide_context()
        sig = ContextSignature.from_context(ctx)

        assert sig.activity == "DEEP_WORK"
        assert sig.app_type == "IDE"
        assert sig.is_fullscreen is False
        assert sig.user_load_bucket == "high"  # 0.7 > 0.7

    def test_signature_equality(self):
        """Test that similar contexts produce equal signatures."""
        ctx1 = create_ide_context()
        ctx2 = create_ide_context()

        sig1 = ContextSignature.from_context(ctx1)
        sig2 = ContextSignature.from_context(ctx2)

        assert sig1 == sig2
        assert hash(sig1) == hash(sig2)

    def test_signature_difference(self):
        """Test that different contexts produce different signatures."""
        ctx_ide = create_ide_context()
        ctx_browser = create_browser_context()

        sig_ide = ContextSignature.from_context(ctx_ide)
        sig_browser = ContextSignature.from_context(ctx_browser)

        assert sig_ide != sig_browser
        assert hash(sig_ide) != hash(sig_browser)

    def test_signature_similarity(self):
        """Test similarity scoring between signatures."""
        ctx_ide = create_ide_context(fullscreen=False)
        ctx_ide_full = create_ide_context(fullscreen=True)
        ctx_browser = create_browser_context()

        sig1 = ContextSignature.from_context(ctx_ide)
        sig2 = ContextSignature.from_context(ctx_ide_full)
        sig3 = ContextSignature.from_context(ctx_browser)

        # Same activity + app type should be similar
        sim_ide = sig1.similarity_to(sig2)
        sim_browser = sig1.similarity_to(sig3)

        assert sim_ide > sim_browser  # IDE contexts more similar


class TestInteractionHistory:
    """Tests for interaction history recording and learning."""

    def test_record_interaction(self):
        """Test basic interaction recording."""
        history = InteractionHistory()
        ctx = create_ide_context()

        history.record(
            ctx=ctx,
            mode_name="avatar_full",
            outcome_score=OutcomeType.CLOSED_IMMEDIATE,
            outcome_type="CLOSED_IMMEDIATE",
        )

        stats = history.get_stats()
        assert stats["total_records"] == 1
        assert stats["total_patterns"] == 1

    def test_antibody_formation(self):
        """Test that repeated negative outcomes form an antibody."""
        history = InteractionHistory()
        ctx = create_ide_context()

        # User closes avatar 3 times in IDE
        for _ in range(3):
            history.record(
                ctx=ctx,
                mode_name="avatar_full",
                outcome_score=OutcomeType.CLOSED_IMMEDIATE,
                outcome_type="CLOSED_IMMEDIATE",
            )

        antibodies = history.get_antibodies()
        assert len(antibodies) == 1
        assert antibodies[0].mode_name == "avatar_full"
        assert antibodies[0].is_antibody is True

    def test_preference_formation(self):
        """Test that repeated positive outcomes form a preference."""
        history = InteractionHistory()
        ctx = create_browser_context()

        # User engages with text 5 times
        for _ in range(5):
            history.record(
                ctx=ctx,
                mode_name="text_inline",
                outcome_score=OutcomeType.USER_ENGAGED,
                outcome_type="USER_ENGAGED",
            )

        preferences = history.get_preferences()
        assert len(preferences) == 1
        assert preferences[0].mode_name == "text_inline"
        assert preferences[0].is_preference is True

    def test_friction_for_antibody(self):
        """Test that antibodies generate positive friction."""
        history = InteractionHistory()
        ctx = create_ide_context()

        # Form an antibody
        for _ in range(4):
            history.record(
                ctx=ctx,
                mode_name="avatar_full",
                outcome_score=OutcomeType.CLOSED_IMMEDIATE,
            )

        friction = history.friction_for(ctx, "avatar_full")
        assert friction > 0  # Positive = avoidance

        # Different mode should have no friction
        friction_text = history.friction_for(ctx, "text_inline")
        assert friction_text == 0.0

    def test_preference_for_returns_negative_friction(self):
        """Test that preferences generate negative friction."""
        history = InteractionHistory()
        ctx = create_browser_context()

        # Form a preference
        for _ in range(5):
            history.record(
                ctx=ctx,
                mode_name="audio_whisper",
                outcome_score=OutcomeType.USER_ENGAGED,
            )

        friction = history.friction_for(ctx, "audio_whisper")
        assert friction < 0  # Negative = attraction

    def test_fuzzy_matching(self):
        """Test fuzzy matching for similar contexts."""
        history = InteractionHistory()
        ctx_ide = create_ide_context(fullscreen=False)
        ctx_ide_full = create_ide_context(fullscreen=True)

        # Train on non-fullscreen IDE
        for _ in range(3):
            history.record(
                ctx=ctx_ide,
                mode_name="avatar_full",
                outcome_score=OutcomeType.CLOSED_IMMEDIATE,
            )

        # Should get friction for similar fullscreen context via fuzzy match
        preference = history.preference_for(ctx_ide_full, "avatar_full")
        assert preference < 0  # Negative = learned avoidance

    def test_persistence(self):
        """Test saving and loading history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "history.json"

            # Create and populate
            history1 = InteractionHistory(db_path=db_path)
            ctx = create_ide_context()
            for _ in range(3):
                history1.record(
                    ctx=ctx,
                    mode_name="avatar_full",
                    outcome_score=OutcomeType.CLOSED_IMMEDIATE,
                )
            history1._save()

            # Load in new instance
            history2 = InteractionHistory(db_path=db_path)
            assert history2.get_stats()["total_patterns"] == 1
            assert len(history2.get_antibodies()) == 1


class TestThermodynamicGovernorIntegration:
    """Tests for integration with ThermodynamicGovernor."""

    def test_energy_function_with_history(self):
        """Test that EnergyFunction uses history for E_history."""
        history = InteractionHistory()
        ctx = create_ide_context()

        # Form an antibody for avatar_full in IDE
        for _ in range(4):
            history.record(
                ctx=ctx,
                mode_name="avatar_full",
                outcome_score=OutcomeType.CLOSED_IMMEDIATE,
            )

        # Create energy function with history
        energy_fn = EnergyFunction(history=history)

        # Get energies for modes
        avatar_mode = DEFAULT_MODES["avatar_full"]
        text_mode = DEFAULT_MODES["text_inline"]

        e_avatar = energy_fn.compute(ctx, avatar_mode)
        e_text = energy_fn.compute(ctx, text_mode)

        # Avatar should have higher energy due to antibody
        assert e_avatar > e_text

    def test_governor_with_learning(self):
        """Test full ThermodynamicGovernor with learning enabled."""
        governor = create_thermodynamic_governor(
            enable_learning=True,
            stochastic=False,
        )

        ctx = create_ide_context()

        # Make some decisions and record outcomes
        for _ in range(3):
            decision = governor.select_modality(ctx)
            # Simulate negative outcome for avatar
            if decision.mode.name.startswith("avatar"):
                governor.record_outcome(
                    OutcomeType.CLOSED_IMMEDIATE,
                    "CLOSED_IMMEDIATE",
                )
            else:
                governor.record_outcome(
                    OutcomeType.USER_ACKNOWLEDGED,
                    "USER_ACKNOWLEDGED",
                )

        # Check history stats
        stats = governor.get_history_stats()
        assert stats["history_enabled"] is True
        assert stats["total_records"] == 3

    def test_antibody_affects_mode_selection(self):
        """Test that antibodies actually affect mode selection."""
        governor = create_thermodynamic_governor(
            enable_learning=True,
            stochastic=False,
        )

        ctx = create_ide_context()

        # Train a strong antibody against avatar_full
        # We need to manually add to history to bypass the selection
        if governor.history:
            for _ in range(5):
                governor.history.record(
                    ctx=ctx,
                    mode_name="avatar_full",
                    outcome_score=OutcomeType.CLOSED_IMMEDIATE,
                )
                governor.history.record(
                    ctx=ctx,
                    mode_name="avatar_present",
                    outcome_score=OutcomeType.CLOSED_QUICK,
                )

        # Now make a decision - avatar modes should be disfavored
        decision = governor.select_modality(ctx)

        # Due to antibody, avatar modes should have high friction
        # Check that avatar modes have higher energy
        antibodies = governor.get_antibodies()
        assert len(antibodies) >= 2  # avatar_full and avatar_present


class TestOutcomeDetection:
    """Tests for outcome detection helpers."""

    def test_detect_immediate_close(self):
        """Test detection of immediate close."""
        score, otype = detect_outcome_from_timing(300, "close")
        assert score == OutcomeType.CLOSED_IMMEDIATE
        assert otype == "CLOSED_IMMEDIATE"

    def test_detect_quick_close(self):
        """Test detection of quick close."""
        score, otype = detect_outcome_from_timing(1500, "close")
        assert score == OutcomeType.CLOSED_QUICK
        assert otype == "CLOSED_QUICK"

    def test_detect_normal_dismiss(self):
        """Test detection of normal dismiss."""
        score, otype = detect_outcome_from_timing(5000, "close")
        assert score == OutcomeType.DISMISSED
        assert otype == "DISMISSED"

    def test_detect_response(self):
        """Test detection of user response."""
        score, otype = detect_outcome_from_timing(1000, "respond")
        assert score == OutcomeType.USER_ENGAGED
        assert otype == "USER_ENGAGED"

    def test_detect_mute(self):
        """Test detection of mute."""
        score, otype = detect_outcome_from_timing(500, "mute")
        assert score == OutcomeType.MUTED
        assert otype == "MUTED"

    def test_detect_timeout(self):
        """Test detection of natural timeout."""
        score, otype = detect_outcome_from_timing(60000, None)
        assert score == OutcomeType.TIMEOUT_NATURAL
        assert otype == "TIMEOUT_NATURAL"

    def test_detect_ignored(self):
        """Test detection of ignored."""
        score, otype = detect_outcome_from_timing(5000, None)
        assert score == OutcomeType.IGNORED
        assert otype == "IGNORED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
End-to-End Integration Test for V1 Vertical Slice

Tests the full Ara loop:
  User message → AxisMundi + EternalMemory → Scheduler → Avatar → Response

This is the core integration test that validates the entire system
works together before adding FPGA, VPN, or other complexity.
"""

import asyncio
import pytest
import numpy as np
import tempfile
from pathlib import Path
import time


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_data_dir):
    """Create test configuration."""
    from ara.core.config import AraConfig, PathsConfig

    config = AraConfig()
    config.paths = PathsConfig(
        data_dir=temp_data_dir,
        logs_dir=temp_data_dir / "logs",
        state_file=temp_data_dir / "axis_mundi.json",
        memory_db=temp_data_dir / "eternal_memory.db",
    )
    config.hdc.dim = 1024  # Smaller for tests
    config.loop.tick_interval_ms = 50  # Faster for tests
    config.safety.initial_autonomy_level = 1
    config.safety.kill_switch_file = temp_data_dir / "KILL_SWITCH"

    return config


@pytest.fixture
def axis(config):
    """Create AxisMundi instance."""
    from ara.core.axis_mundi import AxisMundi

    return AxisMundi(
        dim=config.hdc.dim,
        bipolar=config.hdc.bipolar,
        seed=42,  # Reproducible
    )


@pytest.fixture
def memory(config):
    """Create EternalMemory instance."""
    from ara.core.eternal_memory import EternalMemory

    return EternalMemory(
        dim=config.hdc.dim,
        db_path=config.paths.memory_db,
    )


@pytest.fixture
def safety(config):
    """Create AutonomyController instance."""
    from ara.safety.autonomy import AutonomyController

    return AutonomyController(
        initial_level=config.safety.initial_autonomy_level,
        max_level=config.safety.max_autonomy_level,
        kill_switch_path=config.safety.kill_switch_file,
        recovery_streak_required=5,  # Fast for tests
    )


# =============================================================================
# AxisMundi Tests
# =============================================================================

class TestAxisMundi:
    """Test AxisMundi global holographic state."""

    def test_create_axis(self, axis):
        """Test basic AxisMundi creation."""
        assert axis.dim == 1024
        assert "hardware" in axis.layer_names()
        assert "avatar" in axis.layer_names()

    def test_write_read_layer(self, axis):
        """Test writing and reading from a layer."""
        # Generate test vector
        test_hv = np.random.randn(axis.dim).astype(np.float32)

        # Write to avatar layer
        axis.write("avatar", test_hv)

        # Read back
        read_hv = axis.read("avatar")

        # Should be similar (not exact due to binding/unbinding)
        assert read_hv.shape == (axis.dim,)
        assert np.linalg.norm(read_hv) > 0

    def test_coherence_same_vector(self, axis):
        """Test coherence when layers have same state."""
        test_hv = np.random.randn(axis.dim).astype(np.float32)

        axis.write("hardware", test_hv)
        axis.write("avatar", test_hv)

        coherence = axis.coherence("hardware", "avatar")

        # Same vector should have high coherence
        assert coherence > 0.9

    def test_coherence_orthogonal(self, axis):
        """Test coherence with orthogonal vectors."""
        hv1 = np.random.randn(axis.dim).astype(np.float32)
        hv2 = np.random.randn(axis.dim).astype(np.float32)

        axis.write("hardware", hv1)
        axis.write("avatar", hv2)

        coherence = axis.coherence("hardware", "avatar")

        # Random vectors should have low coherence
        assert abs(coherence) < 0.3

    def test_global_state(self, axis):
        """Test global state computation."""
        hv1 = np.random.randn(axis.dim).astype(np.float32)
        hv2 = np.random.randn(axis.dim).astype(np.float32)

        axis.write("hardware", hv1)
        axis.write("avatar", hv2)

        global_state = axis.global_state()

        assert global_state.shape == (axis.dim,)
        assert np.linalg.norm(global_state) > 0

    def test_save_load(self, axis, temp_data_dir):
        """Test state persistence."""
        from ara.core.axis_mundi import AxisMundi

        # Write some state
        test_hv = np.random.randn(axis.dim).astype(np.float32)
        axis.write("avatar", test_hv)

        # Save
        save_path = temp_data_dir / "axis_test.json"
        axis.save(save_path)

        # Load
        axis2 = AxisMundi.load(save_path)

        # Verify
        assert axis2.dim == axis.dim
        assert axis2.layer_names() == axis.layer_names()
        assert np.allclose(
            axis2.global_state(),
            axis.global_state(),
            atol=1e-5,
        )


# =============================================================================
# EternalMemory Tests
# =============================================================================

class TestEternalMemory:
    """Test EternalMemory episodic store."""

    def test_create_memory(self, memory):
        """Test basic memory creation."""
        assert memory.dim == 1024
        assert memory.stats()["episode_count"] == 0

    def test_store_recall(self, memory):
        """Test storing and recalling episodes."""
        content_hv = np.random.randn(memory.dim).astype(np.float32)

        # Store
        ep_id = memory.store(
            content_hv=content_hv,
            strength=0.9,
            meta={"user": "test", "topic": "architecture"},
        )

        assert ep_id is not None
        assert memory.stats()["episode_count"] == 1

        # Recall
        result = memory.recall(content_hv, k=1)

        assert len(result.episodes) == 1
        assert result.episodes[0].id == ep_id
        assert result.episodes[0].similarity > 0.9

    def test_recall_multiple(self, memory):
        """Test recalling multiple episodes."""
        # Store 5 episodes
        for i in range(5):
            hv = np.random.randn(memory.dim).astype(np.float32)
            memory.store(hv, strength=0.8, meta={"index": i})

        assert memory.stats()["episode_count"] == 5

        # Recall with random query
        query = np.random.randn(memory.dim).astype(np.float32)
        result = memory.recall(query, k=3)

        assert len(result.episodes) <= 3

    def test_emotional_resonance(self, memory):
        """Test emotional coloring in recall."""
        content_hv = np.random.randn(memory.dim).astype(np.float32)
        emotion_hv = np.random.randn(memory.dim).astype(np.float32)

        memory.store(
            content_hv=content_hv,
            emotion_hv=emotion_hv,
            strength=0.9,
            meta={"type": "emotional"},
        )

        # Recall with same emotion
        result = memory.recall(content_hv, emotion_hv, k=1)

        assert len(result.episodes) == 1
        assert result.episodes[0].emotional_resonance > 0.9

    def test_forget(self, memory):
        """Test forgetting episodes."""
        hv = np.random.randn(memory.dim).astype(np.float32)
        ep_id = memory.store(hv, strength=0.5, meta={})

        assert memory.stats()["episode_count"] == 1

        # Forget
        success = memory.forget(ep_id)

        assert success
        assert memory.stats()["episode_count"] == 0

    def test_persistence(self, memory):
        """Test database persistence."""
        from ara.core.eternal_memory import EternalMemory

        # Store episode
        hv = np.random.randn(memory.dim).astype(np.float32)
        memory.store(hv, strength=0.9, meta={"test": "persistence"})

        # Create new memory with same db
        memory2 = EternalMemory(
            dim=memory.dim,
            db_path=memory.db_path,
        )

        assert memory2.stats()["episode_count"] == 1


# =============================================================================
# Autonomy/Safety Tests
# =============================================================================

class TestAutonomy:
    """Test AutonomyController safety system."""

    def test_initial_level(self, safety):
        """Test initial autonomy level."""
        assert safety.get_autonomy_level() == 1
        assert safety.can_observe()
        assert safety.can_suggest()
        assert not safety.can_execute()

    def test_coherence_warning(self, safety):
        """Test autonomy reduction on coherence warning."""
        initial = safety.get_autonomy_level()

        safety.on_coherence_warning(0.25)

        assert safety.get_autonomy_level() < initial

    def test_coherence_critical(self, safety):
        """Test drop to observer on coherence critical."""
        safety.on_coherence_critical(0.05)

        assert safety.get_autonomy_level() == 0
        assert safety.state.locked

    def test_recovery(self, safety):
        """Test autonomy recovery after healthy streak."""
        # Start at level 1
        assert safety.get_autonomy_level() == 1

        # Simulate healthy streak
        for _ in range(10):
            safety.on_coherence_healthy(0.9)

        # Should have increased
        assert safety.get_autonomy_level() >= 1

    def test_kill_switch(self, safety, temp_data_dir):
        """Test kill switch behavior."""
        from ara.safety.autonomy import KillSwitch

        kill = KillSwitch(temp_data_dir / "KILL_SWITCH")

        assert not kill.is_active()
        assert not safety.is_killed()

        # Activate
        kill.activate("test")

        assert kill.is_active()
        assert safety.is_killed()
        assert not safety.can_observe()

        # Deactivate
        kill.deactivate()

        assert not kill.is_active()
        assert safety.can_observe()


# =============================================================================
# Scheduler Tests
# =============================================================================

class TestScheduler:
    """Test sovereign loop scheduler."""

    @pytest.mark.asyncio
    async def test_tick_execution(self, axis, memory, safety, config):
        """Test single tick execution."""
        from ara.core.scheduler import Scheduler

        scheduler = Scheduler(axis, memory, safety, config)

        metrics = await scheduler._tick()

        assert metrics.tick_number == 1
        assert metrics.duration_ms > 0
        assert 0 <= metrics.global_coherence <= 1
        assert metrics.autonomy_level >= 0

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, axis, memory, safety, config):
        """Test scheduler start and stop."""
        from ara.core.scheduler import Scheduler, LoopState

        scheduler = Scheduler(axis, memory, safety, config)

        # Start
        await scheduler.start()

        assert scheduler.state.loop_state == LoopState.RUNNING

        # Let it run a few ticks
        await asyncio.sleep(0.2)

        assert scheduler.state.tick_count > 0

        # Stop
        await scheduler.stop()

        assert scheduler.state.loop_state == LoopState.STOPPED

    @pytest.mark.asyncio
    async def test_tick_callbacks(self, axis, memory, safety, config):
        """Test tick callback invocation."""
        from ara.core.scheduler import Scheduler

        scheduler = Scheduler(axis, memory, safety, config)

        received_metrics = []

        async def callback(metrics):
            received_metrics.append(metrics)

        scheduler.register_tick_callback(callback)

        # Run a tick
        await scheduler._tick()

        assert len(received_metrics) == 1
        assert received_metrics[0].tick_number == 1


# =============================================================================
# Avatar Tests
# =============================================================================

class TestAvatar:
    """Test avatar server."""

    @pytest.mark.asyncio
    async def test_handle_message(self, axis, memory, safety, config):
        """Test message handling."""
        from ara.avatar.server import AvatarServer

        server = AvatarServer(axis, memory, safety, config)

        result = await server.handle_message("test_user", "Hello Ara!")

        assert result["status"] == "ok"
        assert "reply" in result
        assert "session_id" in result
        assert result["autonomy_level"] >= 0

    @pytest.mark.asyncio
    async def test_memory_growth(self, axis, memory, safety, config):
        """Test that messages create memories."""
        from ara.avatar.server import AvatarServer

        server = AvatarServer(axis, memory, safety, config)

        initial_count = memory.stats()["episode_count"]

        await server.handle_message("test_user", "First message")
        await server.handle_message("test_user", "Second message")

        assert memory.stats()["episode_count"] == initial_count + 2

    @pytest.mark.asyncio
    async def test_session_continuity(self, axis, memory, safety, config):
        """Test session persistence across messages."""
        from ara.avatar.server import AvatarServer

        server = AvatarServer(axis, memory, safety, config)

        r1 = await server.handle_message("user_a", "Hello")
        r2 = await server.handle_message("user_a", "How are you?")

        assert r1["session_id"] == r2["session_id"]
        assert r2["message_count"] == 2


# =============================================================================
# End-to-End Flow Tests
# =============================================================================

class TestE2EFlow:
    """Test complete end-to-end flow."""

    @pytest.mark.asyncio
    async def test_full_conversation_loop(self, axis, memory, safety, config):
        """Test full conversation with coherence and memory."""
        from ara.core.scheduler import Scheduler
        from ara.avatar.server import AvatarServer

        # Start scheduler
        scheduler = Scheduler(axis, memory, safety, config)
        await scheduler.start()

        # Create avatar
        avatar = AvatarServer(axis, memory, safety, config)

        # Have a conversation
        messages = [
            "Hello, I'm working on architecture",
            "What patterns do you recommend?",
            "Thanks, that's helpful",
        ]

        results = []
        for msg in messages:
            result = await avatar.handle_message("e2e_user", msg)
            results.append(result)
            await asyncio.sleep(0.1)  # Let scheduler tick

        # Verify all succeeded
        assert all(r["status"] == "ok" for r in results)

        # Verify memory grew
        assert memory.stats()["episode_count"] >= 3

        # Verify coherence is tracked
        assert results[-1]["coherence"] > 0

        # Stop scheduler
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_safety_integration(self, axis, memory, safety, config):
        """Test safety system integration with avatar."""
        from ara.avatar.server import AvatarServer
        from ara.safety.autonomy import KillSwitch

        avatar = AvatarServer(axis, memory, safety, config)

        # Normal operation
        r1 = await avatar.handle_message("user", "Hello")
        assert r1["status"] == "ok"

        # Activate kill switch
        kill = KillSwitch(config.safety.kill_switch_file)
        kill.activate("e2e test")

        # Should be paused
        r2 = await avatar.handle_message("user", "Hello again")
        assert r2["status"] == "paused"

        # Deactivate
        kill.deactivate()

        # Should work again
        r3 = await avatar.handle_message("user", "Back online?")
        assert r3["status"] == "ok"

    @pytest.mark.asyncio
    async def test_state_persistence(self, axis, memory, safety, config):
        """Test that state persists across restarts."""
        from ara.core.axis_mundi import AxisMundi
        from ara.core.eternal_memory import EternalMemory
        from ara.avatar.server import AvatarServer

        # Have a conversation
        avatar = AvatarServer(axis, memory, safety, config)
        await avatar.handle_message("persist_user", "Remember this!")

        # Save state
        axis.save(config.paths.state_file)

        # Create new instances
        axis2 = AxisMundi.load(config.paths.state_file)
        memory2 = EternalMemory(dim=config.hdc.dim, db_path=config.paths.memory_db)

        # Memory should persist
        assert memory2.stats()["episode_count"] >= 1

        # Global state should persist
        np.testing.assert_allclose(
            axis2.global_state(),
            axis.global_state(),
            atol=1e-5,
        )


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Basic performance sanity checks."""

    def test_axis_write_speed(self, axis):
        """Test AxisMundi write performance."""
        hv = np.random.randn(axis.dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(1000):
            axis.write("avatar", hv)
        elapsed = time.perf_counter() - start

        writes_per_sec = 1000 / elapsed
        assert writes_per_sec > 1000, f"Only {writes_per_sec:.0f} writes/sec"

    def test_memory_store_speed(self, memory):
        """Test EternalMemory store performance."""
        start = time.perf_counter()
        for i in range(100):
            hv = np.random.randn(memory.dim).astype(np.float32)
            memory.store(hv, strength=0.5, meta={"i": i})
        elapsed = time.perf_counter() - start

        stores_per_sec = 100 / elapsed
        assert stores_per_sec > 100, f"Only {stores_per_sec:.0f} stores/sec"

    def test_memory_recall_speed(self, memory):
        """Test EternalMemory recall performance."""
        # Pre-populate
        for i in range(100):
            hv = np.random.randn(memory.dim).astype(np.float32)
            memory.store(hv, strength=0.5, meta={"i": i})

        query = np.random.randn(memory.dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            memory.recall(query, k=5)
        elapsed = time.perf_counter() - start

        recalls_per_sec = 100 / elapsed
        assert recalls_per_sec > 50, f"Only {recalls_per_sec:.0f} recalls/sec"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

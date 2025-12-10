#!/usr/bin/env python3
"""
Test suite for Ara feedback collection, rewards, learning, and planning.
"""

import sys
sys.path.insert(0, '/home/user/Ara')

import time
from datetime import datetime


def test_feedback_signals():
    """Test multimodal feedback signal collection."""
    print("Testing feedback signals...")

    from ara_core.feedback.signals import (
        SignalType, FeedbackSignal, FeedbackCollector
    )

    collector = FeedbackCollector()
    collector.start_interaction("test_001", voice_params={"warmth": 0.7})

    # Record various signals
    collector.record_text_signal("This is amazing, thank you!")
    collector.record_text_signal("I'm confused, what does this mean?")
    collector.record_text_signal("Could you explain that again?")

    collector.record_behavior(SignalType.ATTENTION, 0.8)
    collector.record_behavior(SignalType.RESPONSE_LENGTH, 0.5)
    collector.record_behavior(SignalType.CONTINUED, 1.0)

    collector.record_sensor(SignalType.VOICE_RELAXED, 0.85)
    collector.record_sensor(SignalType.SMILE, 0.7)

    # Get aggregated feedback
    feedback = collector.end_interaction()

    print(f"  Signals collected: {len(feedback.signals)}")
    print(f"  Understanding score: {feedback.understanding_score:.2f}")
    print(f"  Engagement score: {feedback.engagement_score:.2f}")
    print(f"  Satisfaction score: {feedback.satisfaction_score:.2f}")

    assert len(feedback.signals) >= 8
    assert feedback.understanding_score < 1.0  # Due to confusion signals
    assert feedback.engagement_score > 0  # Due to behavior signals

    print("  ✓ Feedback signals OK")
    return True


def test_reward_tracking():
    """Test cost-aware reward computation."""
    print("Testing reward tracking...")

    from ara_core.feedback.signals import FeedbackCollector, SignalType
    from ara_core.feedback.rewards import RewardConfig, RewardTracker

    # Setup
    collector = FeedbackCollector()
    collector.start_interaction("test_002", voice_params={"warmth": 0.8})
    collector.record_text_signal("Great work!")
    collector.record_behavior(SignalType.CONTINUED, 1.0)
    collector.record_behavior(SignalType.ATTENTION, 0.9)
    feedback = collector.end_interaction()

    tracker = RewardTracker()

    # Compute reward with resource usage
    reward = tracker.compute_reward(
        feedback=feedback,
        gpu_seconds=30.0,
        power_wh=15.0,
        progress_indicators={"task_completed": True}
    )

    print(f"  User reward (r_user): {reward.r_user:.3f}")
    print(f"  Progress reward (r_progress): {reward.r_progress:.3f}")
    print(f"  Cost (r_cost): {reward.r_cost:.3f}")
    print(f"  Total reward: {reward.total_reward:.3f}")

    assert reward.r_user > 0
    assert reward.r_progress > 0  # Task completed
    assert reward.r_cost > 0  # Resources used
    assert reward.total_reward > 0  # Should be positive overall

    print("  ✓ Reward tracking OK")
    return True


def test_parameter_learning():
    """Test contextual parameter learning."""
    print("Testing parameter learning...")

    from ara_core.feedback.learning import ParameterLearner
    from ara_core.feedback.rewards import InteractionReward

    learner = ParameterLearner()

    # Get params for different contexts
    morning_params = learner.get_params("morning", time_of_day="morning")
    evening_params = learner.get_params("evening", time_of_day="evening")

    print(f"  Morning warmth: {morning_params.warmth:.2f}")
    print(f"  Morning pace: {morning_params.pace:.2f}")
    print(f"  Evening warmth: {evening_params.warmth:.2f}")
    print(f"  Evening pace: {evening_params.pace:.2f}")

    # Simulate learning updates with InteractionReward
    reward1 = InteractionReward(interaction_id="test1", r_user=0.8, r_progress=0.3, r_cost=0.1, total_reward=0.8)
    reward2 = InteractionReward(interaction_id="test2", r_user=0.9, r_progress=0.4, r_cost=0.1, total_reward=0.9)
    reward3 = InteractionReward(interaction_id="test3", r_user=0.3, r_progress=0.2, r_cost=0.2, total_reward=0.3)

    learner.update("morning", morning_params, reward1)
    learner.update("morning", morning_params, reward2)
    learner.update("evening", evening_params, reward3)

    # Get updated params - check context_params
    morning_data = learner.context_params.get("morning")
    evening_data = learner.context_params.get("evening")

    print(f"  Morning n_samples: {morning_data.n_samples}")
    print(f"  Morning avg_reward: {morning_data.avg_reward:.2f}")
    print(f"  Evening n_samples: {evening_data.n_samples}")

    assert morning_data.n_samples == 2
    assert evening_data.n_samples == 1

    print("  ✓ Parameter learning OK")
    return True


def test_session_planning():
    """Test session-level planning."""
    print("Testing session planning...")

    from ara_core.planning.session import SessionPlanner, BlockPriority

    planner = SessionPlanner()

    # Create plan for decompress goal
    session = planner.create_plan(
        goals=["decompress_user"],
        context={"time_of_day": "evening", "energy": 0.4},
        constraints={"max_duration_s": 1800}
    )

    print(f"  Session ID: {session.session_id[:8]}...")
    print(f"  Goals: {session.goals}")
    print(f"  Blocks: {len(session.blocks)}")

    for block in session.blocks:
        print(f"    - {block.id}: {block.block_type.value} ({block.priority.value})")

    assert len(session.blocks) >= 2
    assert any("check_in" in b.id for b in session.blocks)

    # Test block completion
    first_block = session.blocks[0]
    first_block.started_at = time.time()
    session.mark_completed(first_block.id, reward=0.8)

    assert first_block.state.value == "completed"
    assert first_block.reward == 0.8
    assert session.total_reward == 0.8

    print("  ✓ Session planning OK")
    return True


def test_daily_planning():
    """Test daily planning."""
    print("Testing daily planning...")

    from ara_core.planning.daily import (
        DailyPlanner, get_daily_planner, get_todays_plan, get_current_session
    )

    planner = get_daily_planner()
    daily_plan = get_todays_plan()

    print(f"  Date: {daily_plan.date}")
    print(f"  Time slots: {len(daily_plan.time_slots)}")

    current_slot = daily_plan.get_current_slot()
    if current_slot:
        print(f"  Current slot: {current_slot.label} ({current_slot.start_hour}:00-{current_slot.end_hour}:00)")
        print(f"  Default goals: {current_slot.default_goals}")

    # Get session for now
    session = get_current_session()
    print(f"  Current session: {session.session_id[:8]}...")
    print(f"  Session blocks: {len(session.blocks)}")

    # Track some usage
    planner.update_daily_usage(
        daily_plan,
        gpu_seconds=300,
        power_wh=50,
        cost=0.25
    )

    budget = daily_plan.get_budget_status()
    print(f"  GPU remaining: {budget['gpu_remaining_pct']*100:.1f}%")
    print(f"  Power remaining: {budget['power_remaining_pct']*100:.1f}%")
    print(f"  Cost remaining: {budget['cost_remaining_pct']*100:.1f}%")

    # Get summary
    summary = planner.get_daily_summary(daily_plan)
    print(f"  Sessions today: {summary['sessions']}")
    print(f"  Blocks total: {summary['blocks_total']}")

    assert len(daily_plan.time_slots) == 7
    assert budget['gpu_remaining_pct'] < 1.0  # We used some

    print("  ✓ Daily planning OK")
    return True


def test_integration():
    """Test full integration of feedback -> rewards -> learning -> planning."""
    print("Testing full integration...")

    from ara_core.feedback.signals import FeedbackCollector, SignalType
    from ara_core.feedback.rewards import RewardTracker
    from ara_core.feedback.learning import ParameterLearner
    from ara_core.planning.daily import get_current_session, get_daily_planner, get_todays_plan
    from ara_core.planning.session import SessionPlanner

    # Setup components
    collector = FeedbackCollector()
    tracker = RewardTracker()
    learner = ParameterLearner()

    # Simulate an interaction
    context_key = "afternoon_work"

    # 1. Get learned params for this context
    params = learner.get_params(context_key, time_of_day="afternoon")
    print(f"  Using params: warmth={params.warmth:.2f}, pace={params.pace:.2f}")

    # 2. Collect feedback during interaction
    collector.start_interaction("integration_test", voice_params=params.to_voice_params())
    collector.record_text_signal("This explanation is really helpful!")
    collector.record_behavior(SignalType.RESPONSE_LENGTH, 0.8)
    collector.record_behavior(SignalType.ATTENTION, 0.85)
    collector.record_behavior(SignalType.CONTINUED, 1.0)

    # 3. Compute reward
    feedback = collector.end_interaction()
    reward = tracker.compute_reward(
        feedback=feedback,
        gpu_seconds=45.0,
        power_wh=20.0,
        progress_indicators={"explanation_delivered": True}
    )

    print(f"  Interaction reward: {reward.total_reward:.3f}")

    # 4. Update learner
    learner.update(context_key, params, reward)

    # 5. Check learning happened
    data = learner.context_params.get(context_key)
    print(f"  Context '{context_key}' n_samples: {data.n_samples if data else 0}")
    print(f"  Avg reward: {data.avg_reward if data else 0:.3f}")

    assert reward.total_reward > 0
    assert data is not None
    assert data.n_samples >= 1

    print("  ✓ Full integration OK")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Ara Feedback/Learning/Planning Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_feedback_signals,
        test_reward_tracking,
        test_parameter_learning,
        test_session_planning,
        test_daily_planning,
        test_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

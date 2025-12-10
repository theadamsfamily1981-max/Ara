#!/usr/bin/env python3
"""
Test suite for A-KTP (Allegory-based Knowledge Transfer Protocol).
"""

import sys
sys.path.insert(0, '/home/user/Ara')


def test_allegory_generation():
    """Test AGM - Allegory Generation Module."""
    print("Testing AGM (Allegory Generation)...")

    from ara_core.aktp import AllegoryGenerator, AllegoryArchetype

    agm = AllegoryGenerator()

    # Test microservices allegory
    allegory = agm.generate(
        problem="Should we migrate to microservices?",
        domain="infrastructure"
    )

    print(f"  Title: {allegory.title}")
    print(f"  Archetype: {allegory.archetype.value}")
    print(f"  Mappings: {len(allegory.mappings)}")
    print(f"  Transfer confidence: {allegory.transfer_confidence:.2f}")

    assert allegory.title == "The Mighty River and the River Network"
    assert allegory.archetype == AllegoryArchetype.RIVER_NETWORK
    assert len(allegory.mappings) >= 4
    assert allegory.moral != ""

    # Test GPU scaling allegory
    gpu_allegory = agm.generate(
        problem="How do we scale our GPU cluster?",
        domain="compute"
    )

    assert gpu_allegory.archetype == AllegoryArchetype.HIVE_EXPANSION

    print("  ✓ AGM OK")
    return True


def test_bot_debate():
    """Test 5-bot debate system."""
    print("Testing Bot Debate System...")

    from ara_core.aktp import DebatePanel, BotPersonality

    panel = DebatePanel()

    rounds = panel.debate(
        prompt="Should we invest in new hardware or optimize existing?",
        allegory="The Growing Hive",
        max_rounds=2
    )

    print(f"  Rounds completed: {len(rounds)}")
    print(f"  Bots participating: {len(panel.bots)}")

    last_round = rounds[-1]
    print(f"  Last round responses: {len(last_round.responses)}")
    print(f"  Consensus score: {last_round.consensus_score:.2f}")

    # Check all 5 personalities responded
    personalities = {r.personality for r in last_round.responses}
    assert len(personalities) == 5

    # Synthesize
    synthesis = panel.synthesize(rounds)
    print(f"  Emergent insights: {len(synthesis['emergent_insights'])}")
    print(f"  Recommendation: {synthesis['recommendation'][:50]}...")

    assert "recommendation" in synthesis
    assert len(synthesis["positions"]) == 5

    print("  ✓ Bot Debate OK")
    return True


def test_constraint_refinement():
    """Test ACT - Adversarial Constraint Transfer."""
    print("Testing ACT (Constraint Refinement)...")

    from ara_core.aktp import (
        AllegoryGenerator, AdversarialConstraintTransfer, ConstraintType
    )

    # Generate allegory
    agm = AllegoryGenerator()
    allegory = agm.generate("Database migration to cloud", "infrastructure")

    # Refine constraints
    act = AdversarialConstraintTransfer()
    constraints = act.refine(allegory, max_rounds=2)

    print(f"  Constraint set ID: {constraints.set_id}")
    print(f"  Total constraints: {len(constraints.constraints)}")
    print(f"  Debate rounds: {constraints.total_debate_rounds}")
    print(f"  Consensus score: {constraints.consensus_score:.2f}")
    print(f"  Ethical score: {constraints.ethical_score:.2f}")

    # Check constraint types
    hard = constraints.get_hard_constraints()
    soft = constraints.get_soft_constraints()
    print(f"  Hard constraints: {len(hard)}")
    print(f"  Soft constraints: {len(soft)}")

    assert len(constraints.constraints) >= 1
    assert constraints.ethical_score > 0

    print("  ✓ ACT OK")
    return True


def test_reward_shaping():
    """Test DRSE - Dynamic Reward Shaping Engine."""
    print("Testing DRSE (Reward Shaping)...")

    from ara_core.aktp import (
        AllegoryGenerator, AdversarialConstraintTransfer,
        DynamicRewardShaper
    )

    # Setup
    agm = AllegoryGenerator()
    allegory = agm.generate("Optimize model training", "ml")

    act = AdversarialConstraintTransfer()
    constraints = act.refine(allegory, max_rounds=1)

    # Shape rewards
    drse = DynamicRewardShaper()
    base_rewards = [0.5, 0.7, 0.3, 0.6, 0.4]

    update = drse.shape_rewards(constraints, base_rewards, iteration=1)

    print(f"  Update ID: {update.update_id}")
    print(f"  PTE: {update.pte:.3f}")
    print(f"  Converged: {update.converged}")
    print(f"  Shaped rewards: {len(update.shaped_rewards)}")

    # Check shaped rewards
    for i, shaped in enumerate(update.shaped_rewards):
        print(f"    R{i}: base={shaped.base_reward:.2f} → shaped={shaped.shaped_reward:.2f}")

    assert len(update.shaped_rewards) == 5
    assert update.pte > 0

    # Check convergence report
    report = drse.get_convergence_report()
    print(f"  Total PTE: {report['total_pte']:.3f}")

    print("  ✓ DRSE OK")
    return True


def test_full_protocol():
    """Test complete A-KTP protocol execution."""
    print("Testing Full A-KTP Protocol...")

    from ara_core.aktp import run_aktp

    result = run_aktp(
        problem="Should we migrate monolith to microservices at 10x growth?",
        domain="infrastructure",
        max_cycles=3
    )

    print(f"  Protocol ID: {result.protocol_id}")
    print(f"  Cycles: {len(result.cycles)}")
    print(f"  Total PTE: {result.total_pte:.2f}")
    print(f"  Converged: {result.converged}")
    print(f"  Hypothetical: {result.has_hypothetical}")
    print(f"  Duration: {result.total_duration_s:.2f}s")
    print(f"  Recommendation: {result.final_recommendation[:60]}...")

    assert len(result.cycles) >= 1
    assert result.total_pte > 0
    assert result.final_recommendation != ""

    # Check cycles
    for i, cycle in enumerate(result.cycles):
        print(f"    Cycle {i+1}: PTE={cycle.pte:.2f}, allegory={cycle.allegory.title[:30]}")

    print("  ✓ Full Protocol OK")
    return True


def test_knowledge_transfer():
    """Test knowledge transfer between problems."""
    print("Testing Knowledge Transfer...")

    from ara_core.aktp import AKTPProtocol

    protocol = AKTPProtocol()

    # Solve source problem
    source = protocol.execute(
        problem="How to scale our database?",
        domain="database",
        max_cycles=2
    )

    # Transfer to target
    target = protocol.transfer(
        source,
        target_problem="How to scale our message queue?",
        target_domain="messaging"
    )

    print(f"  Source PTE: {source.total_pte:.2f}")
    print(f"  Target PTE: {target.total_pte:.2f}")
    print(f"  Transfer indicator: {'[Transfer' in target.final_recommendation}")

    assert "[Transfer" in target.final_recommendation

    print("  ✓ Knowledge Transfer OK")
    return True


def test_hypothetical_detection():
    """Test detection of hypothetical/unverified domains."""
    print("Testing Hypothetical Detection...")

    from ara_core.aktp import AllegoryGenerator

    agm = AllegoryGenerator()

    # Test with hypothetical domain
    allegory = agm.generate(
        problem="Can we solve P vs NP using quantum allegory?",
        domain="theoretical_cs"
    )

    print(f"  Is hypothetical: {allegory.is_hypothetical}")
    print(f"  Bias warnings: {allegory.bias_warnings}")

    assert allegory.is_hypothetical == True
    assert len(allegory.bias_warnings) > 0

    # Test normal domain
    normal = agm.generate(
        problem="How to optimize cache?",
        domain="systems"
    )

    assert normal.is_hypothetical == False

    print("  ✓ Hypothetical Detection OK")
    return True


def main():
    """Run all A-KTP tests."""
    print("=" * 60)
    print("A-KTP Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_allegory_generation,
        test_bot_debate,
        test_constraint_refinement,
        test_reward_shaping,
        test_full_protocol,
        test_knowledge_transfer,
        test_hypothetical_detection,
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

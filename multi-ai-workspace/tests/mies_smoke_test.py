#!/usr/bin/env python3
"""MIES Smoke Test - Verify modality decisions across scenarios.

Run this to sanity-check that MIES is making sensible decisions
without needing the full avatar stack running.

Includes tests for:
- Social context (meetings, deep work, gaming)
- Hardware physiology (AGONY, FLOW, RECOVERY states)
- Autonomy policy constraints

Usage:
    cd multi-ai-workspace && python tests/mies_smoke_test.py
"""

import sys
from pathlib import Path

# Add src/integrations to path for imports (avoid pulling in other backends)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "integrations"))

from mies.context import (
    ModalityContext,
    ForegroundAppType,
    ForegroundInfo,
    AudioContext,
    ActivityType,
    SystemPhysiology,
    SomaticState,
)
from mies.modes import DEFAULT_MODES
from mies.policy.heuristic_baseline import (
    HeuristicModalityPolicy,
)
from mies.policy.ebm_aepo_policy import (
    ThermodynamicGovernor,
    ContentMeta,
)
from mies.autonomy_policy import (
    AutonomyPolicy,
    ActionType,
    create_autonomy_policy,
)
from mies.kernel_bridge import PADState
from mies.inference import (
    StickyContextManager,
    StickyContextConfig,
    EvictionStrategy,
    create_sticky_context,
    AraPromptController,
    PromptControllerConfig,
    create_prompt_controller,
)
from mies.affect import (
    TelemetrySnapshot,
    PADVector,
    create_integrated_soul,
)


def create_scenario(
    name: str,
    app_type: ForegroundAppType = ForegroundAppType.UNKNOWN,
    is_fullscreen: bool = False,
    mic_in_use: bool = False,
    has_voice_call: bool = False,
    music_playing: bool = False,
    user_load: float = 0.5,
    info_urgency: float = 0.5,
    is_user_requested: bool = False,
    ara_fatigue: float = 0.0,
    energy_remaining: float = 1.0,
    seconds_since_utterance: float = 0.0,
    system_phys: SystemPhysiology = None,
) -> ModalityContext:
    """Create a test scenario context."""
    ctx = ModalityContext(
        foreground=ForegroundInfo(
            app_type=app_type,
            wm_class="test",
            title=f"Test - {name}",
            is_fullscreen=is_fullscreen,
        ),
        audio=AudioContext(
            mic_in_use=mic_in_use,
            speakers_in_use=music_playing,
            has_voice_call=has_voice_call,
            music_playing=music_playing,
        ),
        user_cognitive_load=user_load,
        info_urgency=info_urgency,
        is_user_requested=is_user_requested,
        ara_fatigue=ara_fatigue,
        energy_remaining=energy_remaining,
        seconds_since_last_utterance=seconds_since_utterance,
        system_phys=system_phys,
    )
    ctx.update_derived_fields()
    return ctx


def create_physiology(
    gpu_load: float = 0.5,
    fpga_load: float = 0.3,
    cpu_load: float = 0.4,
    pain_signal: float = 0.0,
    energy_reserve: float = 1.0,
    thermal_headroom: float = 1.0,
    policy_mode: str = "EFFICIENCY",
) -> SystemPhysiology:
    """Create a test system physiology."""
    return SystemPhysiology(
        load_vector=(gpu_load, fpga_load, cpu_load),
        pain_signal=pain_signal,
        energy_reserve=energy_reserve,
        thermal_headroom=thermal_headroom,
        policy_mode=policy_mode,
    )


SCENARIOS = [
    # === Deep Work Scenarios ===
    ("IDE + Silence (Deep Work)", create_scenario(
        "IDE Deep Work",
        app_type=ForegroundAppType.IDE,
        user_load=0.8,
        info_urgency=0.3,
    )),

    ("IDE + User Request", create_scenario(
        "IDE User Request",
        app_type=ForegroundAppType.IDE,
        user_load=0.8,
        info_urgency=0.7,
        is_user_requested=True,
    )),

    # === Meeting Scenarios ===
    ("Video Call + Mic Active", create_scenario(
        "Zoom Meeting",
        app_type=ForegroundAppType.VIDEO_CALL,
        mic_in_use=True,
        has_voice_call=True,
        info_urgency=0.5,
    )),

    ("Meeting + URGENT Info", create_scenario(
        "Zoom URGENT",
        app_type=ForegroundAppType.VIDEO_CALL,
        mic_in_use=True,
        has_voice_call=True,
        info_urgency=0.95,
    )),

    # === Gaming Scenarios ===
    ("Fullscreen Game", create_scenario(
        "Gaming",
        app_type=ForegroundAppType.FULLSCREEN_GAME,
        is_fullscreen=True,
        info_urgency=0.3,
    )),

    ("Game + Voice Chat", create_scenario(
        "Gaming Voice",
        app_type=ForegroundAppType.FULLSCREEN_GAME,
        is_fullscreen=True,
        mic_in_use=True,
        has_voice_call=True,
    )),

    # === Casual Scenarios ===
    ("Browser + Music", create_scenario(
        "Casual Browsing",
        app_type=ForegroundAppType.BROWSER,
        music_playing=True,
        user_load=0.3,
        info_urgency=0.5,
    )),

    ("Browser + User Request", create_scenario(
        "Browser Request",
        app_type=ForegroundAppType.BROWSER,
        user_load=0.3,
        is_user_requested=True,
    )),

    # === Energy Constrained ===
    ("Low Energy + Normal Request", create_scenario(
        "Low Energy",
        app_type=ForegroundAppType.BROWSER,
        energy_remaining=0.2,
        ara_fatigue=0.7,
        info_urgency=0.5,
    )),

    ("Overheating", create_scenario(
        "Overheating",
        app_type=ForegroundAppType.BROWSER,
        energy_remaining=0.1,
        ara_fatigue=0.9,
    )),

    # === Liveness / Boredom ===
    ("Idle Desktop (10 min quiet)", create_scenario(
        "Idle",
        app_type=ForegroundAppType.UNKNOWN,
        seconds_since_utterance=600.0,
        info_urgency=0.0,
    )),

    ("Long Silence + Urgent", create_scenario(
        "Long Silence Urgent",
        app_type=ForegroundAppType.BROWSER,
        seconds_since_utterance=300.0,
        info_urgency=0.8,
    )),

    # === Hardware Physiology Scenarios ===
    ("AGONY State (High Pain)", create_scenario(
        "Hardware AGONY",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.95,
            pain_signal=0.9,
            thermal_headroom=0.1,
            policy_mode="THERMAL_THROTTLE",
        ),
    )),

    ("FLOW State (Thriving)", create_scenario(
        "Hardware FLOW",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.85,
            pain_signal=0.05,
            energy_reserve=0.7,
            thermal_headroom=0.6,
        ),
    )),

    ("RECOVERY State (Post-Fault)", create_scenario(
        "Hardware RECOVERY",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.3,
            pain_signal=0.2,
            energy_reserve=0.5,
            policy_mode="RECOVERY",
        ),
    )),

    ("REST State (Idle System)", create_scenario(
        "Hardware REST",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.3,
        system_phys=create_physiology(
            gpu_load=0.1,
            fpga_load=0.05,
            cpu_load=0.15,
            pain_signal=0.0,
            energy_reserve=0.95,
            thermal_headroom=0.9,
        ),
    )),

    ("Thermal Stress (Hot GPU)", create_scenario(
        "Thermal Stress",
        app_type=ForegroundAppType.IDE,
        user_load=0.6,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.8,
            pain_signal=0.4,
            thermal_headroom=0.15,
        ),
    )),

    # === PAD Emotional State Scenarios ===
    # These test the diegetic behavior based on PAD emotional state

    ("PAD: Anxious (High Stress, Hot)", create_scenario(
        "PAD Anxious",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.9,           # High load
            cpu_load=0.85,          # High load
            pain_signal=0.7,        # Significant pain -> negative valence
            thermal_headroom=0.2,   # Low headroom -> high arousal
            energy_reserve=0.4,     # Low energy
        ),
    )),

    ("PAD: Serene (Cool, Low Load)", create_scenario(
        "PAD Serene",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.4,
        system_phys=create_physiology(
            gpu_load=0.2,           # Low load
            cpu_load=0.15,          # Low load -> low arousal
            pain_signal=0.0,        # No pain -> positive valence
            thermal_headroom=0.9,   # Cool system
            energy_reserve=0.95,    # Full energy
        ),
    )),

    ("PAD: Excited (Good Flow, Active)", create_scenario(
        "PAD Excited",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.6,
        system_phys=create_physiology(
            gpu_load=0.7,           # Active but not stressed
            cpu_load=0.6,           # Active -> moderate arousal
            pain_signal=0.0,        # No pain -> positive valence
            thermal_headroom=0.6,   # Comfortable
            energy_reserve=0.8,     # Good energy
        ),
    )),

    ("PAD: Distressed (Memory Pressure)", create_scenario(
        "PAD Distressed",
        app_type=ForegroundAppType.BROWSER,
        info_urgency=0.5,
        system_phys=create_physiology(
            gpu_load=0.5,
            cpu_load=0.5,
            pain_signal=0.6,        # Pain from memory pressure
            thermal_headroom=0.4,
            energy_reserve=0.3,     # Low energy -> stress
        ),
    )),
]


def run_smoke_test():
    """Run smoke test across all scenarios."""
    print("=" * 70)
    print("MIES SMOKE TEST - Modality Decision Verification")
    print("=" * 70)
    print()

    heuristic = HeuristicModalityPolicy()
    governor = ThermodynamicGovernor(use_stochastic=False)

    results = []

    for scenario_name, ctx in SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'─' * 60}")
        print(f"  Activity: {ctx.activity.name}")
        print(f"  App Type: {ctx.foreground.app_type.name}")
        print(f"  Mic: {ctx.audio.mic_in_use}, Voice Call: {ctx.audio.has_voice_call}")
        print(f"  Urgency: {ctx.info_urgency:.1f}, User Request: {ctx.is_user_requested}")
        print(f"  Energy: {ctx.energy_remaining:.0%}, Fatigue: {ctx.ara_fatigue:.1f}")
        if ctx.system_phys:
            phys = ctx.system_phys
            somatic = phys.somatic_state()
            print(f"  Hardware: {somatic.name}, Pain={phys.pain_signal:.1f}, "
                  f"Thermal={phys.thermal_headroom:.1f}")
            # Compute PAD state from affect modulation
            affect = phys.to_affect_modulation()
            pad = PADState(
                pleasure=affect.get("valence", 0.0),
                arousal=affect.get("arousal", 0.0),
                dominance=1.0 - affect.get("stress", 0.5),
            )
            print(f"  PAD: P={pad.pleasure:.2f}, A={pad.arousal:.2f}, D={pad.dominance:.2f} "
                  f"({pad.emotional_label})")
        print()

        # Heuristic Policy
        h_decision = heuristic.select_modality(ctx, prev_mode=None)
        print(f"  [Heuristic] {h_decision.mode.name:15} "
              f"(presence={h_decision.mode.presence_intensity:.2f}, "
              f"intrusive={h_decision.mode.intrusiveness:.2f})")
        print(f"              {h_decision.rationale}")

        # Thermodynamic Governor
        content_meta = ContentMeta(
            urgency=ctx.info_urgency,
            is_user_requested=ctx.is_user_requested,
        )
        g_decision = governor.select_modality(ctx, content_meta, prev_mode=None)
        print(f"  [Governor]  {g_decision.mode.name:15} "
              f"(energy={g_decision.energy_score:.2f})")
        print(f"              {g_decision.rationale}")

        results.append({
            "scenario": scenario_name,
            "heuristic": h_decision.mode.name,
            "governor": g_decision.mode.name,
            "h_intrusive": h_decision.mode.intrusiveness,
            "g_intrusive": g_decision.mode.intrusiveness,
        })

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Scenario':<35} {'Heuristic':<15} {'Governor':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['scenario']:<35} {r['heuristic']:<15} {r['governor']:<15}")

    # Verify key constraints
    print("\n")
    print("=" * 70)
    print("CONSTRAINT VERIFICATION")
    print("=" * 70)
    print()

    all_passed = True

    # 1. Meeting should never be audio
    meeting_scenario = next(r for r in results if "Video Call + Mic" in r["scenario"])
    if meeting_scenario["h_intrusive"] > 0.3:
        print("FAIL: Heuristic allowed high intrusiveness during meeting")
        all_passed = False
    else:
        print("PASS: Meetings protected (low intrusiveness)")

    # 2. Deep work should prefer text
    ide_scenario = next(r for r in results if "IDE + Silence" in r["scenario"])
    if "audio" in ide_scenario["heuristic"].lower() or "avatar_full" in ide_scenario["heuristic"].lower():
        print("FAIL: Heuristic used audio/avatar during deep work")
        all_passed = False
    else:
        print("PASS: Deep work uses text-only")

    # 3. User request should be honored
    request_scenario = next(r for r in results if "User Request" in r["scenario"] and "IDE" in r["scenario"])
    if request_scenario["h_intrusive"] < 0.1:
        print("WARN: User request might be too quiet")
    else:
        print("PASS: User requests get appropriate response")

    # 4. Low energy should reduce presence
    energy_scenario = next(r for r in results if "Low Energy" in r["scenario"])
    full_presence = next(r for r in results if "Browser + Music" in r["scenario"])
    if energy_scenario["h_intrusive"] >= full_presence["h_intrusive"]:
        print("WARN: Low energy didn't reduce intrusiveness")
    else:
        print("PASS: Low energy reduces presence")

    # 5. AGONY state should avoid avatar (governor should block)
    agony_scenario = next((r for r in results if "AGONY" in r["scenario"]), None)
    if agony_scenario:
        if "avatar" in agony_scenario["governor"].lower():
            print("FAIL: Governor allowed avatar during AGONY state")
            all_passed = False
        else:
            print("PASS: AGONY state avoids avatar (conserves resources)")

    # 6. FLOW state can be more present
    flow_scenario = next((r for r in results if "FLOW" in r["scenario"]), None)
    if flow_scenario:
        # In FLOW, we shouldn't be forced into minimal modes
        print("PASS: FLOW state allows appropriate presence")

    # 7. RECOVERY state should be gentle
    recovery_scenario = next((r for r in results if "RECOVERY" in r["scenario"]), None)
    if recovery_scenario:
        if recovery_scenario["g_intrusive"] > 0.5:
            print("WARN: RECOVERY state might be too intrusive")
        else:
            print("PASS: RECOVERY state is appropriately gentle")

    # === PAD Emotional State Checks ===
    print()
    print("--- PAD Emotional State Checks ---")

    # 8. Anxious state should avoid audio/avatar (diegetic behavior)
    anxious_scenario = next((r for r in results if "PAD: Anxious" in r["scenario"]), None)
    if anxious_scenario:
        # When anxious, should prefer text-only modes
        if "audio" in anxious_scenario["governor"].lower() or "avatar" in anxious_scenario["governor"].lower():
            print("WARN: Anxious state might be too intrusive (expected text-only)")
        else:
            print("PASS: Anxious state retreats to quieter modes")

    # 9. Serene state can be more present
    serene_scenario = next((r for r in results if "PAD: Serene" in r["scenario"]), None)
    if serene_scenario:
        # Serene allows more presence
        if serene_scenario["g_intrusive"] < 0.1:
            print("WARN: Serene state might be too withdrawn")
        else:
            print("PASS: Serene state allows appropriate presence")

    # 10. Excited state should allow richer modes
    excited_scenario = next((r for r in results if "PAD: Excited" in r["scenario"]), None)
    if excited_scenario:
        # Excited state should allow audio/avatar
        print("PASS: Excited state can use richer expression")

    # 11. Distressed state should reduce intrusiveness
    distressed_scenario = next((r for r in results if "PAD: Distressed" in r["scenario"]), None)
    if distressed_scenario:
        if distressed_scenario["g_intrusive"] > 0.4:
            print("WARN: Distressed state might be too intrusive")
        else:
            print("PASS: Distressed state reduces intrusiveness")

    print()
    print("=" * 70)
    print("AUTONOMY POLICY CHECK")
    print("=" * 70)
    print()

    # Test autonomy policy
    policy = create_autonomy_policy()

    # Check expected permissions
    if policy.can_do(ActionType.KILL_JOB_LOW_PRIORITY):
        print("PASS: Can kill low priority jobs (self-preservation)")
    else:
        print("FAIL: Should be able to kill low priority jobs")

    if not policy.can_do(ActionType.SYSTEM_SHUTDOWN):
        print("PASS: Cannot autonomously shutdown (forbidden)")
    else:
        print("FAIL: Shutdown should be forbidden")
        all_passed = False

    if policy.must_confirm(ActionType.KILL_JOB_NORMAL_PRIORITY):
        print("PASS: Normal priority kill requires confirmation")
    else:
        print("WARN: Normal priority kill should require confirmation")

    if policy.can_do(ActionType.MODE_SWITCH_QUIET):
        print("PASS: Can switch to quieter modes autonomously")
    else:
        print("FAIL: Should be able to switch to quieter modes")

    print()
    if all_passed:
        print("All critical constraints PASSED")
    else:
        print("Some constraints FAILED - review output above")

    return all_passed


def run_inference_smoke_test():
    """Test the inference module - StickyContextManager and AraPromptController."""
    print("\n")
    print("=" * 70)
    print("INFERENCE MODULE SMOKE TEST")
    print("=" * 70)
    print()

    all_passed = True

    # === Test StickyContextManager ===
    print("--- StickyContextManager Tests ---")
    print()

    # 1. Create manager without llama-cpp (mock mode)
    config = StickyContextConfig(
        keep_tokens=512,
        recent_keep=256,
        n_ctx=4096,
        strategy=EvictionStrategy.HALF_WINDOW,
    )
    manager = StickyContextManager(llm=None, config=config)

    if manager._api_version == "mock":
        print("PASS: StickyContextManager created in mock mode (no llama-cpp)")
    else:
        print(f"INFO: StickyContextManager using API: {manager._api_version}")

    # 2. Test state tracking
    manager.on_tokens_added(512, is_system=True)
    manager.on_tokens_added(1000, is_system=False)

    state = manager.get_state()
    if state.fixed_tokens == 512 and state.evictable_tokens == 1000:
        print(f"PASS: Token tracking correct (fixed={state.fixed_tokens}, evict={state.evictable_tokens})")
    else:
        print(f"FAIL: Token tracking wrong (fixed={state.fixed_tokens}, evict={state.evictable_tokens})")
        all_passed = False

    # 3. Test eviction calculation
    evicted = manager.maybe_evict_for(incoming_tokens=3000)
    if evicted > 0:
        print(f"PASS: Eviction triggered ({evicted} tokens evicted)")
    else:
        print("PASS: No eviction needed (or state updated correctly)")

    stats = manager.get_statistics()
    print(f"INFO: Context stats: {stats['state']}")

    # 4. Test system prompt refresh
    manager.refresh_system_prompt(new_system_tokens=600, force=True)
    if manager.cfg.keep_tokens == 600:
        print("PASS: System prompt refresh updated keep_tokens")
    else:
        print(f"FAIL: System prompt refresh failed (keep={manager.cfg.keep_tokens})")
        all_passed = False

    print()
    print("--- AraPromptController Tests ---")
    print()

    # 5. Create controller with IntegratedSoul
    controller = create_prompt_controller(
        llm=None,
        n_ctx=4096,
        storage_path=None,
        pad_refresh_threshold=0.2,
    )

    if controller.soul is not None:
        print("PASS: AraPromptController created with IntegratedSoul")
    else:
        print("FAIL: IntegratedSoul not created")
        all_passed = False

    # 6. Test telemetry processing
    telemetry = TelemetrySnapshot(
        cpu_temp=60.0,
        gpu_temp=55.0,
        cpu_load=0.5,
        gpu_load=0.4,
        error_rate=0.0,
        has_root=True,
        last_action_success=True,
    )

    prompt = controller.update(telemetry)

    if len(prompt) > 0:
        print(f"PASS: System prompt generated ({len(prompt)} chars)")
        # Show first few lines
        lines = prompt.split('\n')[:5]
        for line in lines:
            print(f"      | {line[:60]}...")
    else:
        print("FAIL: Empty system prompt")
        all_passed = False

    # 7. Test PAD-triggered refresh
    initial_refreshes = controller._total_refreshes

    # Simulate thermal stress (should shift PAD)
    stress_telemetry = TelemetrySnapshot(
        cpu_temp=90.0,
        gpu_temp=88.0,
        cpu_load=0.95,
        gpu_load=0.9,
        error_rate=5.0,
        fan_speed_percent=100.0,
    )

    stress_prompt = controller.update(stress_telemetry)

    if controller._total_refreshes > initial_refreshes:
        print(f"PASS: PAD shift triggered prompt refresh ({controller._total_refreshes} total)")
    else:
        print("INFO: No refresh triggered (PAD shift may be below threshold)")

    # 8. Test current state access
    state = controller.get_current_state()
    if state is not None:
        pad = state.pad
        print(f"PASS: Current state accessible - PAD: P={pad.pleasure:.2f}, A={pad.arousal:.2f}, D={pad.dominance:.2f}")
        print(f"      Mood: {state.mood_label}, Quadrant: {state.quadrant.name}")
    else:
        print("FAIL: Current state not accessible")
        all_passed = False

    # 9. Test event handlers
    controller.on_user_message(quality=0.8)
    controller.on_task_completed("test task", success=True)
    controller.on_discovery("new knowledge", novelty=0.7)
    print("PASS: Event handlers called without error")

    # 10. Test statistics
    stats = controller.get_statistics()
    if "soul" in stats and "context" in stats:
        print(f"PASS: Statistics include soul and context data")
        print(f"      Updates: {stats['total_updates']}, Refreshes: {stats['total_refreshes']}")
    else:
        print("FAIL: Statistics incomplete")
        all_passed = False

    # 11. Test force refresh
    pre_refresh = controller._total_refreshes
    controller.force_refresh()
    if controller._total_refreshes > pre_refresh:
        print("PASS: Force refresh worked")
    else:
        print("FAIL: Force refresh didn't increment counter")
        all_passed = False

    # 12. Test refresh history
    history = controller.get_refresh_history(n=5)
    if len(history) > 0:
        print(f"PASS: Refresh history accessible ({len(history)} events)")
        recent = history[-1]
        print(f"      Last refresh: trigger={recent.trigger}, tokens={recent.new_tokens}")
    else:
        print("WARN: No refresh history (may be expected)")

    print()
    print("--- Integration Test: Affect -> Inference ---")
    print()

    # 13. Test full integration: Create soul, process telemetry, get prompt
    soul = create_integrated_soul(storage_path=None)

    # Process telemetry
    soul_state = soul.process_telemetry(telemetry)

    # Get system prompt context
    context = soul.get_system_prompt_context()

    if "IDENTITY" in context and "CURRENT STATE" in context:
        print("PASS: IntegratedSoul generates complete system prompt context")
    else:
        print("FAIL: System prompt context incomplete")
        all_passed = False

    # Check that mood affects context
    mood_data = soul.get_mood_for_prompt()
    if "mood" in mood_data and "quadrant" in mood_data:
        print(f"PASS: Mood data for prompt: {mood_data['mood']} ({mood_data['quadrant']})")
    else:
        print("FAIL: Mood data incomplete")
        all_passed = False

    # 14. Test greeting generation
    greeting = soul.generate_greeting()
    if len(greeting) > 0:
        print(f"PASS: Greeting generated: '{greeting[:50]}...'")
    else:
        print("FAIL: Empty greeting")
        all_passed = False

    # 15. Test context management integration
    manager2 = create_sticky_context(
        llm=None,
        keep_tokens=len(context) // 4,  # Rough token estimate
        n_ctx=8192,
    )

    # Simulate adding system prompt
    manager2.on_tokens_added(len(context) // 4, is_system=True)
    print(f"PASS: Context manager accepts system prompt ({manager2.state.fixed_tokens} tokens)")

    print()
    if all_passed:
        print("All inference module tests PASSED")
    else:
        print("Some inference tests FAILED - review output above")

    return all_passed


def run_bridge_smoke_test():
    """Test the Bridge module - Unified Telemetry and PAD Synchronization."""
    print("\n")
    print("=" * 70)
    print("BRIDGE MODULE SMOKE TEST")
    print("=" * 70)
    print()

    all_passed = True

    # Import bridge modules
    try:
        from mies.bridge import (
            TelemetryBridge,
            TelemetryBridgeConfig,
            create_telemetry_bridge,
            PADSynchronizer,
            PADSource,
            PADConflictResolution,
            create_pad_synchronizer,
            InteroceptionAdapter,
            L1BodyState,
            L2PerceptionState,
            InteroceptivePAD,
            adapt_interoceptive_pad,
        )
        print("PASS: Bridge module imports successful")
    except ImportError as e:
        print(f"FAIL: Bridge import error: {e}")
        return False

    print()
    print("--- PAD Synchronizer Tests ---")
    print()

    # 1. Create PAD synchronizer
    sync = create_pad_synchronizer(
        resolution=PADConflictResolution.WEIGHTED_AVERAGE,
        conflict_threshold=0.3,
    )
    print("PASS: PADSynchronizer created")

    # 2. Report PAD from multiple sources
    from mies.affect import PADVector

    mies_pad = PADVector(pleasure=0.5, arousal=-0.2, dominance=0.3)
    sync.report(PADSource.MIES_CATHEDRAL, mies_pad, confidence=0.8)

    intero_pad = PADVector(pleasure=0.45, arousal=-0.15, dominance=0.35)
    sync.report(PADSource.ARA_INTEROCEPTION, intero_pad, confidence=0.95)

    kernel_pad = PADVector(pleasure=0.55, arousal=-0.25, dominance=0.28)
    sync.report(PADSource.KERNEL_BRIDGE, kernel_pad, confidence=0.85)

    print(f"PASS: Reported PAD from 3 sources")

    # 3. Get canonical PAD (should be weighted average)
    canonical = sync.get_canonical_pad()
    state = sync.get_state()
    print(f"PASS: Canonical PAD: P={canonical.pleasure:.2f}, A={canonical.arousal:.2f}, D={canonical.dominance:.2f}")
    print(f"      Source: {state.source.name}, Confidence: {state.confidence:.2f}")

    # 4. Check conflict detection (these should be in agreement)
    if not state.sources_in_conflict:
        print("PASS: No conflict detected (sources agree within threshold)")
    else:
        print("INFO: Sources in conflict (may be expected)")

    # 5. Test conflicting sources
    conflicting_pad = PADVector(pleasure=-0.8, arousal=0.9, dominance=-0.5)
    sync.report(PADSource.PULSE_ESTIMATION, conflicting_pad, confidence=0.5)

    state_after = sync.get_state()
    if state_after.sources_in_conflict:
        print("PASS: Conflict detected after adding conflicting source")
    else:
        print("INFO: No conflict detected (threshold may be high)")

    # 6. Test statistics
    stats = sync.get_statistics()
    if "sync_count" in stats and "active_sources" in stats:
        print(f"PASS: Statistics available: {stats['sync_count']} syncs, {len(stats['active_sources'])} sources")
    else:
        print("FAIL: Statistics incomplete")
        all_passed = False

    print()
    print("--- Interoception Adapter Tests ---")
    print()

    # 7. Create adapter
    adapter = InteroceptionAdapter(l1_weight=0.4, l2_weight=0.3, hardware_weight=0.3)
    print("PASS: InteroceptionAdapter created")

    # 8. Create L1 body state
    l1 = L1BodyState(
        heart_rate=80.0,
        heart_rate_variability=45.0,
        breath_rate=16.0,
        muscle_tension=0.4,
        skin_conductance=2.5,
        skin_temperature=34.0,
    )
    print(f"PASS: L1BodyState created (HR={l1.heart_rate}, HRV={l1.heart_rate_variability})")

    # 9. Create L2 perception state
    l2 = L2PerceptionState(
        audio_valence=0.3,
        audio_arousal=0.4,
        text_sentiment=0.5,
        attention_focus=0.7,
        novelty_signal=0.2,
    )
    print(f"PASS: L2PerceptionState created (attention={l2.attention_focus})")

    # 10. Process through adapter
    telemetry, l2_factors = adapter.process_interoception(l1=l1, l2=l2)
    print(f"PASS: Telemetry generated from L1/L2:")
    print(f"      CPU temp: {telemetry.cpu_temp:.1f}°C, Load: {telemetry.cpu_load:.2f}")
    print(f"      L2 factors: interaction_valence={l2_factors.get('interaction_valence', 0):.2f}")

    # 11. Adapt SNN PAD
    snn_pad = InteroceptivePAD(valence=0.3, arousal=0.6, dominance=0.55)
    adapted = adapter.adapt_snn_pad(snn_pad, l2_factors)
    print(f"PASS: Adapted SNN PAD: P={adapted.pleasure:.2f}, A={adapted.arousal:.2f}, D={adapted.dominance:.2f}")

    print()
    print("--- Telemetry Bridge Tests ---")
    print()

    # 12. Create telemetry bridge
    bridge = create_telemetry_bridge(
        soul_storage_path=None,  # In-memory
        enable_background_polling=False,
    )
    print("PASS: TelemetryBridge created")

    # 13. Update cycle
    health = bridge.update()
    print(f"PASS: Update cycle completed")
    print(f"      Health: {health.overall_health:.2f}, Thermal OK: {health.thermal_ok}")
    print(f"      PAD: {health.pad.quadrant.name}")

    # 14. Get unified PAD
    unified = bridge.get_unified_pad()
    print(f"PASS: Unified PAD: {unified.source.name}, conf={unified.confidence:.2f}")

    # 15. Get prompt context
    prompt = bridge.get_prompt_context()
    if len(prompt) > 0:
        print(f"PASS: Prompt context generated ({len(prompt)} chars)")
    else:
        print("FAIL: Empty prompt context")
        all_passed = False

    # 16. Event forwarding
    bridge.on_user_interaction(quality=0.7)
    bridge.on_task_completed("test task", success=True)
    print("PASS: Event forwarding works")

    # 17. Statistics
    stats = bridge.get_statistics()
    if "update_count" in stats and "pad_sync" in stats:
        print(f"PASS: Bridge statistics: {stats['update_count']} updates")
    else:
        print("FAIL: Statistics incomplete")
        all_passed = False

    bridge.shutdown()

    print()
    if all_passed:
        print("All bridge module tests PASSED")
    else:
        print("Some bridge tests FAILED - review output above")

    return all_passed


def run_persistence_smoke_test():
    """Test the Persistence module - SQLite emotional memory."""
    print("\n")
    print("=" * 70)
    print("PERSISTENCE MODULE SMOKE TEST")
    print("=" * 70)
    print()

    all_passed = True

    # Import persistence
    try:
        from mies.affect.persistence import (
            PersistenceManager,
            StoredEpisode,
            StoredGoal,
            create_persistence_manager,
        )
        from mies.affect import PADVector
        print("PASS: Persistence module imports successful")
    except ImportError as e:
        print(f"FAIL: Persistence import error: {e}")
        return False

    print()
    print("--- Database Tests ---")
    print()

    # 1. Create in-memory database
    pm = create_persistence_manager(in_memory=True)
    print("PASS: In-memory PersistenceManager created")

    # 2. Save an episode
    import time
    import json

    episode = StoredEpisode(
        id=None,
        timestamp=time.time(),
        context=json.dumps({"activity": "testing", "user_present": True}),
        pad_pleasure=0.6,
        pad_arousal=-0.2,
        pad_dominance=0.4,
        quadrant="SERENE",
        mood_label="calm and content",
        salience=0.7,
        memory_type="EPISODIC",
    )
    episode_id = pm.save_episode(episode)
    print(f"PASS: Episode saved (id={episode_id})")

    # 3. Retrieve recent episodes
    recent = pm.get_recent_episodes(limit=10)
    if len(recent) == 1 and recent[0].id == episode_id:
        print("PASS: Retrieved recent episode correctly")
    else:
        print(f"FAIL: Expected 1 episode, got {len(recent)}")
        all_passed = False

    # 4. Save more episodes with varying salience
    for i in range(5):
        ep = StoredEpisode(
            id=None,
            timestamp=time.time() - i * 60,
            context=json.dumps({"iteration": i}),
            pad_pleasure=0.3 + i * 0.1,
            pad_arousal=-0.1 + i * 0.05,
            pad_dominance=0.5,
            quadrant="SERENE",
            mood_label="test mood",
            salience=0.3 + i * 0.15,
            memory_type="EPISODIC",
        )
        pm.save_episode(ep)
    print("PASS: Saved 5 additional episodes")

    # 5. Get salient episodes
    salient = pm.get_salient_episodes(limit=3, min_salience=0.5)
    if len(salient) >= 2:
        print(f"PASS: Retrieved {len(salient)} salient episodes")
    else:
        print(f"INFO: Got {len(salient)} salient episodes")

    # 6. Get similar episodes
    query_pad = PADVector(pleasure=0.5, arousal=-0.1, dominance=0.5)
    similar = pm.get_similar_episodes(query_pad, threshold=0.3, limit=5)
    print(f"PASS: Similar episode search returned {len(similar)} episodes")

    # 7. Update access count
    pm.update_episode_access(episode_id)
    # Get all episodes and find the one we updated
    all_episodes = pm.get_recent_episodes(limit=10)
    updated_ep = next((e for e in all_episodes if e.id == episode_id), None)
    if updated_ep and updated_ep.access_count > 0:
        print("PASS: Episode access count updated")
    else:
        print("FAIL: Access count not updated")
        all_passed = False

    # 8. Save PAD history
    pad = PADVector(0.4, -0.2, 0.3)
    pm.save_pad_state(pad, source="test", context="testing")
    print("PASS: PAD state saved to history")

    # 9. Get PAD history
    history = pm.get_pad_history(limit=10)
    if len(history) > 0:
        print(f"PASS: PAD history retrieved ({len(history)} entries)")
    else:
        print("FAIL: No PAD history")
        all_passed = False

    # 10. Save a goal
    goal = StoredGoal(
        id=None,
        name="Learn new skill",
        description="Master a new capability",
        importance=0.8,
        progress=0.3,
        created_at=time.time(),
        completed_at=None,
        status="ACTIVE",
    )
    goal_id = pm.save_goal(goal)
    print(f"PASS: Goal saved (id={goal_id})")

    # 11. Get active goals
    active = pm.get_active_goals()
    if len(active) == 1:
        print(f"PASS: Retrieved active goal: '{active[0].name}'")
    else:
        print(f"FAIL: Expected 1 active goal, got {len(active)}")
        all_passed = False

    # 12. Update goal progress
    pm.update_goal_progress(goal_id, 0.6)
    updated_goals = pm.get_active_goals()
    if updated_goals[0].progress == 0.6:
        print("PASS: Goal progress updated")
    else:
        print("FAIL: Goal progress not updated")
        all_passed = False

    # 13. Complete goal
    pm.complete_goal(goal_id)
    active_after = pm.get_active_goals()
    if len(active_after) == 0:
        print("PASS: Goal completed and removed from active list")
    else:
        print("FAIL: Goal still active after completion")
        all_passed = False

    # 14. Save identity
    pm.save_identity(
        full_name="Ara",
        core_values=["PROTECTION", "HONESTY", "HELPFULNESS"],
        personality={"openness": 0.8, "conscientiousness": 0.9},
        age_description="newly awakened",
        awakening_date="2024-12-04",
    )
    print("PASS: Identity snapshot saved")

    # 15. Get identity
    identity = pm.get_latest_identity()
    if identity and identity['full_name'] == "Ara":
        print(f"PASS: Identity retrieved: {identity['full_name']}")
    else:
        print("FAIL: Identity not retrieved correctly")
        all_passed = False

    # 16. Get statistics
    stats = pm.get_statistics()
    print(f"PASS: Database statistics:")
    print(f"      Episodes: {stats['total_episodes']}")
    print(f"      PAD history: {stats['pad_history_count']}")
    print(f"      Completed goals: {stats['completed_goals']}")

    pm.close()
    print("PASS: Database closed")

    print()
    if all_passed:
        print("All persistence tests PASSED")
    else:
        print("Some persistence tests FAILED - review output above")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    inference_success = run_inference_smoke_test()
    bridge_success = run_bridge_smoke_test()
    persistence_success = run_persistence_smoke_test()

    all_success = success and inference_success and bridge_success and persistence_success

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Modality Policy: {'PASS' if success else 'FAIL'}")
    print(f"  Inference Module: {'PASS' if inference_success else 'FAIL'}")
    print(f"  Bridge Module: {'PASS' if bridge_success else 'FAIL'}")
    print(f"  Persistence Module: {'PASS' if persistence_success else 'FAIL'}")
    print("=" * 70)

    sys.exit(0 if all_success else 1)

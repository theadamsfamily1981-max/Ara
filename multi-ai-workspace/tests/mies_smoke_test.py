#!/usr/bin/env python3
"""MIES Smoke Test - Verify modality decisions across scenarios.

Run this to sanity-check that MIES is making sensible decisions
without needing the full avatar stack running.

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
)
from mies.modes import DEFAULT_MODES
from mies.policy.heuristic_baseline import (
    HeuristicModalityPolicy,
)
from mies.policy.ebm_aepo_policy import (
    ThermodynamicGovernor,
    ContentMeta,
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
    )
    ctx.update_derived_fields()
    return ctx


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

    print()
    if all_passed:
        print("All critical constraints PASSED")
    else:
        print("Some constraints FAILED - review output above")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

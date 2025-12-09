"""
Sovereign Tick: The 5-Phase Execution Loop

Each tick runs through:
    sense → soul → teleology → plan → act

This module provides both:
1. Individual phase functions (for testing/customization)
2. The complete sovereign_tick() function

Integration with V1 components:
- AxisMundi: Updated in soul phase with current state HV
- EternalMemory: Queried in soul phase, stored in act phase
- AutonomyController: Synced in teleology phase

Usage:
    from ara.sovereign.tick import SovereignTick
    from ara.sovereign.state import create_initial_state

    tick_runner = SovereignTick(axis, memory, safety)
    state = create_initial_state()

    while running:
        state = tick_runner.run(state)
        await asyncio.sleep(0.1)  # 10 Hz
"""

from __future__ import annotations

import time
import logging
import platform
import psutil
from dataclasses import replace
from typing import Optional, List, Dict, Any, Callable

from .state import (
    SovereignState,
    TimeState,
    HardwareState,
    NodeHardwareState,
    DeviceLoad,
    UserState,
    SoulState,
    TeleologyState,
    WorkState,
    SafetyState,
    AvatarState,
    TraceState,
    AutonomyLevel,
    RiskLevel,
    InitiativeStatus,
    SkillInvocation,
    clone_state,
    compute_global_coherence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Phase Protocols
# =============================================================================

PhaseFunction = Callable[[SovereignState], SovereignState]


# =============================================================================
# Sense Phase
# =============================================================================

def sense_phase(state: SovereignState) -> SovereignState:
    """
    SENSE: Read hardware telemetry, user context, and skill status.

    Owners: BANOS node agents, MindReader, Skill monitors

    Updates:
    - hardware: CPU/GPU/FPGA/NVMe metrics, alerts
    - user: Presence, fatigue, context
    - work: Skill status transitions
    """
    start = time.perf_counter()

    # --- Hardware telemetry ---
    hardware = _read_hardware_telemetry(state.hardware)

    # --- User state inference ---
    user = _infer_user_state(state.user, state.avatar)

    # --- Skill status updates ---
    work = _update_skill_status(state.work)

    # Profile this phase
    elapsed_ms = (time.perf_counter() - start) * 1000
    profiling = dict(state.trace.profiling_ms)
    profiling["sense"] = elapsed_ms

    return replace(
        state,
        hardware=hardware,
        user=user,
        work=work,
        trace=replace(state.trace, profiling_ms=profiling),
    )


def _read_hardware_telemetry(prev: HardwareState) -> HardwareState:
    """Read current hardware state using psutil."""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        # Try to get CPU temperature
        cpu_temp = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        cpu_temp = entries[0].current
                        break
        except Exception:
            pass

        node = NodeHardwareState(
            hostname=platform.node(),
            cpu_util=cpu_percent / 100.0,
            cpu_temp_c=cpu_temp,
            mem_used_gb=mem.used / (1024**3),
            mem_total_gb=mem.total / (1024**3),
        )

        # Check for alerts
        alerts = []
        if cpu_percent > 90:
            alerts.append("cpu_high")
        if mem.percent > 90:
            alerts.append("memory_high")

        return HardwareState(
            nodes={platform.node(): node},
            alerts=alerts,
            total_gpu_util=0.0,  # No GPU monitoring yet
            total_fpga_util=0.0,
            cathedral_health=1.0 if not alerts else 0.8,
        )

    except Exception as e:
        logger.warning(f"Hardware telemetry failed: {e}")
        return prev


def _infer_user_state(prev: UserState, avatar: AvatarState) -> UserState:
    """Infer user state from avatar activity and time."""
    now = time.time()

    # Infer presence from last message
    seconds_since_msg = now - avatar.last_user_msg_ts if avatar.last_user_msg_ts else float("inf")

    if seconds_since_msg < 60:
        presence = "present"
    elif seconds_since_msg < 300:
        presence = "idle"
    else:
        presence = "offline"

    # Night mode check (10pm - 6am)
    from datetime import datetime
    hour = datetime.now().hour
    night_mode = hour >= 22 or hour < 6

    return replace(
        prev,
        presence=presence,
        night_mode=night_mode,
    )


def _update_skill_status(work: WorkState) -> WorkState:
    """Update skill queue based on completion/failure."""
    # Move completed skills from running to completed
    still_running = []
    newly_completed = []

    for skill in work.running_skills:
        if skill.status in ("done", "failed"):
            newly_completed.append(skill)
        else:
            still_running.append(skill)

    completed = list(work.completed_skills) + newly_completed
    # Keep only last 100 completed
    completed = completed[-100:]

    return replace(
        work,
        running_skills=still_running,
        completed_skills=completed,
    )


# =============================================================================
# Soul Phase
# =============================================================================

def soul_phase(
    state: SovereignState,
    axis: Optional["AxisMundi"] = None,
    memory: Optional["EternalMemory"] = None,
) -> SovereignState:
    """
    SOUL: Update hyperdimensional state, compute resonance, apply plasticity.

    Owners: SoulEncoder, SoulFPGA bridge, Plasticity engine

    Updates:
    - soul: h_moment, resonance_scores, plasticity_events
    - Writes to AxisMundi if provided
    - Queries EternalMemory if provided
    """
    start = time.perf_counter()

    soul = state.soul

    # Encode current moment as HV
    h_moment = _encode_moment_hv(state)
    soul = replace(soul, h_moment_173=h_moment)

    # Update AxisMundi if available
    resonance_scores = {}
    if axis is not None:
        try:
            import numpy as np
            # Convert bytes to array
            if h_moment:
                moment_array = np.frombuffer(h_moment, dtype=np.float32)
                if len(moment_array) == axis.dim:
                    axis.write("soul", moment_array)

                    # Compute resonance with other layers
                    for layer in axis.layer_names():
                        if layer != "soul":
                            resonance_scores[f"soul-{layer}"] = axis.coherence("soul", layer)
        except Exception as e:
            logger.warning(f"AxisMundi update failed: {e}")

    soul = replace(soul, resonance_scores=resonance_scores)

    # Query memory for context
    if memory is not None:
        try:
            import numpy as np
            if h_moment:
                moment_array = np.frombuffer(h_moment, dtype=np.float32)
                if len(moment_array) == memory.dim:
                    result = memory.recall(moment_array, k=3)
                    # Update resonance with memory strength
                    soul = replace(
                        soul,
                        attractor_activations={
                            ep.meta.get("topic", "unknown"): ep.similarity
                            for ep in result.episodes
                        },
                    )
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")

    # Compute plasticity reward (simple version)
    reward = _compute_reward(state)
    if abs(reward) > 0.01:
        soul = replace(
            soul,
            last_reward=reward,
            cumulative_reward=soul.cumulative_reward + reward,
            plasticity_events=soul.plasticity_events + 1,
        )

    # Profile
    elapsed_ms = (time.perf_counter() - start) * 1000
    profiling = dict(state.trace.profiling_ms)
    profiling["soul"] = elapsed_ms

    return replace(
        state,
        soul=soul,
        trace=replace(state.trace, profiling_ms=profiling),
    )


def _encode_moment_hv(state: SovereignState, dim: int = 1024) -> bytes:
    """Encode current state as a hypervector."""
    try:
        import numpy as np

        # Create HV from state components
        rng = np.random.default_rng(state.time.tick_id)

        # Base vector
        hv = np.zeros(dim, dtype=np.float32)

        # Encode presence
        presence_seed = hash(state.user.presence) % (2**31)
        presence_hv = np.random.default_rng(presence_seed).choice(
            [-1.0, 1.0], size=dim
        ).astype(np.float32)
        hv += presence_hv

        # Encode autonomy level
        autonomy_seed = int(state.teleology.autonomy_level) * 12345
        autonomy_hv = np.random.default_rng(autonomy_seed).choice(
            [-1.0, 1.0], size=dim
        ).astype(np.float32)
        hv += autonomy_hv * 0.5

        # Encode risk level
        risk_seed = hash(state.safety.global_risk.value) % (2**31)
        risk_hv = np.random.default_rng(risk_seed).choice(
            [-1.0, 1.0], size=dim
        ).astype(np.float32)
        hv += risk_hv * 0.3

        # Normalize to bipolar
        hv = np.sign(hv).astype(np.float32)

        return hv.tobytes()

    except Exception as e:
        logger.warning(f"Moment encoding failed: {e}")
        return b""


def _compute_reward(state: SovereignState) -> float:
    """Compute reward signal for plasticity."""
    reward = 0.0

    # Positive: User engagement
    if state.user.presence == "present":
        reward += 0.1
    if state.user.presence == "deep_focus":
        reward += 0.2

    # Negative: Risk/problems
    if state.safety.global_risk == RiskLevel.HIGH:
        reward -= 0.2
    if state.safety.global_risk == RiskLevel.CRITICAL:
        reward -= 0.5

    # Positive: Skills completing
    if state.work.skills_completed_today > 0:
        reward += 0.05 * min(state.work.skills_completed_today, 5)

    # Negative: Skills failing
    if state.work.skills_failed_today > 0:
        reward -= 0.1 * min(state.work.skills_failed_today, 3)

    return max(-1.0, min(1.0, reward))


# =============================================================================
# Teleology Phase
# =============================================================================

def teleology_phase(
    state: SovereignState,
    safety_controller: Optional["AutonomyController"] = None,
) -> SovereignState:
    """
    TELEOLOGY: Evaluate goals, update alignment, adjust autonomy.

    Owners: Teleology engine, Covenant manager, Safety engine

    Updates:
    - teleology: Goal progress, alignment, autonomy adjustments
    - safety: Risk level, permissions, lockouts
    """
    start = time.perf_counter()

    teleology = state.teleology
    safety = state.safety

    # --- Goal evaluation ---
    # Update goal progress based on completed work
    goals = []
    for goal in teleology.active_goals:
        # Simple progress update (real version would be more sophisticated)
        if goal.goal_id == "g_assist":
            progress = min(1.0, state.work.skills_completed_today / 10)
            goal = replace(goal, progress=progress)
        goals.append(goal)

    teleology = replace(teleology, active_goals=goals)

    # --- Founder protection ---
    protection_engaged = False
    veto_reasons = []

    if state.user.burnout_risk > 0.7:
        protection_engaged = True
        veto_reasons.append("High burnout risk detected")

    if state.user.night_mode and state.user.last_sleep_hours < 6:
        protection_engaged = True
        veto_reasons.append("Night mode + insufficient sleep")

    teleology = replace(
        teleology,
        founder_protection_engaged=protection_engaged,
        veto_reasons=veto_reasons,
    )

    # --- Safety state update ---
    # Compute global risk
    risk = RiskLevel.LOW
    anomalies = list(state.hardware.alerts)

    if state.hardware.cathedral_health < 0.5:
        risk = RiskLevel.HIGH
        anomalies.append("cathedral_degraded")
    elif state.hardware.cathedral_health < 0.8:
        risk = RiskLevel.MEDIUM

    # Check for lockouts
    night_lockout = state.user.night_mode and teleology.autonomy_level >= AutonomyLevel.EXEC_SAFE
    burnout_lockout = state.user.burnout_risk > 0.8

    # Sync with safety controller if available
    if safety_controller is not None:
        try:
            # Update controller with current coherence
            coherence = compute_global_coherence(state)
            safety_controller.update_autonomy(coherence)

            # Get controller's autonomy level
            controller_level = safety_controller.get_autonomy_level()
            if controller_level < int(teleology.autonomy_level):
                teleology = replace(
                    teleology,
                    autonomy_level=AutonomyLevel(controller_level),
                )

            # Check kill switch
            if safety_controller.is_killed():
                safety = replace(
                    safety,
                    kill_switch_active=True,
                    kill_switch_reason="External kill switch",
                )
        except Exception as e:
            logger.warning(f"Safety controller sync failed: {e}")

    safety = replace(
        safety,
        global_risk=risk,
        anomaly_flags=anomalies,
        night_lockout_active=night_lockout,
        burnout_lockout_active=burnout_lockout,
    )

    # Profile
    elapsed_ms = (time.perf_counter() - start) * 1000
    profiling = dict(state.trace.profiling_ms)
    profiling["teleology"] = elapsed_ms

    return replace(
        state,
        teleology=teleology,
        safety=safety,
        trace=replace(state.trace, profiling_ms=profiling),
    )


# =============================================================================
# Plan Phase
# =============================================================================

def plan_phase(state: SovereignState) -> SovereignState:
    """
    PLAN: ChiefOfStaff decides what to work on, schedules skills.

    Owners: ChiefOfStaff (CEO), BANOS scheduler

    Updates:
    - work: Active initiatives, skill queue
    """
    start = time.perf_counter()

    work = state.work

    # Skip planning if safety says no
    if state.safety.kill_switch_active:
        profiling = dict(state.trace.profiling_ms)
        profiling["plan"] = (time.perf_counter() - start) * 1000
        return replace(state, trace=replace(state.trace, profiling_ms=profiling))

    # --- Select active initiatives ---
    # Simple priority-based selection
    active = []
    for init_id, init in work.initiatives.items():
        if init.status == InitiativeStatus.ACTIVE:
            # Check teleology alignment
            if init.teleology_alignment > 0:
                active.append(init_id)

    work = replace(work, active_initiatives=active[:5])  # Top 5

    # --- Schedule skills from queue ---
    # Move queued skills to running if capacity available
    max_concurrent = 3
    if len(work.running_skills) < max_concurrent and work.skill_queue:
        can_start = max_concurrent - len(work.running_skills)
        to_start = work.skill_queue[:can_start]
        remaining = work.skill_queue[can_start:]

        # Check autonomy requirements
        approved = []
        for skill in to_start:
            if skill.autonomy_required <= state.teleology.autonomy_level:
                skill = replace(skill, status="running")
                approved.append(skill)
            else:
                remaining.insert(0, skill)  # Back to queue

        work = replace(
            work,
            running_skills=list(work.running_skills) + approved,
            skill_queue=remaining,
        )

    # Profile
    elapsed_ms = (time.perf_counter() - start) * 1000
    profiling = dict(state.trace.profiling_ms)
    profiling["plan"] = elapsed_ms

    return replace(
        state,
        work=work,
        trace=replace(state.trace, profiling_ms=profiling),
    )


# =============================================================================
# Act Phase
# =============================================================================

def act_phase(
    state: SovereignState,
    memory: Optional["EternalMemory"] = None,
) -> SovereignState:
    """
    ACT: Execute skills, update avatar, log to memory.

    Owners: Skill runtime, Avatar layer, Trace

    Updates:
    - work: Skill execution results
    - avatar: Mode, channel updates
    - trace: Errors, profiling
    - Stores significant moments to EternalMemory
    """
    start = time.perf_counter()

    work = state.work
    trace = state.trace
    soul = state.soul

    # --- Execute running skills ---
    # In real implementation, this would dispatch to skill executors
    # Here we just simulate completion
    updated_skills = []
    completed_count = work.skills_completed_today
    failed_count = work.skills_failed_today

    for skill in work.running_skills:
        # Simulate: skills complete after some time
        elapsed = time.time() - skill.scheduled_ts
        if elapsed > skill.eta_ms / 1000:
            skill = replace(skill, status="done")
            completed_count += 1
        updated_skills.append(skill)

    work = replace(
        work,
        running_skills=updated_skills,
        skills_completed_today=completed_count,
        skills_failed_today=failed_count,
    )

    # --- Store to memory if significant ---
    if memory is not None and _is_significant_moment(state):
        try:
            import numpy as np
            if soul.h_moment_173:
                moment_array = np.frombuffer(soul.h_moment_173, dtype=np.float32)
                if len(moment_array) == memory.dim:
                    memory.store(
                        content_hv=moment_array,
                        strength=0.7,
                        meta={
                            "tick": state.time.tick_id,
                            "presence": state.user.presence,
                            "autonomy": state.teleology.autonomy_level.name,
                            "risk": state.safety.global_risk.value,
                        },
                    )
                    soul = replace(
                        soul,
                        last_memory_store_ts=time.time(),
                        memories_stored_this_session=soul.memories_stored_this_session + 1,
                    )
        except Exception as e:
            logger.warning(f"Memory store failed: {e}")

    # --- Finalize trace ---
    elapsed_ms = (time.perf_counter() - start) * 1000
    profiling = dict(trace.profiling_ms)
    profiling["act"] = elapsed_ms

    # Compute total tick time
    total_ms = sum(profiling.values())

    trace = replace(
        trace,
        profiling_ms=profiling,
        avg_tick_ms=(trace.avg_tick_ms * 0.9 + total_ms * 0.1),  # EMA
        max_tick_ms=max(trace.max_tick_ms, total_ms),
    )

    return replace(
        state,
        work=work,
        soul=soul,
        trace=trace,
    )


def _is_significant_moment(state: SovereignState) -> bool:
    """Determine if this moment should be stored to memory."""
    # Store if:
    # - First tick or every 100 ticks
    if state.time.tick_id == 0 or state.time.tick_id % 100 == 0:
        return True

    # - User just became present
    if state.user.presence == "present":
        return True

    # - Risk changed to high/critical
    if state.safety.global_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return True

    # - Skill completed or failed
    if state.work.skills_completed_today > 0 or state.work.skills_failed_today > 0:
        return True

    return False


# =============================================================================
# Complete Tick Runner
# =============================================================================

class SovereignTick:
    """
    Complete sovereign tick runner with V1 integration.

    Usage:
        tick_runner = SovereignTick(axis, memory, safety)
        state = create_initial_state()

        while running:
            state = tick_runner.run(state)
    """

    def __init__(
        self,
        axis: Optional["AxisMundi"] = None,
        memory: Optional["EternalMemory"] = None,
        safety: Optional["AutonomyController"] = None,
    ):
        self.axis = axis
        self.memory = memory
        self.safety = safety

    def run(self, state: SovereignState) -> SovereignState:
        """
        Execute one complete sovereign tick.

        Goes through all 5 phases: sense → soul → teleology → plan → act
        """
        # Advance time
        now = time.time()
        mono = time.monotonic() * 1000

        state = replace(
            state,
            time=replace(
                state.time,
                tick_id=state.time.tick_id + 1,
                dt_ms=mono - state.time.monotonic_ms,
                monotonic_ms=mono,
                wallclock_ts=now,
            ),
        )

        # Clear previous tick errors
        state = replace(
            state,
            trace=replace(state.trace, last_tick_errors=[], profiling_ms={}),
        )

        try:
            # Phase 1: Sense
            state = replace(state, time=replace(state.time, phase="sense"))
            state = sense_phase(state)

            # Phase 2: Soul
            state = replace(state, time=replace(state.time, phase="soul"))
            state = soul_phase(state, self.axis, self.memory)

            # Phase 3: Teleology
            state = replace(state, time=replace(state.time, phase="teleology"))
            state = teleology_phase(state, self.safety)

            # Phase 4: Plan
            state = replace(state, time=replace(state.time, phase="plan"))
            state = plan_phase(state)

            # Phase 5: Act
            state = replace(state, time=replace(state.time, phase="act"))
            state = act_phase(state, self.memory)

        except Exception as e:
            logger.error(f"Tick failed: {e}", exc_info=True)
            state = replace(
                state,
                trace=replace(
                    state.trace,
                    last_tick_errors=[str(e)],
                ),
            )

        # Check for overrun
        total_ms = sum(state.trace.profiling_ms.values())
        if total_ms > state.trace.tick_budget_ms:
            state = replace(
                state,
                time=replace(
                    state.time,
                    overrun_count=state.time.overrun_count + 1,
                ),
            )

        return state


# =============================================================================
# Convenience function
# =============================================================================

def sovereign_tick(
    prev: SovereignState,
    axis: Optional["AxisMundi"] = None,
    memory: Optional["EternalMemory"] = None,
    safety: Optional["AutonomyController"] = None,
) -> SovereignState:
    """
    Execute a single sovereign tick (functional interface).

    For object-oriented usage, see SovereignTick class.
    """
    runner = SovereignTick(axis, memory, safety)
    return runner.run(prev)

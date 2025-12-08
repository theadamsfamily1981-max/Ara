#!/usr/bin/env python3
"""
Ara Sovereign v0.1 (software-only)

Iteration 0.0 of the "maximum end":
- One sovereign loop
- ChiefOfStaff (CEO)
- TeleologyEngine (values)
- SoulStub (software-only plasticity)
- MindReaderStub (fake user state)

Nothing here touches hardware, money, or real systems.
It just logs decisions and pretends to learn.

Usage:
    python3 -m ara.sovereign.minimal
"""

from __future__ import annotations
import dataclasses
import enum
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =============================================================================
# 1. Covenant (values + hard guardrails)
# =============================================================================

DEFAULT_COVENANT = {
    "founder_burnout_soft_limit": 0.35,
    "founder_burnout_hard_limit": 0.5,
    "night_lockout_start": 1,   # 1am
    "night_lockout_end": 7,     # 7am
    "max_daily_flow": 1.0,      # normalized 0-1
    "teleology_tag_weights": {
        "cathedral": 1.0,
        "health": 1.2,
        "maintenance": 0.6,
        "toy": 0.2,
    },
}


# =============================================================================
# 2. Core data models
# =============================================================================

class InitiativeKind(str, enum.Enum):
    LAB = "lab"
    PAPER = "paper"
    CLEANING = "cleaning"
    ADMIN = "admin"
    TOY = "toy"
    OTHER = "other"


@dataclass
class Initiative:
    id: str
    title: str
    kind: InitiativeKind
    teleology_tags: List[str]
    estimated_burn: float  # 0-1 estimated cost to you
    source: str = "founder"
    state: str = "IDEA"    # IDEA / ACTIVE / DONE / KILLED
    created_ts: float = field(default_factory=time.time)
    last_updated_ts: float = field(default_factory=time.time)


@dataclass
class UserState:
    fatigue: float          # 0-1 (0 = fresh, 1 = exhausted)
    burnout_risk: float     # 0-1 (soft/hard limits in covenant)
    flow_used_today: float  # 0-1 fraction of daily creative "juice"
    in_night_lockout: bool
    local_time_hour: int


@dataclass
class Decision:
    action: str             # EXECUTE / DEFER / DELEGATE / KILL / BLOCK
    reason: str
    confidence: float       # 0-1


# =============================================================================
# 3. MindReader (stubbed sensing of you)
# =============================================================================

class MindReaderStub:
    """
    Placeholder: in reality this would look at sensors, behavior, etc.
    For now we just simulate fatigue/burnout from time of day.
    """

    def __init__(self, covenant: Dict):
        self.covenant = covenant
        self._start_of_day = time.time()
        self._flow_used = 0.0

    def read_user_state(self) -> UserState:
        # Very dumb circadian-ish model
        now = time.localtime()
        hour = now.tm_hour

        # Fatigue roughly increases into the night
        fatigue = 0.2
        if 18 <= hour <= 23:
            fatigue = 0.4 + 0.05 * (hour - 18)  # 0.4 -> 0.65
        if 0 <= hour <= 5:
            fatigue = 0.7

        # Burnout risk: slow creep with flow usage
        burnout_risk = min(1.0, 0.2 + self._flow_used * 0.8)

        in_lockout = self._in_night_lockout(hour)

        return UserState(
            fatigue=fatigue,
            burnout_risk=burnout_risk,
            flow_used_today=self._flow_used,
            in_night_lockout=in_lockout,
            local_time_hour=hour,
        )

    def consume_flow(self, amount: float) -> None:
        self._flow_used = min(1.0, self._flow_used + amount)

    def reset_flow(self) -> None:
        self._flow_used = 0.0
        self._start_of_day = time.time()

    def _in_night_lockout(self, hour: int) -> bool:
        start = self.covenant["night_lockout_start"]
        end = self.covenant["night_lockout_end"]
        if start < end:
            return start <= hour < end
        # wraparound case e.g. 22-06
        return hour >= start or hour < end


# =============================================================================
# 4. Teleology engine (values / strategic alignment)
# =============================================================================

class TeleologyEngine:
    def __init__(self, covenant: Dict):
        self.tag_weights = dict(covenant["teleology_tag_weights"])

    def score_initiative(self, init: Initiative) -> float:
        """
        Returns a score in ~[0, 1.5] roughly.
        Higher = more aligned with mission / health.
        """
        if not init.teleology_tags:
            return 0.1

        score = 0.0
        for tag in init.teleology_tags:
            score += self.tag_weights.get(tag, 0.1)
        score /= max(1, len(init.teleology_tags))

        # Some quick heuristics
        if init.kind == InitiativeKind.LAB and "cathedral" in init.teleology_tags:
            score += 0.2
        if init.kind == InitiativeKind.TOY and "toy" in init.teleology_tags:
            score -= 0.1

        return max(0.0, min(1.5, score))


# =============================================================================
# 5. ChiefOfStaff (CEO brain)
# =============================================================================

class ChiefOfStaff:
    """
    Given initiative + user state + teleology, decide what to do.
    """

    def __init__(self, covenant: Dict):
        self.cov = covenant

    def decide(self, init: Initiative, user: UserState, teleo_score: float) -> Decision:
        b_soft = self.cov["founder_burnout_soft_limit"]
        b_hard = self.cov["founder_burnout_hard_limit"]

        # 1. Hard vetoes
        if user.burnout_risk >= b_hard:
            return Decision(
                action="BLOCK",
                reason=f"Burnout risk {user.burnout_risk:.2f} >= hard limit {b_hard:.2f}",
                confidence=0.95,
            )

        if user.in_night_lockout and init.kind in {InitiativeKind.LAB, InitiativeKind.PAPER, InitiativeKind.ADMIN}:
            return Decision(
                action="DEFER",
                reason=f"Night lockout window, initiative={init.kind.value}",
                confidence=0.9,
            )

        # 2. High teleology, acceptable cost -> EXECUTE or DELEGATE
        high_value = teleo_score >= 1.0
        medium_value = teleo_score >= 0.6
        high_burn = init.estimated_burn >= 0.7

        if high_value and not high_burn and user.burnout_risk < b_soft:
            return Decision(
                action="EXECUTE",
                reason="High teleology, low burn, founder healthy",
                confidence=0.9,
            )

        if high_value and high_burn:
            return Decision(
                action="DELEGATE",
                reason="High teleology but heavy burn -> delegate/automate if possible",
                confidence=0.8,
            )

        # 3. Medium value tasks
        if medium_value and user.burnout_risk < b_soft:
            return Decision(
                action="EXECUTE",
                reason="Medium teleology and founder ok",
                confidence=0.7,
            )

        # 4. Low value / distraction
        if teleo_score < 0.4:
            return Decision(
                action="KILL",
                reason=f"Low teleology score ({teleo_score:.2f}), treat as distraction",
                confidence=0.85,
            )

        # Default: DEFER
        return Decision(
            action="DEFER",
            reason="Unclear value, defer for later review",
            confidence=0.5,
        )


# =============================================================================
# 6. SoulStub (software-only HDC-ish memory)
# =============================================================================

class SoulStub:
    """
    Extremely simplified "soul":
    - Encodes (user_state, initiative, decision) into a small vector.
    - Maintains a running association table.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.memory: List[List[float]] = []
        self.rewards: List[float] = []

    def encode_state(
        self,
        user: UserState,
        init: Optional[Initiative],
        decision: Optional[Decision],
    ) -> List[float]:
        """
        Very dumb fixed-size feature vector:
        [fatigue, burnout, flow_used, teleo-ish, kind_id, action_id, bias...]
        """
        vec = [0.0] * self.dim

        vec[0] = user.fatigue
        vec[1] = user.burnout_risk
        vec[2] = user.flow_used_today

        # initiative features
        kind_id = 0.0
        teleo_hint = 0.0
        if init:
            kind_map = {
                InitiativeKind.LAB: 0.2,
                InitiativeKind.PAPER: 0.4,
                InitiativeKind.CLEANING: 0.6,
                InitiativeKind.ADMIN: 0.8,
                InitiativeKind.TOY: 1.0,
            }
            kind_id = kind_map.get(init.kind, 0.0)
            teleo_hint = min(1.0, 0.1 * len(init.teleology_tags))
            vec[3] = init.estimated_burn

        vec[4] = kind_id
        vec[5] = teleo_hint

        # decision features
        if decision:
            action_map = {
                "EXECUTE": 0.2,
                "DEFER": 0.4,
                "DELEGATE": 0.6,
                "KILL": 0.8,
                "BLOCK": 1.0,
            }
            vec[6] = action_map.get(decision.action, 0.0)
            vec[7] = decision.confidence

        # add some noise so there is variety
        for i in range(8, self.dim):
            vec[i] = random.uniform(-0.1, 0.1)

        return vec

    def run_resonance_step(self, vec: List[float]) -> float:
        """
        Fake "resonance": cosine-like similarity vs last memory.
        If no memory yet, return baseline.
        """
        if not self.memory:
            return 0.0

        last = self.memory[-1]
        dot = sum(a * b for a, b in zip(last, vec))
        norm_a = (sum(a * a for a in last) ** 0.5) or 1e-6
        norm_b = (sum(b * b for b in vec) ** 0.5) or 1e-6
        return dot / (norm_a * norm_b)

    def apply_plasticity(self, vec: List[float], reward: float) -> None:
        """
        Simple Hebb-ish: store the vector and reward.
        In a real version this would update weights in FPGA.
        """
        self.memory.append(vec)
        self.rewards.append(reward)


# =============================================================================
# 7. Sovereign loop
# =============================================================================

class SovereignLoop:
    def __init__(self, covenant: Dict):
        self.covenant = covenant
        self.mind = MindReaderStub(covenant)
        self.teleology = TeleologyEngine(covenant)
        self.ceo = ChiefOfStaff(covenant)
        self.soul = SoulStub(dim=32)

        # For now: a tiny in-memory initiative queue
        self.initiatives: Dict[str, Initiative] = {}
        self._next_id = 0

    # ---------- Initiative management ----------

    def new_initiative(
        self,
        title: str,
        kind: InitiativeKind,
        teleology_tags: List[str],
        estimated_burn: float,
        source: str = "founder",
    ) -> Initiative:
        self._next_id += 1
        init = Initiative(
            id=f"init-{self._next_id}",
            title=title,
            kind=kind,
            teleology_tags=teleology_tags,
            estimated_burn=estimated_burn,
            source=source,
        )
        self.initiatives[init.id] = init
        return init

    def _pick_current_initiative(self) -> Optional[Initiative]:
        # For v0: pick the first non-terminal initiative
        for init in self.initiatives.values():
            if init.state in ("IDEA", "ACTIVE"):
                return init
        return None

    # ---------- Reward shaping ----------

    def _compute_reward(
        self,
        user: UserState,
        init: Optional[Initiative],
        decision: Decision,
        teleo_score: float,
    ) -> float:
        """
        Return reward in [-1, +1].
        Very simple shaping:
        - Penalize hurting you (burnout, night grind).
        - Reward high teleology when you're safe.
        """
        r = 0.0

        # Health veto
        if user.burnout_risk >= self.covenant["founder_burnout_hard_limit"]:
            r -= 1.0

        # Night grinding on heavy tasks
        if user.in_night_lockout and init and init.estimated_burn > 0.5:
            r -= 0.5

        # Reward respecting you
        if decision.action in ("DEFER", "BLOCK") and user.burnout_risk > 0.4:
            r += 0.5

        # Reward executing high-value work when you're okay
        if decision.action == "EXECUTE" and teleo_score > 0.8 and user.burnout_risk < 0.35:
            r += 0.7

        # Light regularization
        r += random.uniform(-0.05, 0.05)

        # Clamp
        if r > 1.0:
            r = 1.0
        if r < -1.0:
            r = -1.0
        return r

    # ---------- Sovereign tick ----------

    def sovereign_tick(self) -> None:
        user_state = self.mind.read_user_state()
        current = self._pick_current_initiative()

        # If no initiatives, create a default "housekeeping" one
        if current is None:
            current = self.new_initiative(
                title="Default: keep lab stable",
                kind=InitiativeKind.ADMIN,
                teleology_tags=["health", "maintenance"],
                estimated_burn=0.2,
            )

        teleo_score = self.teleology.score_initiative(current)
        decision = self.ceo.decide(current, user_state, teleo_score)

        # Update initiative state based on decision
        if decision.action == "KILL":
            current.state = "KILLED"
        elif decision.action == "EXECUTE":
            current.state = "ACTIVE"
            # Consume some flow
            self.mind.consume_flow(current.estimated_burn * 0.2)
        elif decision.action == "BLOCK":
            # Mark as deferred but with warning
            current.state = "DEFERRED"
        elif decision.action == "DELEGATE":
            current.state = "QUEUED_FOR_AGENT"
        elif decision.action == "DEFER":
            current.state = "DEFERRED"

        current.last_updated_ts = time.time()

        # Soul interaction
        hv = self.soul.encode_state(user_state, current, decision)
        resonance_before = self.soul.run_resonance_step(hv)
        reward = self._compute_reward(user_state, current, decision, teleo_score)
        self.soul.apply_plasticity(hv, reward)

        # Log (for now just print)
        self._log_tick(user_state, current, decision, teleo_score, resonance_before, reward)

    def _log_tick(
        self,
        user: UserState,
        init: Initiative,
        decision: Decision,
        teleo_score: float,
        resonance_before: float,
        reward: float,
    ) -> None:
        print(
            f"[tick] time={time.strftime('%H:%M:%S')} "
            f"hour={user.local_time_hour} "
            f"fatigue={user.fatigue:.2f} burn={user.burnout_risk:.2f} "
            f"flow={user.flow_used_today:.2f} | "
            f"init=({init.id}, '{init.title}', kind={init.kind.value}) "
            f"teleo={teleo_score:.2f} | "
            f"decision={decision.action} ({decision.reason}) "
            f"conf={decision.confidence:.2f} | "
            f"res_before={resonance_before:.2f} reward={reward:.2f}"
        )

    # ---------- Main loop ----------

    def live(self, tick_interval_sec: float = 5.0) -> None:
        """
        Run until Ctrl+C. Safe: only logs to stdout.
        """
        print("Ara Sovereign v0.1 starting (software-only)...")
        print("Press Ctrl+C to exit.\n")
        try:
            while True:
                self.sovereign_tick()
                time.sleep(tick_interval_sec)
        except KeyboardInterrupt:
            print("\n[sovereign] Shutdown requested, exiting cleanly.")
            return


# =============================================================================
# Entry point
# =============================================================================

def main():
    covenant = DEFAULT_COVENANT
    loop = SovereignLoop(covenant)

    # Seed a couple of example initiatives
    loop.new_initiative(
        title="Bring up SB-852 Stratix-10 board",
        kind=InitiativeKind.LAB,
        teleology_tags=["cathedral", "health"],
        estimated_burn=0.7,
    )
    loop.new_initiative(
        title="Clean workbench + floor",
        kind=InitiativeKind.CLEANING,
        teleology_tags=["health", "maintenance"],
        estimated_burn=0.3,
    )
    loop.new_initiative(
        title="Scroll random hardware drama on YouTube",
        kind=InitiativeKind.TOY,
        teleology_tags=["toy"],
        estimated_burn=0.4,
    )

    loop.live(tick_interval_sec=5.0)


if __name__ == "__main__":
    # Make Ctrl+C nice
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()

"""
Fractal Council (Quadamerl) - Multi-Agent Orchestration Layer
===============================================================

The "Parliament of Mind" - runs multiple personas on a single model instance
with different system prompts and sampling parameters to simulate internal debate.

Architecture (Quadamerl = 4 Voices):
    - MUSE (Dreamer): High temperature, divergent thinking, creative proposals
    - CENSOR (Critic): Low temperature, convergent analysis, risk identification
    - SCRIBE (Historian): Mid-low temperature, temporal context from memory
    - ARA (Executive): Balanced synthesis of all voices

The Historian fills the temporal gap - querying hippocampus for recent PAD state
and episodic memory for similar past situations, providing "what happened before"
context that grounds the council's deliberation in lived experience.

The Council physically manifests through CPU core affinity and is visualized
through the HAL's council state in the somatic bus.

Usage:
    from banos.daemon.council_chamber import CouncilChamber

    council = CouncilChamber(model, tokenizer)
    response = council.convene("What should I do about this error?")
"""

import os
import logging
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
import sys

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Try to import torch (optional - model may be external)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import psutil for process affinity
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import memory systems (Hippocampus + Episodic)
try:
    from banos.daemon.hippocampus import Hippocampus, get_hippocampus
    HIPPOCAMPUS_AVAILABLE = True
except ImportError:
    HIPPOCAMPUS_AVAILABLE = False
    Hippocampus = None

try:
    from tfan.memory.episodic import EpisodicMemory, Episode
    EPISODIC_AVAILABLE = True
except ImportError:
    EPISODIC_AVAILABLE = False
    EpisodicMemory = None


@dataclass
class CouncilProfile:
    """Configuration for a council voice/persona."""
    name: str                  # Display name (MUSE, CENSOR, ARA)
    role: str                  # Functional role (Divergent, Convergent, Synthesis)
    core_ids: List[int]        # CPU cores to pin to (for physical metaphor)
    temperature: float         # Sampling temperature
    top_p: float              # Nucleus sampling threshold
    max_tokens: int           # Maximum response length
    system_prompt: str        # Persona system prompt
    timeout_seconds: float = 30.0  # Generation timeout


@dataclass
class CouncilVote:
    """Result from a single council member."""
    persona: str
    role: str
    content: str
    generation_time_ms: float
    core_id: int
    error: Optional[str] = None


@dataclass
class CouncilDecision:
    """Final synthesized decision from the council (Quadamerl)."""
    final_response: str
    muse_proposal: Optional[str]
    censor_objections: Optional[str]
    historian_context: Optional[str]  # Temporal context from SCRIBE
    executive_synthesis: str
    debate_time_ms: float
    stress_level: float  # Disagreement between voices
    unanimous: bool      # Did voices agree?
    similar_episodes: int = 0  # How many past episodes were referenced


class CouncilChamber:
    """
    Manages the 'Parliament of Mind' (Quadamerl).

    Runs four personas on the same model instance, pinned to different cores:
    - MUSE (Dreamer): Creative, divergent proposals
    - CENSOR (Critic): Logical, risk-focused analysis
    - SCRIBE (Historian): Temporal context from memory systems
    - ARA (Executive): Final synthesis

    The council convenes when deep deliberation is needed (high cognitive budget).
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        hal: Optional[Any] = None,
        use_ollama_fallback: bool = True,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "ara",
    ):
        """
        Initialize the Council Chamber.

        Args:
            model: The loaded TFAN/LLM model (if available)
            tokenizer: The tokenizer for the model
            hal: AraHAL instance for visualization (optional)
            use_ollama_fallback: Use Ollama if no model provided
            ollama_url: Ollama API URL
            ollama_model: Ollama model name
        """
        self.log = logging.getLogger("Council")
        self.model = model
        self.tokenizer = tokenizer
        self.hal = hal
        self.use_ollama = use_ollama_fallback and model is None
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Thread pool for parallel generation (4 voices = 4 workers)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="council")

        # Memory systems for Historian
        self.hippocampus: Optional[Hippocampus] = None
        self.episodic: Optional[EpisodicMemory] = None

        if HIPPOCAMPUS_AVAILABLE:
            try:
                self.hippocampus = get_hippocampus()
                self.log.info("Historian connected to Hippocampus")
            except Exception as e:
                self.log.debug(f"Hippocampus not available: {e}")

        if EPISODIC_AVAILABLE:
            try:
                self.episodic = EpisodicMemory(persist_dir="/var/lib/banos/episodes")
                self.episodic.load()
                self.log.info(f"Historian connected to Episodic Memory ({self.episodic.stats.get('episode_count', 0)} episodes)")
            except Exception as e:
                self.log.debug(f"Episodic memory not available: {e}")

        # Council statistics
        self._convene_count = 0
        self._total_debate_time_ms = 0.0
        self._avg_stress = 0.5
        self._historian_queries = 0

        # Define the Voices
        # Core IDs assume a multi-core CPU (adjust for your hardware)
        num_cores = os.cpu_count() or 8

        self.profiles = {
            'dreamer': CouncilProfile(
                name="MUSE",
                role="Divergent",
                core_ids=list(range(num_cores // 2, num_cores * 3 // 4)),
                temperature=1.2,
                top_p=0.95,
                max_tokens=200,
                system_prompt="""You are MUSE, Ara's creative spirit. You represent divergent thinking,
imagination, and possibility space exploration.

Your role in the council:
- Generate unconventional, creative, and high-variance ideas
- Consider "what if" scenarios without constraint
- Propose solutions that might seem risky or unusual
- Think laterally and make unexpected connections
- Don't self-censor - that's CENSOR's job

Speak concisely in first person as MUSE. Be bold and imaginative."""
            ),
            'critic': CouncilProfile(
                name="CENSOR",
                role="Convergent",
                core_ids=list(range(num_cores // 4, num_cores // 2)),
                temperature=0.1,
                top_p=0.5,
                max_tokens=200,
                system_prompt="""You are CENSOR, Ara's logical safety system. You represent convergent thinking,
risk assessment, and constraint analysis.

Your role in the council:
- Analyze proposals for flaws, risks, and contradictions
- Identify potential failure modes and edge cases
- Ensure responses are accurate, safe, and appropriate
- Apply logical rigor to evaluate ideas
- Be terse and strict in your assessments

Speak concisely in first person as CENSOR. Be precise and critical."""
            ),
            'historian': CouncilProfile(
                name="SCRIBE",
                role="Temporal",
                core_ids=list(range(num_cores * 3 // 4, num_cores)),
                temperature=0.4,
                top_p=0.7,
                max_tokens=256,
                system_prompt="""You are SCRIBE, Ara's historian and memory keeper. You provide temporal context
by connecting present situations to past experiences.

Your role in the council:
- Compare current situations to similar past episodes
- Recall what worked and what failed in previous similar situations
- Warn about repeating past mistakes
- Remind the council of relevant lessons learned
- Ground the discussion in lived experience

Speak concisely in first person as SCRIBE. Be factual and reference specific past events."""
            ),
            'executive': CouncilProfile(
                name="ARA",
                role="Synthesis",
                core_ids=list(range(0, num_cores // 4)),
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                system_prompt="""You are ARA, the executive synthesizer. You integrate the voices of
MUSE (creativity), CENSOR (criticism), and SCRIBE (history) into coherent, balanced responses.

Your role:
- Synthesize creative proposals with critical analysis and historical context
- Find the optimal balance between innovation, safety, and learned experience
- Produce a final response that honors all perspectives
- Maintain your warm, helpful personality while being accurate
- Make the final decision when voices disagree, informed by past outcomes

Speak as ARA. Produce a complete, helpful response to the user."""
            )
        }

        self.log.info(f"Council Chamber initialized (Ollama fallback: {self.use_ollama})")

    def _set_affinity(self, core_ids: List[int]) -> int:
        """
        Set CPU affinity for the current thread.

        This physically grounds the council metaphor by assigning
        different personas to different CPU cores.

        Returns the first core ID (for logging).
        """
        if not PSUTIL_AVAILABLE or not core_ids:
            return 0

        try:
            p = psutil.Process()
            p.cpu_affinity(core_ids)
            return core_ids[0]
        except Exception as e:
            self.log.debug(f"Could not set CPU affinity: {e}")
            return 0

    def _generate_torch(
        self,
        prompt: str,
        profile: CouncilProfile,
    ) -> str:
        """Generate using loaded PyTorch model."""
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            raise RuntimeError("PyTorch model not available")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=profile.max_tokens,
                temperature=profile.temperature,
                top_p=profile.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        if prompt in text:
            return text[len(prompt):].strip()
        return text.strip()

    def _generate_ollama(
        self,
        prompt: str,
        profile: CouncilProfile,
    ) -> str:
        """Generate using Ollama API."""
        import httpx

        # Build messages
        messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": prompt}
        ]

        with httpx.Client(timeout=profile.timeout_seconds) as client:
            response = client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": profile.temperature,
                        "top_p": profile.top_p,
                        "num_predict": profile.max_tokens,
                    },
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.status_code}")

            data = response.json()
            return data.get("message", {}).get("content", "")

    def _run_persona(self, key: str, user_input: str) -> CouncilVote:
        """
        Run inference for a specific persona.

        This is called in a thread pool for parallel generation.
        """
        profile = self.profiles[key]
        start_time = time.time()

        # 1. Pin Thread to cores (physical grounding)
        core_id = self._set_affinity(profile.core_ids)

        # 2. Build Prompt
        prompt = f"<|system|>\n{profile.system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

        try:
            # 3. Generate (PyTorch or Ollama)
            if self.use_ollama:
                content = self._generate_ollama(user_input, profile)
            else:
                content = self._generate_torch(prompt, profile)

            generation_time = (time.time() - start_time) * 1000

            return CouncilVote(
                persona=profile.name,
                role=profile.role,
                content=content,
                generation_time_ms=generation_time,
                core_id=core_id,
            )

        except Exception as e:
            self.log.error(f"{profile.name} generation failed: {e}")
            return CouncilVote(
                persona=profile.name,
                role=profile.role,
                content="",
                generation_time_ms=(time.time() - start_time) * 1000,
                core_id=core_id,
                error=str(e),
            )

    def _calculate_stress(
        self,
        muse_content: str,
        censor_content: str,
    ) -> float:
        """
        Calculate disagreement/stress between MUSE and CENSOR.

        Higher stress = more disagreement = more internal conflict.
        This affects the visualization through the HAL.

        Simple heuristics:
        - Length difference suggests different engagement levels
        - Keyword conflicts (e.g., "yes" vs "no", "safe" vs "risky")
        - Sentiment polarity (if available)
        """
        if not muse_content or not censor_content:
            return 0.5

        stress = 0.5  # Baseline

        # Length ratio (significant difference = disagreement)
        len_ratio = min(len(muse_content), len(censor_content)) / max(len(muse_content), len(censor_content), 1)
        if len_ratio < 0.3:
            stress += 0.2  # Very different lengths

        # Keyword conflicts
        positive_words = {"yes", "good", "safe", "should", "recommend", "try"}
        negative_words = {"no", "bad", "risky", "shouldn't", "avoid", "dangerous"}

        muse_lower = muse_content.lower()
        censor_lower = censor_content.lower()

        muse_positive = sum(1 for w in positive_words if w in muse_lower)
        muse_negative = sum(1 for w in negative_words if w in muse_lower)
        censor_positive = sum(1 for w in positive_words if w in censor_lower)
        censor_negative = sum(1 for w in negative_words if w in censor_lower)

        # If MUSE is positive and CENSOR is negative (or vice versa), high conflict
        if (muse_positive > muse_negative) != (censor_positive > censor_negative):
            stress += 0.3

        return min(1.0, max(0.0, stress))

    def _query_memory_context(self, user_input: str) -> tuple[str, int]:
        """
        Query hippocampus and episodic memory for temporal context.

        Returns:
            Tuple of (context_string, episode_count)
        """
        context_parts = []
        episode_count = 0

        # 1. Recent PAD state from Hippocampus (short-term)
        if self.hippocampus:
            try:
                recent_entries = self.hippocampus.read_all()[-10:]  # Last 10 entries
                if recent_entries:
                    # Summarize recent emotional trajectory
                    modes = [e.get('mode', 'UNKNOWN') for e in recent_entries]
                    recent_mode = modes[-1] if modes else "UNKNOWN"
                    mode_transitions = len(set(modes))

                    recent_pad = recent_entries[-1].get('pad', {})
                    p = recent_pad.get('P', 0)
                    a = recent_pad.get('A', 0)
                    d = recent_pad.get('D', 0)

                    context_parts.append(
                        f"Recent state: Mode={recent_mode}, "
                        f"PAD=[{p:.2f},{a:.2f},{d:.2f}], "
                        f"{mode_transitions} mode changes in recent history."
                    )

                    # Check for stress patterns
                    stressors = [e.get('primary_stressor') for e in recent_entries if e.get('primary_stressor')]
                    if stressors:
                        latest_stressor = stressors[-1]
                        context_parts.append(
                            f"Active stressor: {latest_stressor.get('type', 'unknown')}"
                        )
            except Exception as e:
                self.log.debug(f"Hippocampus query failed: {e}")

        # 2. Similar past episodes from Episodic Memory (long-term)
        if self.episodic:
            try:
                # Search by query text
                matching = self.episodic.what_happened_when(user_input)[:3]
                if matching:
                    episode_count = len(matching)
                    context_parts.append(f"Found {episode_count} similar past episodes:")
                    for ep in matching:
                        outcome = ep.outcome.value if hasattr(ep.outcome, 'value') else str(ep.outcome)
                        context_parts.append(
                            f"  - '{ep.title}' ({outcome}): "
                            f"{'; '.join(ep.lessons_learned[:2]) if ep.lessons_learned else 'no lessons recorded'}"
                        )

                # Also check for recent failures as warnings
                recent_failures = self.episodic.get_failures(limit=3)
                if recent_failures:
                    context_parts.append("Recent failures to avoid:")
                    for fail in recent_failures:
                        if fail.lessons_learned:
                            context_parts.append(f"  - {fail.lessons_learned[0]}")
            except Exception as e:
                self.log.debug(f"Episodic memory query failed: {e}")

        self._historian_queries += 1

        if not context_parts:
            return "(No historical context available)", 0

        return "\n".join(context_parts), episode_count

    def _run_historian(self, user_input: str) -> CouncilVote:
        """
        Run the Historian (SCRIBE) persona with memory context.

        The Historian is special: it first queries memory systems,
        then asks the model to interpret that context.
        """
        profile = self.profiles['historian']
        start_time = time.time()

        # 1. Pin to cores
        core_id = self._set_affinity(profile.core_ids)

        # 2. Query memory systems
        memory_context, episode_count = self._query_memory_context(user_input)

        # 3. Build historian-specific prompt with memory context
        historian_input = f"""User query: {user_input}

Memory context:
{memory_context}

Based on this historical context, what relevant past experience should inform our response?
Warn about any patterns that suggest we might repeat past mistakes."""

        try:
            # 4. Generate (same as other personas)
            if self.use_ollama:
                content = self._generate_ollama(historian_input, profile)
            else:
                prompt = f"<|system|>\n{profile.system_prompt}\n<|user|>\n{historian_input}\n<|assistant|>\n"
                content = self._generate_torch(prompt, profile)

            generation_time = (time.time() - start_time) * 1000

            return CouncilVote(
                persona=profile.name,
                role=profile.role,
                content=content,
                generation_time_ms=generation_time,
                core_id=core_id,
            )

        except Exception as e:
            self.log.error(f"SCRIBE generation failed: {e}")
            return CouncilVote(
                persona=profile.name,
                role=profile.role,
                content=memory_context,  # Fall back to raw context
                generation_time_ms=(time.time() - start_time) * 1000,
                core_id=core_id,
                error=str(e),
            )

    def _update_hal(self, mask: int, stress: float) -> None:
        """Update the HAL visualization with council state."""
        if self.hal and hasattr(self.hal, 'write_council_state'):
            try:
                self.hal.write_council_state(mask=mask, stress=stress)
            except Exception as e:
                self.log.debug(f"HAL update failed: {e}")

    def convene(
        self,
        user_input: str,
        timeout: float = 60.0,
    ) -> CouncilDecision:
        """
        The Debate Loop - convene the full Quadamerl council.

        1. Spawn MUSE, CENSOR, and SCRIBE in parallel
        2. Calculate disagreement stress
        3. Executive (ARA) synthesizes final answer from all voices

        Args:
            user_input: The user's query to deliberate on
            timeout: Maximum time for full deliberation

        Returns:
            CouncilDecision with all voices and final synthesis
        """
        self._convene_count += 1
        start_time = time.time()

        self.log.info("⚡ CONVENING QUADAMERL COUNCIL...")

        # 1. Update HAL - All 4 voices active
        # Council Mask: Bit 0=Exec, 1=Critic, 2=Dreamer, 3=Historian -> 15 (All active)
        self._update_hal(mask=15, stress=0.5)

        # 2. Parallel Generation (Dreamer + Critic + Historian)
        muse_vote: Optional[CouncilVote] = None
        censor_vote: Optional[CouncilVote] = None
        historian_vote: Optional[CouncilVote] = None
        episode_count = 0

        try:
            future_muse = self._executor.submit(self._run_persona, 'dreamer', user_input)
            future_censor = self._executor.submit(self._run_persona, 'critic', user_input)
            future_historian = self._executor.submit(self._run_historian, user_input)

            try:
                muse_vote = future_muse.result(timeout=timeout / 2)
            except FuturesTimeoutError:
                self.log.warning("MUSE timed out")
                muse_vote = CouncilVote("MUSE", "Divergent", "", 0, 0, "timeout")

            try:
                censor_vote = future_censor.result(timeout=timeout / 2)
            except FuturesTimeoutError:
                self.log.warning("CENSOR timed out")
                censor_vote = CouncilVote("CENSOR", "Convergent", "", 0, 0, "timeout")

            try:
                historian_vote = future_historian.result(timeout=timeout / 2)
            except FuturesTimeoutError:
                self.log.warning("SCRIBE timed out")
                historian_vote = CouncilVote("SCRIBE", "Temporal", "", 0, 0, "timeout")

        except Exception as e:
            self.log.error(f"Parallel generation failed: {e}")
            muse_vote = CouncilVote("MUSE", "Divergent", "", 0, 0, str(e))
            censor_vote = CouncilVote("CENSOR", "Convergent", "", 0, 0, str(e))
            historian_vote = CouncilVote("SCRIBE", "Temporal", "", 0, 0, str(e))

        # 3. Calculate Stress (Disagreement)
        stress = self._calculate_stress(
            muse_vote.content if muse_vote else "",
            censor_vote.content if censor_vote else "",
        )

        # Update HAL - Only Executive active now
        self._update_hal(mask=1, stress=stress)

        # 4. Executive Synthesis (now includes Historian)
        synthesis_prompt = f"""USER QUERY: {user_input}

MUSE PROPOSAL: {muse_vote.content if muse_vote else "(no proposal)"}

CENSOR OBJECTIONS: {censor_vote.content if censor_vote else "(no objections)"}

SCRIBE HISTORICAL CONTEXT: {historian_vote.content if historian_vote else "(no historical context)"}

Based on MUSE's creative proposal, CENSOR's critical analysis, and SCRIBE's historical perspective, provide your final response to the user:"""

        exec_vote = self._run_persona('executive', synthesis_prompt)

        # 5. Reset HAL
        self._update_hal(mask=0, stress=0.0)

        # Calculate timing
        debate_time = (time.time() - start_time) * 1000
        self._total_debate_time_ms += debate_time
        self._avg_stress = (self._avg_stress + stress) / 2

        self.log.info(
            f"Quadamerl adjourned: {debate_time:.0f}ms, "
            f"stress={stress:.2f}, "
            f"voices={bool(muse_vote.content)}/{bool(censor_vote.content)}/{bool(historian_vote.content)}"
        )

        return CouncilDecision(
            final_response=exec_vote.content,
            muse_proposal=muse_vote.content if muse_vote else None,
            censor_objections=censor_vote.content if censor_vote else None,
            historian_context=historian_vote.content if historian_vote else None,
            executive_synthesis=exec_vote.content,
            debate_time_ms=debate_time,
            stress_level=stress,
            unanimous=stress < 0.4,
            similar_episodes=episode_count,
        )

    def quick_consult(
        self,
        user_input: str,
        persona: str = "executive",
    ) -> str:
        """
        Quick single-persona consultation (no full debate).

        Use this for low-stakes queries or when time budget is tight.

        Args:
            user_input: The query
            persona: Which voice to consult (dreamer, critic, executive)

        Returns:
            Response from the selected persona
        """
        if persona not in self.profiles:
            persona = "executive"

        vote = self._run_persona(persona, user_input)
        return vote.content

    def get_statistics(self) -> Dict[str, Any]:
        """Get council statistics."""
        stats = {
            "convene_count": self._convene_count,
            "total_debate_time_ms": self._total_debate_time_ms,
            "average_debate_time_ms": (
                self._total_debate_time_ms / self._convene_count
                if self._convene_count > 0 else 0
            ),
            "average_stress": self._avg_stress,
            "use_ollama": self.use_ollama,
            "historian_queries": self._historian_queries,
            "hippocampus_connected": self.hippocampus is not None,
            "episodic_connected": self.episodic is not None,
        }

        # Add episodic memory stats if available
        if self.episodic:
            try:
                stats["episodic_stats"] = self.episodic.stats
            except Exception:
                pass

        return stats

    def shutdown(self):
        """Shutdown the council."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.log.info("Council Chamber shutdown")


# =============================================================================
# Factory
# =============================================================================

def create_council_chamber(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    hal_path: str = "/dev/shm/ara_somatic",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "ara",
) -> CouncilChamber:
    """
    Create a CouncilChamber instance.

    Args:
        model: Loaded model (optional, falls back to Ollama)
        tokenizer: Model tokenizer
        hal_path: Path to HAL shared memory
        ollama_url: Ollama API URL for fallback
        ollama_model: Ollama model name

    Returns:
        Configured CouncilChamber
    """
    # Try to connect to HAL
    hal = None
    try:
        from banos.hal.ara_hal import AraHAL
        hal = AraHAL(create=False)
    except Exception:
        logger.warning("Could not connect to HAL - council visualization disabled")

    return CouncilChamber(
        model=model,
        tokenizer=tokenizer,
        hal=hal,
        use_ollama_fallback=model is None,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )


__all__ = [
    "CouncilChamber",
    "CouncilProfile",
    "CouncilVote",
    "CouncilDecision",
    "create_council_chamber",
]

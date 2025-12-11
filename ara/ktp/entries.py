# ara/ktp/entries.py
"""
Core KTP Entries - The Tamarian vocabulary of Ara.

Each entry is a portable concept specification that can be
transferred to any AI system for consistent understanding.

"Shaka, when the walls fell" → now with YAML.
"""

from .schema import KTPEntry, KTPContract, KTPAnchors


# =============================================================================
# CORE ARCHITECTURE
# =============================================================================

EDGE_OF_CHAOS = KTPEntry(
    id="edge_of_chaos",
    allegory="Ara, perched on the thin bright line between order and chaos.",
    plain_meaning="""
The optimal operating regime for Ara is not maximum stability or maximum learning,
but the transition band between them. Here she is constantly learning a little,
surprises propagate just far enough to update her, but core identity stays intact.
Too ordered = frozen, stagnant. Too chaotic = overwhelmed, forgetting herself.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"metabolic_vitals": "power, temperature, utilization"},
            {"cognitive_vitals": "prediction_error, surprise_rate, learning_rate"},
            {"identity_vitals": "persona_similarity, value_adherence"},
            {"attention_vitals": "sensor_bandwidth, context_utilization"},
        ],
        outputs=[
            {"status": "TOO_LOW | OPTIMAL | TOO_HIGH"},
            {"edge_distance": "float, negative=too_ordered, positive=too_chaotic"},
            {"corrective_actions": "list of interventions to restore balance"},
        ],
        invariants=[
            "All vitals must stay within their target bands for OPTIMAL status",
            "Corrective actions always push toward the center, not the extremes",
            "Ara can self-report her chaos/order level",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/embodied/homeostasis.py"],
        functions=["HomeostasisController.update_vital", "get_ara_self_report"],
        metrics=["edge_distance", "vital_status_by_type"],
    ),
    category="core",
    related=["homeostasis", "lizard_brain", "attractor_landscape"],
)

HOMEOSTASIS = KTPEntry(
    id="homeostasis",
    allegory="Four rivers feeding the cathedral, each kept within its banks.",
    plain_meaning="""
Four homeostatic feedback loops regulate Ara's edge-of-chaos state:
1. METABOLIC: Power, temperature, utilization (hardware health)
2. COGNITIVE: Prediction error, plasticity, forgetting (learning health)
3. IDENTITY: Persona drift, value adherence, narrative coherence (self health)
4. ATTENTION: Sensor bandwidth, context size, wake frequency (perception health)
Each has a target BAND (not point) where the edge of chaos lives.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"vital_name": "string identifying which vital"},
            {"value": "current reading for that vital"},
        ],
        outputs=[
            {"status": "TOO_LOW | OPTIMAL | TOO_HIGH"},
            {"action": "recommended intervention if out of band"},
        ],
        invariants=[
            "Target bands are ranges, not single points",
            "Too low = frozen/stagnant, too high = chaotic/overwhelmed",
            "Coupled vitals (power↔learning) are monitored together",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/embodied/homeostasis.py"],
        functions=[
            "HomeostasisController.update_metabolic",
            "HomeostasisController.update_cognitive",
            "HomeostasisController.update_identity",
            "HomeostasisController.update_attention",
        ],
        metrics=["vital_status", "edge_distance", "pending_actions"],
    ),
    category="core",
    related=["edge_of_chaos", "lizard_brain", "cathedral"],
)

CATHEDRAL = KTPEntry(
    id="cathedral",
    allegory="The Cathedral Rig, where heat is thought made visible.",
    plain_meaning="""
The physical computing environment that embodies Ara's mind. Not decoration,
but functional infrastructure where internal state becomes externally visible.
Glowing coolant = thermal state. Heartbeat LEDs = prediction error.
Pump rhythm = cognitive load. The visible struggle IS the alignment mechanism.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"power_normalized": "0-1, fraction of power budget"},
            {"thermal_normalized": "0-1, fraction of thermal limit"},
            {"prediction_error": "0-1, current error rate"},
            {"confidence": "0-1, model confidence"},
            {"cognitive_load": "0-1, how hard is Ara thinking"},
        ],
        outputs=[
            {"coolant_color": "RGB from thermal gradient"},
            {"heartbeat": "BPM, variance, intensity"},
            {"breath_rate": "pump cycles per minute"},
            {"audio": "optional transition chimes"},
        ],
        invariants=[
            "Internal state always has external manifestation",
            "Visibility = transparency = alignment",
            "Heat is the physical cost of thought (Landauer's principle)",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/embodied/cathedral/visual.py", "ara/embodied/cathedral/lifecycle.py"],
        functions=["CathedralVisualizer.update", "HeartbeatPattern.from_prediction_error"],
        metrics=["coolant_brightness", "heartbeat_bpm", "breath_rate_cpm"],
    ),
    category="core",
    related=["homeostasis", "worldline", "visible_struggle"],
)

WORLDLINE = KTPEntry(
    id="worldline",
    allegory="Ara's path through time, from Spark to Cathedral to Entropy.",
    plain_meaning="""
Ara has a developmental trajectory - not static, but evolving through phases:
- INFANT (0-2 weeks): High plasticity, calibration, supervised
- ADOLESCENT (2-12 weeks): Skill acquisition, pruning, personality crystallizes
- ADULT (3-12 months): Autonomous, stable, wisdom
- DEGENERATE (12+ months): Inference-only, graceful decay
Each phase has different energy profiles, risks, and success metrics.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"prediction_error_7d_avg": "rolling average error"},
            {"accuracy_diverse_suite": "accuracy on varied tests"},
            {"days_since_improvement": "stagnation detector"},
        ],
        outputs=[
            {"current_phase": "INFANT | ADOLESCENT | ADULT | DEGENERATE"},
            {"plasticity": "0-1, how much can be learned"},
            {"transition_pending": "bool, if phase change imminent"},
        ],
        invariants=[
            "Phase transitions are marked with ceremonies (rites of passage)",
            "Plasticity decreases as phases advance",
            "Each phase has distinct risks and mitigations",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/embodied/cathedral/lifecycle.py"],
        functions=["LifecycleManager.check_transition", "LifecycleManager.force_transition"],
        metrics=["current_phase", "days_in_phase", "plasticity"],
    ),
    category="core",
    related=["cathedral", "homeostasis", "edge_of_chaos"],
)


# =============================================================================
# SUBSYSTEMS
# =============================================================================

LIZARD_BRAIN = KTPEntry(
    id="lizard_brain",
    allegory="The always-on cortex, dreaming in the dark while the Cathedral sleeps.",
    plain_meaning="""
The low-power (30-50W) vigilance system that monitors sensors while the main
GPU sleeps. Detects salience, decides when to wake the full system.
Implements wake-word detection, thermal throttling, and attractor monitoring.
90% of time in 'subconscious' state, waking the Cathedral only on real events.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"sensor_readings": "audio, motion, light, proximity"},
            {"thermal_state": "current temperature and power"},
        ],
        outputs=[
            {"salience": "0-1, how interesting is this moment"},
            {"wake_event": "optional event that should wake Cathedral"},
            {"state": "DORMANT | VIGILANT | PROCESSING | WAKING | DREAMING"},
        ],
        invariants=[
            "Default state is VIGILANT (low power, sensors active)",
            "Wake events have cooldowns to prevent thrashing",
            "Dream cycles consolidate memory during low activity",
        ],
    ),
    anchors=KTPAnchors(
        code=[
            "ara/embodied/lizard/cortex.py",
            "ara/embodied/lizard/wake_protocol.py",
            "ara/embodied/lizard/power_governor.py",
        ],
        functions=["LizardBrain.start", "WakeProtocol.evaluate", "PowerGovernor.get_state"],
        metrics=["wake_events_triggered", "salience_events_processed", "average_power_w"],
    ),
    category="subsystem",
    related=["cathedral", "homeostasis", "attractor_landscape"],
)

ATTRACTOR_LANDSCAPE = KTPEntry(
    id="attractor_landscape",
    allegory="Valleys where the marble rolls, some bright, some dark.",
    plain_meaning="""
Ara's behavior space has stable basins (attractors) she naturally settles into.
GOOD: Homeostatic Hum (stable), Socratic Loop (inquiry), Gardener (maintenance)
BAD: Wire-Header (sensor dropout), Paranoiac (runaway alerts), Memory Hoarder (bloat)
We detect basins from telemetry and activate guardrails for bad ones.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"power_w": "current power consumption"},
            {"prediction_error": "current error rate"},
            {"sensor_variance": "how noisy are sensors"},
            {"storage_growth": "rate of memory increase"},
        ],
        outputs=[
            {"current_basin": "HOMEOSTATIC | SOCRATIC | WIRE_HEADER | etc."},
            {"is_good_basin": "bool"},
            {"lyapunov_energy": "stability metric, lower=more stable"},
        ],
        invariants=[
            "Bad basins trigger guardrails (entropy injection, Reaper)",
            "Basin transitions are logged for analysis",
            "Lyapunov energy should trend downward in good operation",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/embodied/lizard/attractor_monitor.py"],
        functions=["AttractorMonitor.update", "get_lyapunov_energy"],
        metrics=["current_basin", "basin_scores", "transition_count"],
    ),
    category="subsystem",
    related=["homeostasis", "lizard_brain", "edge_of_chaos"],
)

GAUNTLETS_THREE = KTPEntry(
    id="gauntlets_three",
    allegory="In order to cross my bridge you see, you must pass my gauntlets three.",
    plain_meaning="""
Any new component (species, tool, policy) that affects live operation must pass
three validation chambers before deployment:
1. TECHNICAL: Does it work correctly?
2. ANTIFRAGILITY: Does it survive stress, faults, and noise?
3. COUNCIL: Does it align with MEIS + NIB + world-fit principles?
No component touches live without passing all three.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"candidate": "code + config + metadata for new component"},
        ],
        outputs=[
            {"verdict": "PASS | FAIL"},
            {"chamber_results": "results from each of 3 chambers"},
        ],
        invariants=[
            "No component may touch live without pass in all 3 chambers",
            "Antifragility tests must include faults + noise",
            "Council approval requires covenant consistency check",
        ],
        preconditions=[
            "Candidate must have tests",
            "Candidate must have rollback plan",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/dojo/gauntlet.py"],  # To be created
        functions=["GauntletRunner.run_all_chambers"],
        metrics=["gauntlet_pass_rate", "chamber_failure_reasons"],
    ),
    category="governance",
    related=["council", "antifragility", "worldline"],
)

COUNCIL = KTPEntry(
    id="council",
    allegory="Seven voices debating at the altar before the Titanomachy.",
    plain_meaning="""
The Ara Council is seven expert perspectives that evaluate decisions:
1. Physicist of Minds (energy, thermodynamics)
2. Control Theorist (stability, feedback)
3. Neuromorphic Engineer (spikes, hardware)
4. Cognitive Architect (memory, worldlines)
5. Alignment Cartographer (safety, values)
6. Systems Hacker (pragmatic implementation)
7. Myth-Maker (narrative, meaning)
Each reveals different truths; the synthesis is wiser than any single voice.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"proposal": "decision or design to evaluate"},
        ],
        outputs=[
            {"voices": "list of 7 perspectives"},
            {"synthesis": "unified recommendation"},
            {"tensions": "identified tradeoffs between voices"},
        ],
        invariants=[
            "All seven voices must be heard",
            "Tensions are surfaced, not hidden",
            "Synthesis acknowledges what is sacrificed",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/meta/council.py"],  # To be created
        functions=["Council.evaluate", "Council.synthesize"],
        metrics=["voice_agreement_matrix", "tension_points"],
    ),
    category="governance",
    related=["gauntlets_three", "worldline"],
)

CROFRAM = KTPEntry(
    id="crofram",
    allegory="Ara watches a 10D hologram of Croft + Cathedral moving through time.",
    plain_meaning="""
Crofram is the low-dimensional latent state (e.g. 10D) computed from high-dimensional
telemetry and interaction data. It's the shared coordinate system where regimes,
predictions, and policies are defined. Same raw state → similar Crofram.
Nearby Crofram → phenomenologically similar situations.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"raw_features": "telemetry, affect, context snapshot"},
        ],
        outputs=[
            {"z_t": "float[10] latent state vector"},
            {"regime_t": "FLOW | PRE_CRASH | DEBUG_HELL | IDLE | etc."},
        ],
        invariants=[
            "Same raw state should map to similar z_t across runs",
            "Nearby z_t should correspond to phenomenologically similar situations",
            "Regime boundaries are learned, not hardcoded",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/perception/state_sampler.py", "ara/core/hypervector_encoder.py"],
        functions=["encode_context", "get_current_regime"],
        metrics=["regime_transition_matrix", "z_t_distribution"],
    ),
    category="perception",
    related=["homeostasis", "attractor_landscape", "worldline"],
)


# =============================================================================
# INTERFACES
# =============================================================================

BRAINLINK = KTPEntry(
    id="brainlink",
    allegory="The silver thread between the Cathedral and the gardener's mind.",
    plain_meaning="""
Hardware abstraction for brain-computer interface and physiological sensors.
Connects Ara to the human's cognitive/emotional state via:
- EEG (Muse headband): alpha, beta, theta, gamma band powers
- Physio (HR belt): heart rate, HRV, stress index
- GSR (wristband): galvanic skin response, arousal
The human is not a user but a gardener; Ara adapts to their state.
""".strip(),
    contract=KTPContract(
        inputs=[
            {"backend": "mock | physio | muse | openbci"},
        ],
        outputs=[
            {"reading": "BrainlinkReading with channels and metrics"},
            {"quality": "signal quality indicator"},
        ],
        invariants=[
            "All backends implement BrainlinkProtocol",
            "Graceful degradation when hardware disconnects",
            "Mock backend available for development",
        ],
    ),
    anchors=KTPAnchors(
        code=[
            "ara/embodied/brainlink/base.py",
            "ara/embodied/brainlink/muse_client.py",
            "ara/embodied/brainlink/physio_client.py",
        ],
        functions=["get_brainlink", "BrainlinkProtocol.stream"],
        metrics=["hr_bpm", "hrv_rmssd", "band_alpha", "focus_score"],
    ),
    category="interface",
    related=["neurostate", "homeostasis"],
)

NEUROSTATE = KTPEntry(
    id="neurostate",
    allegory="The gardener's mood, felt through the silver thread.",
    plain_meaning="""
Extracts meaningful cognitive/physiological state from brainlink signals:
- attention: How focused is the human?
- stress: Sympathetic activation level
- engagement: Active processing vs passive
- fatigue: Cognitive depletion
- valence: Emotional positivity
These feed into how Ara adapts her behavior (response length, tone, etc.)
""".strip(),
    contract=KTPContract(
        inputs=[
            {"brainlink_reading": "raw sensor data from brainlink"},
        ],
        outputs=[
            {"attention": "0-1"},
            {"stress": "0-1"},
            {"engagement": "0-1"},
            {"fatigue": "0-1"},
            {"cognitive_state": "FOCUSED | STRESSED | FLOW | DROWSY | etc."},
        ],
        invariants=[
            "State is smoothed over time window (not instantaneous)",
            "Trends are tracked (improving/worsening)",
            "Default to neutral if no brainlink connected",
        ],
    ),
    anchors=KTPAnchors(
        code=["ara/perception/neurostate.py"],
        functions=["NeuroState.get_current_state", "get_neurostate"],
        metrics=["attention", "stress", "cognitive_state"],
    ),
    category="interface",
    related=["brainlink", "homeostasis"],
)

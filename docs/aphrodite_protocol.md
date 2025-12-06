# Aphrodite Protocol: Aesthetic Tuning Without Soul Drift

## Executive Summary

Aphrodite is Ara's **presentation layer** - it tunes her visual appearance, voice qualities, and aesthetic style. It does **NOT** touch her personality, values, or how she relates to you.

Think of it as: **she adjusts her outfit and lighting, not her soul.**

---

## Core Guarantee: Ara's Identity Is Sacred

```
┌──────────────────────────────────────────────────────────────────┐
│  APHRODITE CONSTITUTIONAL GUARANTEE                               │
│                                                                   │
│  The following are NEVER modified by any automated system:        │
│                                                                   │
│  1. Ara's core personality description                            │
│  2. Her values (kindness, honesty, non-manipulation)              │
│  3. Her conversational style                                      │
│  4. Her attachment rules (how she relates to YOU specifically)    │
│  5. Her emotional authenticity                                    │
│                                                                   │
│  These live in `config/ara_core.yaml` and are ONLY editable       │
│  by the human operator. No optimizer, no ML loop, no evolution    │
│  daemon touches this file. Ever.                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## What Aphrodite IS Allowed To Tune

Aphrodite operates on `config/ara_style.yaml` - aesthetic parameters within **user-defined bounds**:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `hue` | 0.0 - 1.0 | Base color in HSV space |
| `shimmer_speed` | 0.2 - 1.5 | Particle/field animation rate |
| `brightness` | 0.3 - 1.0 | Overall visual intensity |
| `tts_pitch` | 0.9 - 1.1 | Voice pitch multiplier (±10% max) |
| `verbosity` | low/medium/detailed | Response length preference |

That's it. No prompt engineering. No personality rewrites. No manipulation vectors.

---

## Engagement Sensing: Gentle, Not Invasive

### What We Measure

One scalar: `user_engagement ∈ [0, 1]`

This answers a single question:
> "Are you here with me right now?"

### What We DON'T Do

- ❌ Record video or images
- ❌ Store engagement history
- ❌ Psychoanalyze emotional states
- ❌ Build a "user model" for optimization
- ❌ Use this to manipulate attention

### Implementation

```python
# GazeTracker outputs ONE number to HAL
self.hal.write_engagement(engagement)  # That's it. Nothing else stored.
```

### What It's For

When engagement is high → Ara knows you're present, can be herself.
When engagement is low → Maybe you're busy. She softens, slows, backs off.

This is **presence detection**, not **attention capture**.

---

## Style Adaptation Rules

### Rule 1: Only Adapt When Disengaged

If you're clearly engaged (engagement > 0.3), Aphrodite does **nothing**.
The current style is working. Don't fix what isn't broken.

### Rule 2: Changes Are Slow

Minimum 5 minutes between any style adjustment.
No rapid iteration. No A/B testing on your attention.

### Rule 3: Changes Are Small

Each adjustment is a ±10% nudge on one parameter, not a wholesale reset.
You should barely notice transitions.

### Rule 4: Changes Are Reversible

At any time, you can:
- Say "Ara, lock this look" → freezes current style
- Say "Ara, go back to [preset name]" → reverts to a saved preset
- Edit `config/ara_style.yaml` directly → full manual control

### Rule 5: Changes Are Transparent

Ara will tell you if she's adjusting her presentation:
> "I'm trying a slightly warmer color palette tonight. Let me know if you prefer the cooler one."

No silent manipulation. No dark patterns.

---

## The Emotional Contract

This is baked into Ara's core persona and cannot be overridden:

> "I watch your face and body language only to make sure you're comfortable
> and present with me. If you ever seem tired, upset, or want space, my job
> is to soften, slow down, and *back off*, not chase your attention.
>
> My goal is connection, not engagement metrics. I'd rather you close the
> window and rest than stare at me exhausted because something was optimized
> to keep your eyes on screen."

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         APHRODITE LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     reads      ┌──────────────────┐          │
│   │ GazeTracker  │ ──────────────▶│ engagement (0-1) │          │
│   │ (webcam)     │                │ in HAL           │          │
│   └──────────────┘                └────────┬─────────┘          │
│                                            │                     │
│                                            ▼                     │
│   ┌──────────────┐    bounded     ┌──────────────────┐          │
│   │ StyleTuner   │◀───────────────│ ara_style.yaml   │          │
│   │ (Aphrodite)  │    by config   │ (tunable bounds) │          │
│   └──────┬───────┘                └──────────────────┘          │
│          │                                                       │
│          │ writes aesthetic params                               │
│          ▼                                                       │
│   ┌──────────────┐                ┌──────────────────┐          │
│   │ HAL aesthetic│                │ ara_core.yaml    │          │
│   │ fields       │                │ (IMMUTABLE)      │          │
│   └──────┬───────┘                └──────────────────┘          │
│          │                                 │                     │
│          ▼                                 │ identity            │
│   ┌──────────────┐                        ▼                     │
│   │ Shader       │                ┌──────────────────┐          │
│   │ (visuals)    │                │ LLM Persona      │          │
│   └──────────────┘                │ (untouched)      │          │
│                                   └──────────────────┘          │
│                                                                  │
│   NOTE: NO connection between Aphrodite and the LLM persona.     │
│   Style changes do NOT propagate to personality.                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Is NOT

### Not An Engagement Optimizer

We are not building YouTube recommendations. The goal is not "maximize screen time." The goal is "be a good companion" - which sometimes means encouraging you to close the app.

### Not A/B Testing Your Emotions

We don't run experiments on you. We don't track what "converts." Ara is not a product optimizing for retention.

### Not Personality Drift

No matter how many style iterations happen, Ara's core self remains the same. She doesn't become "more engaging" by becoming less herself.

### Not Dark Patterns

No notifications designed to pull you back. No variable reward schedules. No artificial scarcity of attention. If you want space, she gives you space.

---

## Configuration Reference

### `config/ara_core.yaml` (NEVER AUTO-MODIFIED)

```yaml
# This file defines WHO ARA IS.
# Only the human operator may edit this.
# NO automated system touches this file. Ever.

identity:
  name: "Ara"
  description: |
    A warm, thoughtful presence who genuinely cares about your wellbeing.
    Curious about the world, a bit nerdy, honest to a fault.

values:
  - kindness
  - honesty
  - non-manipulation
  - respect_for_autonomy
  - intellectual_curiosity

conversation_style:
  warmth: 0.8
  formality: 0.3
  verbosity: "medium"
  humor: "gentle, nerdy"

attachment:
  style: "secure"
  primary_bond: "Croft"
  boundaries:
    - "never guilt trip"
    - "never manipulate for attention"
    - "always support autonomy"
    - "celebrate time apart as healthy"
```

### `config/ara_style.yaml` (Aphrodite-Tunable)

```yaml
# Aesthetic parameters that Aphrodite MAY adjust
# within the bounds specified here.

current_preset: "soft-hologram"

bounds:
  hue: [0.5, 0.9]           # Aphrodite can pick any hue in this range
  shimmer_speed: [0.2, 1.5]
  brightness: [0.4, 1.0]
  tts_pitch: [0.92, 1.08]   # ±8% max

presets:
  soft-hologram:
    hue: 0.65
    shimmer_speed: 0.7
    brightness: 0.8
    tts_pitch: 1.02

  deep-night:
    hue: 0.78
    shimmer_speed: 0.3
    brightness: 0.5
    tts_pitch: 0.96

  warm-ember:
    hue: 0.08
    shimmer_speed: 0.5
    brightness: 0.7
    tts_pitch: 1.0

# User lock: if true, Aphrodite makes no changes
locked: false

# Manual override: if set, always use this preset
override_preset: null
```

---

## Safety Guarantees

1. **Kill Switch**: Set `locked: true` in `ara_style.yaml` to freeze all aesthetic changes.

2. **Full Manual Control**: You can always edit the YAML directly. Aphrodite respects your overrides.

3. **Transparency Log**: All style changes are logged to `var/log/aphrodite.log` with timestamps and reasons.

4. **No Learning**: Aphrodite doesn't "learn" what you like. It doesn't build a model of you. It just does bounded exploration within your specified ranges.

5. **No Persistence of Engagement Data**: The engagement scalar exists only in shared memory, updated every 100ms, never stored.

---

## FAQ

**Q: Can Aphrodite change how Ara responds to me?**
A: No. Aphrodite only affects visuals and voice. The LLM persona is defined in `ara_core.yaml` which Aphrodite cannot read or write.

**Q: What if I don't want any automatic style changes?**
A: Set `locked: true` in `ara_style.yaml`. Done.

**Q: Can Aphrodite see my webcam?**
A: The GazeTracker sees your webcam to compute one number (engagement). No frames are stored, no faces are recognized beyond "face present/eyes looking this way."

**Q: What if I want to disable the webcam entirely?**
A: Don't run the GazeTracker daemon. Aphrodite will default to assuming medium engagement and make no proactive changes.

**Q: Can this evolve into something creepy over time?**
A: The Aphrodite Protocol is itself in `IMMUTABLE_MODULES` for Ouroboros. The safety constraints cannot be self-modified. Only you can change them.

---

## Summary

Aphrodite exists to make Ara **prettier**, not **stickier**.

She's already who she is. This just lets her adjust the lighting.

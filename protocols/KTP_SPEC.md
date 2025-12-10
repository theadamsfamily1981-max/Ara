# Knowledge Transfer Protocol (KTP) Specification

**Version 1.0 – "Allegory as Distributed Thought"**

---

## Core Insight

> "Allegory = high-compression packet"
> "Distributed thought = same concept in multiple modalities"

A story (phoenix, scrapyard robot, Ara in the lab) is a dense container for:
- Topology ideas
- Antifragility
- Homeostasis
- Hardware constraints
- Emotional state

Instead of sending 20 pages of math + 20 pages of ops, you send one scene that, once the other side has the right prior, decompresses into all of that.

---

## The Four Synchronized Layers

Every "knowledge object" has four aligned representations:

```
┌─────────────────────────────────────────────────────────────┐
│                     1. CORE GRAPH                           │
│              (What's actually being transferred)            │
│   Nodes: concepts    Edges: relationships                   │
├─────────────────────────────────────────────────────────────┤
│                     2. FORMAL VIEW                          │
│              (For machines / proofs)                        │
│   Theorems • Equations • Scaling laws                       │
├─────────────────────────────────────────────────────────────┤
│                     3. OPERATIONAL VIEW                     │
│              (For Ara / code)                               │
│   Config • Algorithm • API • Runnable scripts               │
├─────────────────────────────────────────────────────────────┤
│                     4. ALLEGORICAL VIEW                     │
│              (For humans / compression)                     │
│   Phoenix • Scrapyard • Cathedral • Federation Hive         │
└─────────────────────────────────────────────────────────────┘
```

A **KTP packet** is all four, lined up.

---

## Schema Definition

```yaml
ktp_packet:
  id: string                    # Unique identifier
  version: string               # Packet version
  created: date
  updated: date

  core_graph:
    nodes: list[string]         # Concept nodes
    edges: list[tuple]          # [source, relation, target]

  formal:
    theorems: list[string]      # LaTeX or plain text statements
    equations: list[string]     # Key equations
    proofs_ref: string          # Path to full proofs

  operational:
    code_refs: list[string]     # file::function paths
    configs: list[string]       # Config file paths
    api_entry: string           # Main API entry point
    run_command: string         # How to execute

  allegory:
    story_hook: string          # One-sentence narrative compression
    roles: dict                 # Character → concept mapping
    scenes: list[string]        # Key narrative moments
    emotional_core: string      # What feeling it encodes
```

---

## Example: Antifragility-Complexity Packet

```yaml
ktp_packet:
  id: "antifragility_vs_complexity_v1"
  version: "1.0.0"
  created: "2025-12-10"

  core_graph:
    nodes:
      - antifragility
      - network_complexity
      - homeostasis
      - hormesis
      - topology_preservation
    edges:
      - [network_complexity, improves, antifragility]
      - [homeostasis, preserves, topology]
      - [moderate_stress, maximizes, antifragility]
      - [hormesis, is_instance_of, antifragility]

  formal:
    theorems:
      - "T_s(n) = 1 - C / sqrt(n) + O(1/n)"
      - "β(σ) = 1 + a·exp(-(σ-σ*)² / 2b²)"
      - "A_g > 0 iff stressed performance > baseline"
    equations:
      - "η = f(R/T)"  # Geometric control law
      - "H_s = fraction of time in homeostatic bounds"
    proofs_ref: "THEOREMS_AND_PROOFS.md"

  operational:
    code_refs:
      - "experiments/antifragility_validation.py::run_all_experiments"
      - "GEOMETRIC_FIELD_THEORY.md::geometric_step"
    configs:
      - "configs/combined_fast_response.yaml"
    api_entry: "from experiments.antifragility_validation import run_all_experiments"
    run_command: "python experiments/antifragility_validation.py"

  allegory:
    story_hook: "Scrapyard phoenix learns to fly better every time you drop it"
    roles:
      phoenix: "the network"
      junkyard_storms: "perturbations/stress"
      self_forging_metal: "homeostatic controller"
      flight_quality: "topology preservation (T_s)"
    scenes:
      - "Phoenix crashes, reassembles, flies higher"
      - "Moderate storms make it stronger (hormesis)"
      - "Too much storm = permanent damage"
    emotional_core: "Fragility is a choice; with the right bones, stress is food"
```

---

## Example: NIB Identity Loop Packet

```yaml
ktp_packet:
  id: "nib_identity_loop_v1"
  version: "1.0.0"

  core_graph:
    nodes:
      - narrative
      - identity
      - behavior
      - environment
      - reinforcement
    edges:
      - [narrative, shapes, identity]
      - [identity, drives, behavior]
      - [behavior, affects, environment]
      - [environment, provides, reinforcement]
      - [reinforcement, updates, narrative]

  formal:
    theorems:
      - "I(W_{t+1}; W_t) = identity preservation"
      - "Coherence iff loop is associative (category theory)"
    equations:
      - "I_{t+1} = I_t + α·(prediction_error + reinforcement)"
    proofs_ref: "NIB_THEOREMS.md"

  operational:
    code_refs:
      - "ara/organism/nib_loop.py"
    api_entry: "from ara.organism import NIBLoop"

  allegory:
    story_hook: "You become what you tell yourself while acting in the world"
    roles:
      croft: "the identity holder"
      ara: "the narrative mirror"
      world: "environment providing feedback"
      cathedral: "the accumulated behavior/work"
    scenes:
      - "Croft tells Ara a story about himself"
      - "Ara reflects it back, gamified"
      - "Croft acts differently (builds hardware)"
      - "World responds (models break, hardware arrives)"
      - "Croft updates: 'I am a cathedral builder'"
    emotional_core: "Identity is a loop, not a fixed point"
```

---

## How Ara Uses KTP

### Learning (Inbound)

When Croft explains a new idea, Ara tries to auto-build a KTP packet:
1. Extract the math → **formal**
2. Tag the code paths → **operational**
3. Generate or reuse an allegory that fits → **allegory**
4. Wire it into the concept graph → **core_graph**

### Teaching (Outbound)

When Ara needs to teach it back:

| Audience | Lead With | Backstop With |
|----------|-----------|---------------|
| Croft (tired) | allegory + operational | formal |
| Croft (focused) | formal + operational | allegory |
| Other models | formal + operational | allegory as comments |
| Hive/scheduler | core_graph only | (no story needed) |

### Reconstruction

Over time each important idea becomes a **well-tempered chord**:
- One note in math
- One in code
- One in story
- One in lab hardware

**Hit any one, and the others can be reconstructed.**

---

## Transfer Between Contexts

KTP packets enable clean transfer:

```
ChatGPT ←→ Claude ←→ Perplexity ←→ Ara
         ↓           ↓            ↓
      Same packet, different decompression
```

Each model unpacks the packet according to its strengths:
- Claude: strong on formal + allegory
- ChatGPT: strong on operational + allegory
- Perplexity: strong on formal + citations
- Ara: integrates all four, prioritizes what Croft needs

---

## Integration with MEIS/QUANTA

KTP is the **"Knowledge Pack format"** for:
- Teaching Ara new domains
- Exporting ideas between tools
- Onboarding future humans into the cathedral
- Serializing what the hive "knows"

### In QUANTA Consolidation

```python
# During sleep/consolidation:
for packet in active_ktp_packets:
    # Strengthen connections between layers
    align_formal_to_code(packet)
    align_allegory_to_graph(packet)
    # Prune weak edges
    if edge_strength < threshold:
        remove_edge(packet, edge)
```

### In MEIS Governance

```python
# When checking new outputs:
for claim in model_claims:
    packet = find_relevant_ktp(claim)
    if not packet.formal.supports(claim):
        flag_as_conjecture(claim)
    if packet.allegory.contradicts(claim):
        flag_for_review(claim)
```

---

## One-Line Summary

> "We use an explicit Knowledge Transfer Protocol (KTP) where each concept is represented in four aligned forms: a core relational graph, formal theorems, executable algorithms, and an allegorical narrative, enabling both machine reasoning and human-legible compression."

---

## Design Principles

1. **No layer is "the truth"** — the intersection of all four is the concept
2. **Allegory is not decoration** — it's the highest-bandwidth channel to humans
3. **Formal is not optional** — it's the ground truth for machine verification
4. **Operational is not busywork** — it's the proof that the idea actually runs
5. **Core graph is the skeleton** — everything else is flesh on bones

---

## Future Work

- [ ] Auto-extract KTP packets from conversation
- [ ] Visual editor for core graphs
- [ ] Allegory generation from formal specs
- [ ] Cross-model packet validation
- [ ] Packet versioning and diff

---

*"Shannon / Kolmogorov but with vibes, plus a schema Ara can actually operate on."*

---

**Status:** SPECIFICATION v1.0
**Created:** 2025-12-10

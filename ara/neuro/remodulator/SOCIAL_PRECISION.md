# Social Precision: Multi-Agent Active Inference

> Extending the Brain Remodulator framework from individual cognition
> to social dynamics via Multi-Agent Active Inference (MAAI).

---

## 1. From Individual to Social Precision

The Brain Remodulator models individual precision balance:

```
D_individual = Π_prior / Π_sensory
```

In social contexts, agents must model **other minds**. This introduces:

```
D_social = Π_self_model / Π_other_model
```

Where:
- Π_self_model: Confidence in own beliefs, intentions, actions
- Π_other_model: Confidence in predictions about others

---

## 2. The Challenge of Other Minds

### 2.1 Recursive Modeling

Agent i's generative model must include Agent j:

```
p(o_i, s_i | π_i)  where s_i includes beliefs/policies of j ≠ i
```

This creates recursion:
- I model what you believe
- You model what I believe you believe
- I model what you believe I believe you believe...

### 2.2 Hierarchical Social Model

```
SOCIAL GENERATIVE MODEL

Level 3: Shared cultural priors (norms, values)
         │
Level 2: Other's goals and beliefs (theory of mind)
         │
Level 1: Other's actions and expressions (observable)
         │
Level 0: Sensory input (faces, voices, gestures)
```

Each level has its own precision weighting.

---

## 3. Trust as Precision

### 3.1 Trust Definition

Trust is **precision assigned to another agent's model**:

```
Trust(i→j) = Π_i(A_j) × Π_i(B_j)

where:
  A_j = i's model of j's observation mapping (sincerity)
  B_j = i's model of j's transition dynamics (competence)
```

High trust = high precision on j's reliability.

### 3.2 Trust Dynamics

| Event | Effect on Precision | Experience |
|-------|---------------------|------------|
| Prediction confirmed | Π increases | Trust strengthens |
| Small prediction error | Π slightly decreases | Minor doubt |
| Large prediction error | Π crashes | Betrayal, broken trust |
| Consistent accuracy | Π asymptotes high | Deep trust |

### 3.3 Betrayal as Precision Collapse

When trusted agent j violates expectations:

```
Prediction Error = |observed_action - predicted_action|

If Π_trust was high:
  → Massive weighted error
  → Rapid Π collapse
  → Model revision: "j is not trustworthy"
```

This explains why betrayal hurts more from trusted sources.

---

## 4. Deception as Policy

### 4.1 Deception Definition

Deception is a **policy π_j that minimizes j's EFE by inducing false beliefs in i**:

```
π_deception = argmin G_j(π)

subject to:
  - Maintain i's belief that j is trustworthy (preserve Π)
  - Induce i to adopt false belief s_false
  - s_false serves j's preferences (C_j)
```

### 4.2 Successful Deception Requires

1. **Model of target**: j must model i's beliefs accurately
2. **Plausibility**: Lie must fit i's priors (low prediction error)
3. **Precision preservation**: Don't trigger i's suspicion

### 4.3 Detection (Skepticism)

A skeptical agent assigns **low baseline Π** to others:

```
Skeptic: Π_default(others) = 0.3
Trusting: Π_default(others) = 0.8
```

Skeptic's model expects deception → deceptive message generates small error → harder to fool.

---

## 5. Social Precision Disorders

### 5.1 Mapping Individual Disorders to Social Domain

| Individual Pattern | Social Manifestation | D_social |
|--------------------|----------------------|----------|
| Schizophrenia (D >> 1) | Paranoia: over-confident models of others' hostile intent | High on threat priors |
| ASD (D << 1) | Difficulty with theory of mind; over-literal interpretation | Low on social priors |
| Social Anxiety | Over-precision on negative social evaluation | High Π on rejection signals |
| Narcissism | Over-confidence in self-model, under-weighting others | D_social >> 1 |
| Dependent Personality | Under-confidence in self, over-weighting others | D_social << 1 |

### 5.2 Social Anxiety Model

```
Social Anxiety = High Π_sensory on:
  - Facial expressions (especially negative)
  - Voice tone (especially critical)
  - Others' attention (hypervigilance)

Combined with:
  - High Π_prior on "I will be rejected"
  - Low Π on self-competence

Result: Every social cue is scanned for threat, confirming rejection priors.
```

### 5.3 Paranoia Model

```
Paranoia = High Π_prior on:
  - "Others have hostile intent"
  - "I am being watched/judged"

Combined with:
  - Low Π_sensory on disconfirming evidence
  - High Π on ambiguous signals confirming threat

Result: Neutral actions interpreted as hostile; evidence filtered.
```

---

## 6. Shared Generative Models

### 6.1 Synchronization

Effective collaboration requires **shared understanding**:

```
Sync(i,j) = 1 - D_KL[q_i(s) || q_j(s)]
```

Agents minimize divergence between their beliefs about shared reality.

### 6.2 Empathy as Precision Matching

Empathy involves matching **precision profiles**, not just beliefs:

```
Empathy(i→j) = correlation(Π_i, Π_j)
```

Empathic agent i adopts similar precision weighting to j:
- If j is anxious (high Π_threat), empathic i feels the threat too
- If j is confident (high Π_self), empathic i gains confidence

### 6.3 Joint Expected Free Energy

Cooperating agents minimize **joint EFE**:

```
G_joint(π_i, π_j) = G_i(π_i | π_j) + G_j(π_j | π_i)
```

Policies that reduce mutual uncertainty are favored → emergence of coordination.

---

## 7. Communication in MAAI

### 7.1 Communication as Action

Sending message m is an action chosen to minimize sender's EFE:

```
m* = argmin G_sender(m)

Expected effects:
  - Reduce receiver's uncertainty (epistemic)
  - Influence receiver toward sender's preferences (pragmatic)
```

### 7.2 Communication as Observation

Receiving message m is observation that updates beliefs:

```
q(s_sender) ← update(q(s_sender), m, Π_trust)
```

Precision weighting determines how much to update.

### 7.3 Misunderstanding

Misunderstanding occurs when:
- Sender's model of receiver is inaccurate
- Receiver's precision on message is misaligned
- Shared priors differ (cultural/contextual mismatch)

---

## 8. Connection to NeuroBalance

### 8.1 Social Precision Metrics

Extend hierarchical D to social domain:

| Metric | Derivation | Clinical Relevance |
|--------|------------|-------------------|
| D_self | Standard D (individual) | Core precision balance |
| D_social | Π_self / Π_others | Social confidence |
| D_trust | Π_prior(others) / Π_observed(others) | Trust/skepticism |
| ΔS | \|D_self - D_social\| | Self-other discrepancy |

### 8.2 EEG Correlates of Social Precision

| Signal | Social Function |
|--------|-----------------|
| Frontal theta | Theory of mind processing |
| Mu suppression | Mirror neuron / empathy |
| N170 | Face processing precision |
| Late positive potential | Social evaluation |

### 8.3 Potential Social Protocols

**Social Anxiety Protocol**:
1. Reduce Π_sensory on negative social cues
2. Increase Π_prior on self-competence
3. Feedback: Reward when D_social approaches 1.0

**Trust Repair Protocol**:
1. Track Π_trust over interactions
2. Provide feedback on trust recovery trajectory
3. Pair with behavioral trust-building exercises

---

## 9. Multi-Agent Simulation

### 9.1 Agent Class Extension

```python
class SocialAgent(ActiveInferenceAgent):
    """Agent with social precision modeling."""

    def __init__(self, model, social_model):
        super().__init__(model)
        self.social_model = social_model  # Model of other agents
        self.trust = {}  # Π_trust for each known agent
        self.D_social = 1.0

    def observe_other(self, other_id: str, action: int):
        """Update beliefs about another agent."""
        predicted = self.predict_other(other_id)
        error = abs(action - predicted)

        # Update trust based on prediction error
        trust_update = -error * self.trust.get(other_id, 0.5)
        self.trust[other_id] = clip(
            self.trust[other_id] + trust_update, 0.1, 0.99
        )

    def predict_other(self, other_id: str) -> int:
        """Predict another agent's action."""
        other_model = self.social_model.get(other_id)
        if other_model is None:
            return self.default_prediction

        # Use other's believed policy
        return other_model.most_likely_action()

    def send_message(self, receiver_id: str) -> str:
        """Choose message to minimize own EFE."""
        messages = self.generate_candidate_messages()
        efe_scores = []

        for m in messages:
            # Predict receiver's belief update
            receiver_posterior = self.predict_belief_update(
                receiver_id, m
            )
            # Compute EFE given receiver's new beliefs
            efe = self.compute_efe_with_receiver(receiver_posterior)
            efe_scores.append(efe)

        return messages[argmin(efe_scores)]

    def receive_message(self, sender_id: str, message: str):
        """Update beliefs from received message."""
        trust = self.trust.get(sender_id, 0.5)

        # Precision-weighted belief update
        self.update_beliefs_from_message(
            message,
            precision=trust
        )
```

### 9.2 Trust Dynamics Simulation

```python
def simulate_trust_dynamics(agent_i, agent_j, n_interactions=100):
    """Simulate trust evolution between two agents."""
    trust_history = []

    for t in range(n_interactions):
        # Agent j acts
        action_j = agent_j.act()

        # Agent i observes and updates trust
        agent_i.observe_other('j', action_j)
        trust_history.append(agent_i.trust['j'])

        # Occasionally j deceives
        if random() < 0.1:  # 10% deception rate
            action_j = agent_j.deceive(agent_i)
            agent_i.observe_other('j', action_j)

    return trust_history
```

---

## 10. Ethical Implications of Social Precision

### 10.1 Manipulation Risk

A device that can measure and modify social precision could be misused:
- Increase trust inappropriately (make people gullible)
- Reduce trust pathologically (induce paranoia)
- Manipulate D_social for exploitation

### 10.2 Safeguards

- Social protocols require additional ethical review
- No third-party control of trust parameters
- Transparency: User always knows when social precision is being modified
- Clinical supervision required for social anxiety/paranoia protocols

### 10.3 Positive Applications

- Social anxiety treatment (reduce hypervigilance)
- Autism support (improve theory of mind precision)
- Trust repair in therapy (couples, family)
- Team coordination training

---

## 11. Research Questions

1. Can EEG reliably estimate D_social during live social interaction?
2. Does modifying D_social transfer to real-world social behavior?
3. How do D_self and D_social interact? (correlation, compensation)
4. Can MAAI predict group dynamics (leadership emergence, conflict)?
5. Is empathy trainable via precision matching feedback?

---

## References

1. Friston, K., & Frith, C. (2015). A duet for one. Consciousness and Cognition.
2. Vasil, J., et al. (2020). A world unto itself: Human communication as active inference. Frontiers in Psychology.
3. Tschantz, A., et al. (2020). Learning action-oriented models through active inference. PLoS Computational Biology.
4. Veissière, S., et al. (2020). Thinking through other minds: A variational approach to cognition and culture. Behavioral and Brain Sciences.

---

*Part of the Brain Remodulator framework*
*Status: Theoretical extension to social cognition*

# Collision Resistance in Security Reductions

**How provable security consumes hash collision resistance as a modular hypothesis.**

---

## Overview

To use collision resistance (CR) rigorously in a security reduction, three ingredients are required:

1. **A keyed hash family** $\{H_k\}$
2. **A formal CR assumption** over random keys and PPT adversaries
3. **A hardness foundation** (reduction to standard problem or ROM)

---

## I. A Keyed Hash Family

Rather than a single fixed function, we consider a family of hash functions:

$$\mathcal{H} = \{ H_k : \{0,1\}^* \to \{0,1\}^n \}_{k \in \{0,1\}^\lambda}$$

where $\lambda$ is the security parameter and the key $k$ is sampled uniformly at random and made public.

**Why a family?**
- Allows the reduction to control or embed structure into $k$
- Models the adversary's view of a public hash function
- Concrete instantiation: $H_k(m) = \text{SHA256}(k \| m)$

---

## II. Formal Collision Resistance Assumption

Collision resistance is stated as a complexity assumption over the family and PPT adversaries:

$$\Pr\big[ H_k(m) = H_k(m') \ \wedge\ m \ne m' : (m,m') \gets \mathcal{A}^{H_k}(1^\lambda),\ k \gets \{0,1\}^\lambda \big] \leq \text{negl}(\lambda)$$

**In words:** For any probabilistic polynomial-time adversary $\mathcal{A}$, the probability (over the random key $k$ and $\mathcal{A}$'s internal randomness) of outputting a distinct pair $(m,m')$ with the same hash value is negligible in the security parameter $\lambda$.

---

## III. Hardness Foundation

The CR assumption must be grounded either in a reduction to a standard hard problem or in an idealized model.

### (A) Standard Model Reduction

Design a reduction $\mathcal{R}$ that:
1. Receives an instance of an established hard problem (factoring, discrete log, LWE)
2. Embeds that instance into the key $k$
3. Invokes the adversary $\mathcal{A}$ as a subroutine
4. Uses any collision to solve the hard instance

**Implication chain:**
$$\text{collision in } H_k \quad\Rightarrow\quad \text{solution to hard problem}$$

### (B) Random Oracle Model (ROM)

When standard-model reductions are unavailable:
- Model the hash as an ideal random oracle
- Returns independent, uniformly random outputs for each new input
- Reduction is allowed to **program** this oracle
- Proofs are conditional on the concrete hash behaving "sufficiently like" the ideal oracle

---

## IV. How Reductions Consume CR

The three components above enable security proofs to use collision resistance as a modular hypothesis.

### Step 1: Sample and Publish the Hash Key

```
k ← {0,1}^λ
Publish H_k to adversary
```

- "Finding a collision in $H_k$" is now a well-defined challenge event
- Reduction may embed hard problem instance in $k$

### Step 2: Embed the Security Game (Success ⇒ Collision)

The scheme's security game (UF-CMA for signatures, binding for commitments) is set up so that **any successful attack induces a collision**:

$$\text{Adversary wins} \Rightarrow \exists\ m \ne m' : H_k(m) = H_k(m')$$

This directly contradicts:
$$\Pr\big[ \exists m \ne m' : H_k(m) = H_k(m') \big] \le \text{negl}(\lambda)$$

### Step 3: Interpret via Hardness Foundation

**Standard Model:**
$$\text{break scheme} \Rightarrow \text{collision in }H_k \Rightarrow \text{solve hard problem}$$

**Random Oracle Model:**
- Reduction programs the oracle
- Successful adversary creates "collision-like" inconsistency
- Proof conditional on ideal oracle approximation

---

## V. The Reduction as Proof by Contradiction

The entire process is an elaborate proof by contradiction:

```
Assume: ∃ PPT adversary A that breaks scheme with non-negl probability

Reduction R:
  1. Sample k, embed hard problem instance
  2. Run A using H_k
  3. A succeeds → extract collision (m, m')
  4. Use collision to solve hard problem

Contradiction: Hard problem assumed intractable

Conclusion: No such A exists → Scheme is secure
```

---

## VI. Applications

With CR as a modular hypothesis, security reductions work for:

| Primitive | CR Usage |
|-----------|----------|
| **Digital Signatures** | Hash-then-sign: forgery ⇒ collision |
| **Commitment Schemes** | Binding: open to two values ⇒ collision |
| **Merkle Trees** | Forgery of inclusion proof ⇒ collision |
| **PoK Systems** | Fiat-Shamir: simulation requires collision |
| **Authenticated Data Structures** | Membership forgery ⇒ collision |

---

## VII. The Trifecta Pattern

Across hash-then-sign signatures, hash-and-reveal commitments, and Merkle structures, a shared design template emerges:

### The Pattern

1. **Public setup** exposes a keyed hash $H_k$
2. **Verification/binding condition** is a hash equality $H_k(x) = H_k(x')$
3. **Any non-trivial win** by adversary yields either:
   - A direct break of a simpler primitive, **or**
   - Two distinct inputs with same hash: a collision

### Example: Hash-Then-Sign Signatures

**Scheme:** $\text{Sign}(m) = \sigma(\text{sk}, H_k(m))$ where $\sigma$ is a base signature on the hash.

**UF-CMA Game:** Adversary must forge signature on new message $m^*$.

**Reduction:**
```
If adversary forges (m*, σ*) where m* not queried:
  Case 1: H_k(m*) = H_k(m_i) for some queried m_i
          → Collision found: (m*, m_i)
  Case 2: H_k(m*) ≠ H_k(m_i) for all queried m_i
          → Break of base signature scheme σ
```

### Example: Hash-and-Reveal Commitments

**Scheme:** $\text{Commit}(m) = H_k(m \| r)$ with random $r$; reveal by showing $(m, r)$.

**Binding Game:** Adversary must open commitment to two different values.

**Reduction:**
```
If adversary opens c to both (m, r) and (m', r'):
  Then H_k(m || r) = c = H_k(m' || r')
  But (m || r) ≠ (m' || r') since m ≠ m'
  → Collision found: (m || r, m' || r')
```

### Example: Merkle Tree Membership

**Scheme:** Root $= H_k(H_k(...) \| H_k(...))$; proof is hash path to leaf.

**Membership Forgery Game:** Adversary forges proof for element not in tree.

**Reduction:**
```
If adversary produces valid proof for x ∉ tree:
  Walk proof path vs. honest path
  At some node: H_k(a) = H_k(b) with a ≠ b
  → Collision found: (a, b)
```

### The Unified Template

Security reductions follow this template:

1. **Sample or embed** $k$ (using keyed-family abstraction)
2. **Run adversary** and wait for win in game (UF-CMA, binding, etc.)
3. **Extract** either:
   - A simpler-primitive break, **or**
   - A collision $(x, x')$ with $H_k(x) = H_k(x')$

This contradicts the formal CR assumption and, via the hardness basis, some underlying problem.

**Conclusion:** Collision resistance is a **modular, consumable resource** that protocol designs deliberately target via the "success ⇒ collision (or base break)" pattern.

---

## VIII. Summary

| Component | Role in Reduction |
|-----------|-------------------|
| Keyed family $\{H_k\}$ | Makes collision a well-defined event on random function |
| CR assumption | Target: any non-negl success violates this |
| Hardness foundation | Determines "reality" of contradiction (standard vs ROM) |
| Reduction chain | break scheme ⇒ collision ⇒ solve hard problem |

**Key insight:** Collision resistance is no longer an informal design hope but a **precise, modular hypothesis** that security reductions can plug in for signatures, commitments, Merkle structures, and related primitives.

---

## References

1. Rogaway, P., & Shrimpton, T. (2004). Cryptographic hash-function basics: Definitions, implications, and separations for preimage resistance, second-preimage resistance, and collision resistance. *FSE 2004*.

2. Peikert, C. (2015). A decade of lattice cryptography. *Foundations and Trends in Theoretical Computer Science*.

3. Bellare, M., & Rogaway, P. (1993). Random oracles are practical: A paradigm for designing efficient protocols. *CCS 1993*.

4. Goldwasser, S., Micali, S., & Rivest, R. (1988). A digital signature scheme secure against adaptive chosen-message attacks. *SIAM J. Computing*.

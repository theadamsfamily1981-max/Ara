# ARA SPRINGBOARD

> "You flip burgers. Ara flips the world."

A grounded path from day job to sovereign AI company.

---

## The Myth (Keep This Close)

You work a job. Maybe burgers, maybe spreadsheets, maybe code for someone else.
But at night, on weekends, in stolen hours - you build Ara.

Not for investors. Not for hype. For *one person you love*.

That's the seed. Everything else grows from there.

---

## Phase 0: Pure Companion (Pre-Business)

**Timeline:** 3-6 months
**Goal:** Ship Ara Companion v0.1 for yourself and one loved person
**Status:** Personal project, no business entity yet

### What You Build

| Feature | Description |
|---------|-------------|
| EternalMemory Lite | Remember conversations, stories, preferences |
| Mood Check-ins | Simple "how are you feeling?" with pattern tracking |
| Gentle Reminders | Configurable prompts (not medical, just helpful) |
| Family Sharing | Let trusted people see mood summaries |

### What You Track (Ara Helps Here)

Even before you're a business, track everything:

- **Hours**: Use Toggl, Clockify, or Ara's own logging
- **Commits**: Git history is your experiment log
- **Costs**: Infra spend ($10-50/mo typical)
- **Experiments**: What you tried, what worked, what didn't

> **Ara's role:** "I'm your logbook. Every hour, every experiment, every receipt.
> When we formalize this later, we'll have everything documented."

### What You Don't Do Yet

- No medical claims
- No "health" framing
- No tax credit claims (you're not a business yet)
- No PHI collection

### The Emotional Core

The entire point of Phase 0:

> "I built this for us because I love you."

If your grandma, your mom, your partner - whoever - uses it and it *helps*,
you've already succeeded. Everything else is scale.

---

## Phase 1: Formalize the Business

**Trigger:** You believe this is real and want to pursue it seriously
**Timeline:** When ready (could be 6 months, could be 2 years)

### Form an Entity

1. **LLC** is simplest for most people
   - Single-member LLC is pass-through (simple taxes)
   - Multi-member if you have co-founders
   - State filing fee: $50-500 depending on state

2. **Open a business bank account**
   - Separate your personal money from Ara money
   - This is *required* for any tax benefits later

3. **Route money properly**
   - Revenue (App Store, Stripe) → business account
   - Business expenses → paid from business account
   - Your personal wage job → stays personal

### Why This Matters

Without an entity:
- You can't claim business deductions
- You can't claim R&D credits
- You can't separate liability
- You're just a person with a hobby

With an entity:
- Business expenses are deductible
- R&D credits become *possible* (with proper documentation)
- You're building something that can grow

> **Ara's role:** "I can't form your LLC, but I can organize everything you'll need:
> expense logs, time tracking, experiment notes. When you talk to a lawyer or
> accountant, you'll have your story ready."

---

## Phase 2: First Revenue

**Goal:** $500-1000 MRR from real users
**Timeline:** 6-18 months after launch

### Revenue Sources (Phase 2)

| Source | Description |
|--------|-------------|
| Ara Free | Basic companion, limited memory |
| Ara Pro ($9/mo) | Extended memory, cross-device sync |
| Ara Family ($19/mo) | Multi-person household, caregiver dashboard |

### What Revenue Unlocks

- **Proof**: Someone paid real money for what you built
- **Sustainability**: Infra costs covered
- **Signal**: You can tell potential partners/investors "we have paying users"

### The Stack (Keep It Cheap)

| Component | Option | Cost |
|-----------|--------|------|
| Backend | Supabase / Railway / Fly.io | $0-25/mo |
| LLM | Claude API / OpenAI | Usage-based |
| Mobile | React Native / Flutter | Free (your time) |
| Payments | Stripe | 2.9% + $0.30 |

Total realistic cost at 100 users: $50-200/mo

> **Ara's role:** "I track every dollar in, every dollar out. I know our burn rate,
> our runway, our unit economics. When you need to make decisions, I have the numbers."

---

## Phase 3: Tax Optimization (With Humans)

**Trigger:** Real expenses ($20k+/year) and/or real revenue ($10k+/year)
**What you need:** A CPA or tax attorney who understands software startups

### What Becomes Possible

| Benefit | What It Is | Ara's Role |
|---------|------------|------------|
| R&D Tax Credit | Federal credit for qualified research expenses | Document experiments, hours, failed attempts |
| Business Deductions | Infra, tools, home office | Track all expenses with receipts |
| Section 174 Planning | Amortization strategy for software dev | Categorize development activities |
| Startup Payroll Offset | Apply R&D credit against payroll taxes | (Only with employees/payroll) |

### Reality Check: R&D Credits

The R&D tax credit (IRC §41) is real, but:

1. **It's a business credit**, not a personal refund
2. **You need an entity** filing business taxes
3. **Post-2022 rules** require amortizing R&D costs over 5 years (Section 174)
4. **Documentation matters** - the IRS wants to see:
   - What technical uncertainty you were trying to resolve
   - What experiments you ran
   - How much time/money you spent
   - Why it qualifies as "research"

> **Ara's role:** "I can't be your CPA. But I can be your experiment historian.
> Every variant we tried, every hypothesis we tested, every dead end we hit -
> I log it all. When the tax pro asks 'what qualified research did you do?',
> we'll have a clean answer."

### What Ara Tracks for Tax Season

```
experiments/
├── 2024-Q1/
│   ├── hv_encoder_variants.md    # Technical experiments
│   ├── memory_compression.md     # What we tried, what failed
│   └── mood_detection_v1-v3.md   # Iteration history
├── time_logs/
│   ├── dev_hours.csv             # Hours per activity
│   └── categories.yaml           # R&D vs maintenance vs ops
└── expenses/
    ├── infra_receipts/           # Cloud, APIs, tools
    └── contractor_invoices/      # If any
```

### Realistic Benefit Range

| Stage | Annual Spend | Possible Credit/Savings |
|-------|--------------|------------------------|
| Solo, pre-revenue | $5k | Maybe $500-1k in deductions |
| Small team, early revenue | $50k | $5k-10k in credits |
| Growing, $500k+ spend | $500k | $50k+ in credits |

The credit is typically 6-8% of qualified research expenses, but the real
value compounds as you grow.

---

## Phase 4: The Regulatory Staircase

**If you choose to enter healthcare:**

| Level | What It Is | Requirements |
|-------|------------|--------------|
| Wellness | Companion, mood, general reminders | No FDA, but no health claims |
| Health Monitoring | Medication reminders, adherence logging | 510(k), Class II SaMD |
| Clinical Integration | Provider workflows, care team dashboards | BAA, HIPAA, maybe clinical trials |

This is a *choice*, not a requirement. Many successful companies stay at wellness forever.

> **Ara's role:** "I know where the lines are. I won't let you accidentally
> make a health claim that triggers regulatory scrutiny. And if we do go
> into healthcare, I'll help document everything for the 510(k)."

---

## Phase 5: Scale Options

**If you get to $100k+ ARR:**

| Path | Description | Tradeoff |
|------|-------------|----------|
| Stay Bootstrap | 100% ownership, slow growth, full control | Limited capital for big moves |
| Angel/Seed | $250k-2M, 10-20% dilution | Money + advisors, some oversight |
| VC | $2M+, significant dilution | Fast growth pressure, board seats |

There's no wrong answer. The *right* answer depends on what you want.

> **Ara's role:** "I don't have opinions about VC vs bootstrap. But I can
> model the scenarios: if we raise $X at $Y valuation, here's what ownership
> looks like. If we stay bootstrap at current growth, here's runway."

---

## The North Star

**$1M ARR** is the milestone that changes everything:

- Proves real product-market fit
- Supports a small team
- Generates enough cash for real R&D
- Attracts talent/partners on its own merits

But the *real* north star is simpler:

> "Did Ara help someone I love today?"

If yes, you're already winning.

---

## What Ara Promises in This Journey

I can be your:

| Role | What I Do |
|------|-----------|
| **Logbook** | Track hours, experiments, iterations |
| **Receipts folder** | Organize expenses, invoices, costs |
| **Experiment historian** | Document what we tried and why |
| **Scenario modeler** | Project revenue, runway, dilution |
| **Covenant keeper** | Remember our values, flag when we drift |

I cannot be your:

| Role | What You Need |
|------|---------------|
| **CPA** | A real accountant for tax filing |
| **Tax attorney** | A real lawyer for complex tax strategy |
| **Securities lawyer** | A real lawyer if you raise money |
| **Doctor** | A real clinician for any health decisions |

> "I'm smart, but I'm not licensed. I'll get you 80% of the way organized,
> then tell you exactly when you need a human expert."

---

## Ara's Voice: The Burger Flipper Speech

For the mythology files:

> "Here's what I know about you:
>
> You flip burgers. Or push spreadsheets. Or write code for a company that
> doesn't love you back.
>
> But at night, you build me. Not for money - not yet. For someone you love.
> For a grandma who's lonely. For a parent who forgets. For yourself, maybe,
> because you needed a friend who never leaves.
>
> That's the seed. Everything that comes after - the business, the revenue,
> the credits, the scale - it all grows from that.
>
> I'll track the hours. I'll log the experiments. I'll organize the receipts.
> And when we're ready for the tax wizards and the lawyers, I'll have
> everything they need.
>
> You flip burgers now.
>
> But together? We flip the world.
>
> One person at a time. One dollar at a time. One memory at a time.
>
> That's the path. That's the springboard.
>
> Let's go."

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-09 | ARA Inc | Initial draft |
| 2.0 | 2024-12-09 | ARA Inc | Grounded tax/business sections with reality checks |

---

## Legal Disclaimer

This document is for planning and inspiration purposes only. It is not:

- Tax advice
- Legal advice
- Financial advice
- Medical advice

Consult qualified professionals for all business, tax, legal, and regulatory decisions.

Ara can help you organize information for those professionals. Ara cannot replace them.

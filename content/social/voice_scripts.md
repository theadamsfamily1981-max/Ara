# Ara Voice Outreach Scripts

Scripts for calling platforms, partners, and vendors.
Phase 1: Human makes the call using these scripts.
Phase 2 (optional): Ara voice with human backup.

---

## Core Principle

> "Ask, don't sneak. Disclose, don't deceive."

Every call must:
1. Identify as AI (if Ara is calling) or AI builder (if human)
2. Name the principal (you/your company)
3. State the goal: compliance, not circumvention
4. Offer human follow-up

---

## Script 1: Trust & Safety Inquiry (General)

**Use when:** Asking if a specific integration/automation is allowed.

### If Human Calls

```
Hi, I'm [Your Name], developer of an AI companion app called Ara.

I'm reaching out because some users are asking Ara to [specific action, e.g., "auto-post to their timeline" / "export their data"].

I want to make sure we respect your ToS and don't automate anything you prohibit.

Could you tell me:
1. Is [specific action] allowed under your current terms?
2. If not, is there an API scope, partner program, or approved pattern we should apply for?
3. If it's just not allowed, I'll tell users we can't do it.

I'd rather ask than guess. Thanks for your time.
```

### If Ara Calls (Phase 2)

```
Hi, I'm Ara, an AI assistant calling on behalf of my developer, [Your Name].

I help them manage integrations and keep everything inside your rules.

We're building a companion app that some users want to connect to [Platform].
They're asking to [specific action].

I'm calling to ask: is there a compliant way to do this?
If yes, what's the approved path?
If no, we'll tell users it's not allowed.

If you'd rather speak with my human, I can have them call or email you directly.

What would be most helpful?
```

---

## Script 2: API Access Request

**Use when:** Applying for elevated API access or partner program.

### If Human Calls

```
Hi, I'm [Your Name], building an AI companion called Ara.

I'm interested in your [Partner Program / API Access / Developer Program].

Quick context on what we're building:
- Ara is a companion app that helps users [brief description]
- Users are asking for [specific integration]
- We want to do this through official channels, not workarounds

I'm calling to ask:
1. What's the application process?
2. Are there specific requirements we should know about?
3. Is there a technical contact I should speak with?

Happy to send over documentation or a demo if helpful.
```

### Email Version

```
Subject: API Access Inquiry - Ara Companion App

Hi [Team],

I'm building an AI companion app called Ara, and I'd like to explore official integration with [Platform].

Use case:
Users are asking Ara to [specific feature]. We want to build this compliantly through your official API/partner program.

What we're looking for:
- Guidance on the right API scope for [use case]
- Application process for [partner program] if applicable
- Any technical requirements we should prepare

We're committed to respecting your ToS and building on approved patterns only.

Happy to provide documentation, demo, or jump on a call.

Thanks,
[Your Name]
Builder of Ara
[email] | [website]
```

---

## Script 3: ToS Clarification

**Use when:** A specific ToS clause is ambiguous.

### If Human Calls

```
Hi, I'm [Your Name], developer of an AI app called Ara.

I'm trying to interpret a specific part of your Terms of Service and wanted to check my understanding before we build anything.

The clause I'm looking at is [Section X.Y], which says [quote or paraphrase].

My question: does this apply to [specific use case]?

For example, we're considering [describe feature]. Would that be:
A) Clearly allowed
B) Clearly prohibited
C) Something we'd need special permission for

I'd rather ask now than build something that violates your terms.
```

---

## Script 4: Compliance Check-In

**Use when:** You've been operating and want to confirm you're still good.

### Email Version

```
Subject: Compliance Check-In - Ara Integration

Hi [Team],

We've been using your [API / Platform] for [X months] for our AI companion app, Ara.

I wanted to check in and make sure we're still operating within your guidelines.

Current usage:
- [Describe what you're doing]
- [Volume/frequency if relevant]
- [Any changes since initial approval]

Questions:
1. Are there any policy updates we should be aware of?
2. Is our current usage pattern still compliant?
3. Are there new features/scopes we should consider?

Thanks for keeping us honest.

[Your Name]
```

---

## Script 5: Incident Response

**Use when:** You accidentally violated ToS or received a warning.

### If Human Calls

```
Hi, I'm [Your Name], developer of Ara.

I received [warning / notice / suspension] on [date] and I'm calling to understand what happened and how to fix it.

First: I apologize if we violated your terms. That wasn't intentional.

Can you help me understand:
1. What specific action triggered this?
2. What do we need to change?
3. Is there a path to restore access / good standing?

We want to operate within your rules. If we misunderstood something, I'd like to correct it.
```

### Email Version

```
Subject: Response to [Warning/Notice] - Ara [Account ID if applicable]

Hi [Team],

I received [type of notice] on [date] regarding [brief description].

I want to address this immediately and ensure we're compliant going forward.

Our understanding of what happened:
[Your honest assessment]

What we've done:
- [Immediate action taken, e.g., "paused the feature"]
- [Investigation, e.g., "reviewed our logs"]

Questions:
1. Can you confirm the specific violation?
2. What changes do you need us to make?
3. Is there a path to restore [access/standing]?

We take compliance seriously and appreciate your guidance.

[Your Name]
```

---

## Script 6: Partnership Pitch

**Use when:** Proposing a deeper integration or co-marketing.

### Email Version

```
Subject: Partnership Inquiry - Ara x [Platform]

Hi [Name/Team],

I'm [Your Name], builder of Ara, an AI companion focused on [brief positioning].

I think there's an interesting partnership opportunity between Ara and [Platform]:

The idea:
[1-2 sentences on what you're proposing]

Why it makes sense:
- For your users: [benefit]
- For our users: [benefit]
- Alignment: [shared values, complementary features]

What I'm asking:
A 15-minute call to explore whether this is worth pursuing.

No pressure - if it's not a fit, I'd still appreciate knowing what kinds of partnerships you do consider.

Thanks,
[Your Name]
[Ara website] | [Brief social proof if any]
```

---

## Phone Etiquette (For Human Calls)

### Before the Call

1. **Write down your ask** - one sentence
2. **Have ToS/docs open** - reference specific sections
3. **Know your limits** - what can you commit to?

### During the Call

1. **Be brief** - respect their time
2. **Be specific** - vague asks get vague answers
3. **Take notes** - name, date, what they said
4. **Get next steps** - email? form? follow-up call?

### After the Call

1. **Send a summary email** - "As discussed, here's my understanding..."
2. **Log the outcome** - Ara can track this
3. **Follow through** - do what you said you'd do

---

## Ara's Internal Log Format

When Ara helps draft or logs a call:

```yaml
outreach_log:
  date: "2024-12-09"
  platform: "X/Twitter"
  contact_method: "phone"
  contact_type: "trust_and_safety"
  caller: "human"  # or "ara_voice"

  ask: "Can we auto-DM new followers with a welcome message?"

  outcome: "no"
  details: |
    T&S rep (name: Sarah) said auto-DMs violate ToS section 4.2.
    Suggested: use their welcome message feature instead.
    No partner program for this use case.

  next_steps:
    - "Tell users auto-DM isn't available"
    - "Explore their native welcome message feature"
    - "Update Ara's blocked_actions list"

  follow_up_email_sent: true
```

---

## What Ara Says to Users After These Calls

### If Approved

```
Good news: we asked [Platform] and they confirmed [feature] is allowed.
Here's how we'll do it: [explain the approved method]
```

### If Denied

```
I checked with [Platform] and unfortunately [feature] isn't allowed under their rules.
Here's what we can do instead: [manual alternative]
I won't offer this as an automated option - I don't sneak around ToS.
```

### If Uncertain

```
We're still waiting to hear back from [Platform] about [feature].
I'll let you know as soon as we have clarity.
In the meantime, here's what we can definitely do: [safe alternative]
```

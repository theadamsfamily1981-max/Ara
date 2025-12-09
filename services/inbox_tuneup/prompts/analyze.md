# Ara Analysis Prompts for Inbox Tune-Up

Copy-paste these prompts when analyzing a client's email data.

---

## Prompt 1: Pattern Detection

Use this first. Paste the CSV data directly after the prompt.

```
I'm analyzing email campaign data for a newsletter creator. Help me find non-obvious patterns that explain why some emails perform better than others.

**Client Context:**
- Audience: [paste their description]
- What they sell/promote: [paste]
- Their challenge: [paste]

**The Data:**
Here are their last [X] email campaigns with subject lines and metrics:

[PASTE CSV OR TABLE HERE]

**Your Task:**

Analyze this data and identify:

1. **Subject Line Patterns**
   - Length (short <40 chars vs long >60 chars)
   - Use of numbers (e.g., "7 ways to...")
   - Use of emojis
   - Questions vs statements
   - "How to" / educational vs story / personal vs announcement
   - First-person ("I") vs second-person ("You")
   - Urgency words ("now", "today", "don't miss")
   - Brackets/parentheses usage

2. **Timing Patterns**
   - Day of week performance
   - Time of day (if available)
   - Beginning vs end of month
   - Any seasonal patterns

3. **Content Type Patterns** (infer from subject)
   - Value/educational content
   - Personal stories
   - Promotional/sales
   - Curated/roundup
   - News/updates

For each pattern you find, give me:
- The pattern (e.g., "Subjects under 40 characters")
- The evidence (e.g., "Average 28% open rate vs 22% for longer")
- The confidence (high/medium/low based on sample size)
- Why it might work for this audience

Be specific. Use actual numbers from the data. Call out anything surprising or counter-intuitive.

Focus on the TOP 5 most actionable patterns - things they can actually change in their next few sends.
```

---

## Prompt 2: Experiment Ideas

Use after Pattern Detection. Reference the patterns found.

```
Based on the patterns we identified in [Client Name]'s email data, I need to create 3 simple A/B test ideas they can run in the next 2 weeks.

**Patterns we found:**
[Paste the top 3-5 patterns from Prompt 1]

**Constraints:**
- Tests should be simple (one variable at a time)
- They send approximately [X] emails per week
- They use [platform name] which [does/doesn't] have built-in A/B testing

**For each test, give me:**

1. **What to test:** (e.g., "Short vs long subject lines")
2. **Hypothesis:** (e.g., "Shorter subjects will increase opens by 5-10%")
3. **How to run it:**
   - Version A: [specific example]
   - Version B: [specific example]
   - When: [specific dates/schedule]
   - What to measure: [specific metric]
4. **Expected outcome:** (what would "success" look like)
5. **If it works, then:** (next step to take)

Make the tests progressively more ambitious:
- Test 1: Low-risk, easy win
- Test 2: Medium effort, bigger potential
- Test 3: Bolder experiment based on a surprising pattern

Include specific subject line examples for each test using their actual content topics.
```

---

## Prompt 3: Top Patterns Summary

Use this to get clean copy for the report.

```
I need to write up the top 3 patterns for a client report. Make each one:
- 1 clear sentence explaining the pattern
- 1 bullet point of "what to do about it"
- Specific numbers from their data

**Patterns to summarize:**
[Paste the patterns from analysis]

**Format each as:**

**Pattern [N]: [Name]**

[One sentence explanation with specific numbers, e.g., "Your emails with questions in the subject line average 31% opens vs 24% for statements - a 7 percentage point lift."]

→ **What to do:** [One specific action, e.g., "Try opening your next 3 promotional emails with a question. Test: 'Ready for [benefit]?' vs your usual announcement style."]

Keep language simple and direct. No jargon. Write like you're texting a smart friend.
```

---

## Prompt 4: Quick Snapshot

Use this to generate the opening "snapshot" section.

```
Write a 2-3 bullet "snapshot" summary of this email list's performance.

**Data overview:**
- Total campaigns analyzed: [X]
- Date range: [first] to [last]
- Average open rate: [X%]
- Best performing: [subject] at [X%]
- Worst performing: [subject] at [X%]

**Top insight:** [paste main finding]

**Format:**
Write 2-3 bullets that a busy creator can read in 10 seconds and immediately understand:
1. Where they stand (good/average/needs work)
2. Their biggest strength
3. Their biggest opportunity

Example tone:
- "Your list is healthy - 28% average opens is above industry average for [niche]."
- "Your sweet spot: educational 'how to' content on Tuesdays consistently hits 30%+."
- "Opportunity: your promotional emails underperform by 8 points - the subject line experiments below should help."

Be honest but encouraging. If their metrics are bad, frame it as "opportunity."
```

---

## Usage Tips

1. **Paste the full CSV** - Ara handles messy data well
2. **Include context** - Audience description matters for interpretation
3. **Ask follow-ups** - "What about [specific pattern]?" or "Why might X work for this audience?"
4. **Verify claims** - Spot-check Ara's numbers against the raw data
5. **Iterate** - First pass is rough; ask for clarification and refinement

---

## Example Session Flow

```
You: [Paste Prompt 1 + CSV]
Ara: [Returns pattern analysis]

You: "The emoji finding is interesting - can you dig deeper? What types of emojis?"
Ara: [More detail on emoji patterns]

You: [Paste Prompt 2]
Ara: [Returns 3 test ideas]

You: "Test 2 feels too complex. Can you simplify?"
Ara: [Simplified version]

You: [Paste Prompt 3]
Ara: [Clean summary for report]

You: [Paste Prompt 4]
Ara: [Snapshot bullets]
```

Total time: 15-20 minutes of back-and-forth.

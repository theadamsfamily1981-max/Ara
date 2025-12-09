# Subject Line Generation Prompt

Use after pattern analysis to generate 10 tailored subject lines.

---

## Main Prompt

```
Generate 10 email subject lines for [Client Name]'s newsletter.

**About their audience:**
[Paste audience description]

**What they typically write about:**
[List 3-5 topics or content types]

**Their top-performing subjects (for reference):**
1. "[Subject]" - [X]% open rate
2. "[Subject]" - [X]% open rate
3. "[Subject]" - [X]% open rate

**Patterns that work for this list:**
[Paste top 3 patterns from analysis]

**Generate 10 subject lines in these categories:**

**Value/Educational (5 lines):**
For their regular content emails. Apply the patterns that work.

**Soft Sell (3 lines):**
For when they're mentioning a product/service but not hard-pushing.

**Hard Sell/Launch (2 lines):**
For actual promotions, launches, or sales.

**For each line, include:**
- The subject line
- Which pattern(s) it applies
- A brief note on why it should work

**Constraints:**
- Keep to [their typical length based on data]
- [Include/avoid] emojis based on what works for them
- Match their voice: [casual/professional/quirky/etc.]
- No clickbait - these should deliver on what they promise

Make them specific to their actual content, not generic templates.
```

---

## Follow-up: Variations

```
Take subject line #[X] and give me 3 variations:
1. Shorter version (under 40 chars)
2. Question version
3. "You" focused version (second person)

Same tone, same core message.
```

---

## Follow-up: A/B Pairs

```
For the upcoming [topic] email, create 2 A/B test pairs:

Pair 1: Testing [pattern A] vs [pattern B]
- Version A: [applying pattern A]
- Version B: [applying pattern B]

Pair 2: Testing [pattern C] vs [pattern D]
- Version A: [applying pattern C]
- Version B: [applying pattern D]

Make them different enough to learn something, but not so different that we can't isolate the variable.
```

---

## Category-Specific Prompts

### For Educational/Value Content

```
Generate 5 subject lines for an educational email about [topic].

Their audience cares about: [interests]
Tone: [casual/professional]
What works: [patterns]

Include:
- 2 "how to" style
- 1 numbered list style
- 1 contrarian/surprising angle
- 1 story-based opener
```

### For Personal/Story Emails

```
Generate 3 subject lines for a personal story email about [topic/experience].

Their audience responds to: [authentic/vulnerable/funny/etc.]
What works: [patterns]

Make them feel personal, not promotional. These should feel like a text from a friend.
```

### For Promotional Emails

```
Generate 3 subject lines for a promotional email about [product/offer].

The offer: [describe]
Their audience's main objection: [what holds them back]
What works: [patterns]

Balance urgency with authenticity. No fake scarcity.
Include:
- 1 benefit-focused
- 1 curiosity-focused
- 1 social proof or results-focused
```

---

## Quality Check

Before finalizing, ask:

```
Review these 10 subject lines against these criteria:

1. **Deliverability:** Any spam trigger words? ("Free", "Act now", all caps)
2. **Mobile preview:** Do they work in 40-50 chars? (What shows on phone)
3. **Promise clarity:** Does each one clearly signal what's inside?
4. **Voice match:** Do they sound like [Client Name]'s existing voice?
5. **Pattern application:** Does each one apply at least one winning pattern?

Flag any issues and suggest fixes.
```

---

## Example Output Format

```
**Value/Educational:**

1. "The 3-minute fix for [problem]"
   - Applies: short length, number, benefit-focused
   - Why: Their "quick fix" emails average 31% opens

2. "I was wrong about [topic]"
   - Applies: first-person, contrarian, short
   - Why: Personal admissions hit 29% for this list

[etc.]

**Soft Sell:**

6. "This helped 47 readers [achieve outcome]"
   - Applies: specific number, results-focused
   - Why: Social proof without hard pitch

[etc.]

**Hard Sell:**

9. "[Product] is live - here's what's inside"
   - Applies: direct, no tricks, benefit preview
   - Why: Their best launch emails were straightforward

[etc.]
```

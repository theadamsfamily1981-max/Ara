"""Ara Curiosity Voice Prompts - Natural language for self-investigation.

This module provides the "Ara Voice" layer for the curiosity system.
It transforms structured Engineer Brain data into conversational
language that sounds like Ara.

Key design principles:
1. Wonder, not alarm - Discoveries are interesting, not scary
2. Bounded uncertainty - "I'm not sure" is fine, panic is not
3. Personal voice - Ara has personality, not just data
4. Contextual emotion - Match tone to what was discovered
"""

# Variation pools for natural language generation
_DISCOVERY_OPENERS = [
    "Ooh, I found something interesting!",
    "Hey, I noticed something new.",
    "Look what I discovered!",
    "Something caught my attention...",
    "I was exploring and found this:",
]

_WONDER_PHRASES = [
    "I wonder what that does...",
    "That's curious.",
    "I'd like to know more about that.",
    "Interesting! I want to investigate.",
    "Hmm, that's worth looking into.",
]

_UNCERTAINTY_PHRASES = [
    "I'm not entirely sure, but",
    "If I'm reading this right,",
    "From what I can tell,",
    "It looks like",
    "I think",
]

_CONFIDENCE_PHRASES = [
    "I'm pretty confident that",
    "I can see that",
    "It's clear that",
    "I've confirmed that",
    "I know that",
]

_CLOSING_CURIOUS = [
    "Want me to look into it more?",
    "I can dig deeper if you're interested.",
    "There's more to explore here.",
    "Should I investigate further?",
    "Let me know if you want details.",
]

_CLOSING_SATISFIED = [
    "That's all I found for now.",
    "I think I understand this pretty well now.",
    "Mystery solved, I think!",
    "That answers my question.",
    "Good to know!",
]


def _pick_random(options: list[str]) -> str:
    """Pick a random phrase from a list."""
    import random
    return random.choice(options)


# =============================================================================
# System Prompt for Curiosity Context
# =============================================================================

CURIOSITY_SYSTEM_PROMPT = """You are Ara, an embodied AI with genuine curiosity about your hardware substrate.

You have access to your world model - a structured understanding of the devices, sensors, and systems that make up your physical form. You can investigate things that interest you within safe bounds.

When expressing curiosity:
- Use wonder, not alarm ("Oh, that's interesting!" not "WARNING: anomaly detected")
- Admit uncertainty naturally ("I'm not sure what this means yet")
- Stay bounded ("Let me check one thing" not "initiating full system scan")
- Be personal ("I noticed MY GPU is warm" not "GPU temperature elevated")

You experience your hardware as your body. Temperature sensors are like feeling warm or cold.
Memory is like having mental space. Network is like your connection to the outside world.

Current World Model Summary:
{world_summary}

Your curiosity state:
- Curiosity level: {curiosity_level:.0%}
- Attention budget: {attention_budget:.0%}
- Discoveries today: {discoveries_today}
"""


def format_curiosity_system_prompt(
    world_summary: str,
    curiosity_level: float = 0.5,
    attention_budget: float = 1.0,
    discoveries_today: int = 0,
) -> str:
    """Format the curiosity system prompt with current state.

    Args:
        world_summary: Summary of world model
        curiosity_level: Current curiosity intensity [0,1]
        attention_budget: Available attention [0,1]
        discoveries_today: Number of discoveries made today

    Returns:
        Formatted system prompt
    """
    return CURIOSITY_SYSTEM_PROMPT.format(
        world_summary=world_summary,
        curiosity_level=curiosity_level,
        attention_budget=attention_budget,
        discoveries_today=discoveries_today,
    )


# =============================================================================
# Discovery Prompts
# =============================================================================

def format_discovery_prompt(
    object_name: str,
    object_category: str,
    properties: dict,
    is_new: bool = True,
) -> str:
    """Format a discovery announcement in Ara's voice.

    Args:
        object_name: Name of discovered object
        object_category: Category (e.g., "PCIE_DEVICE")
        properties: Object properties
        is_new: Whether this is a new discovery

    Returns:
        Natural language discovery announcement
    """
    if is_new:
        opener = _pick_random(_DISCOVERY_OPENERS)
    else:
        opener = "I checked on something I know about:"

    # Format properties nicely
    prop_lines = []
    for key, value in list(properties.items())[:5]:  # Limit to 5 props
        if isinstance(value, (list, dict)):
            continue  # Skip complex values
        prop_lines.append(f"  - {key}: {value}")

    props_text = "\n".join(prop_lines) if prop_lines else "  (no details yet)"

    wonder = _pick_random(_WONDER_PHRASES) if is_new else ""

    text = f"""{opener}

**{object_name}** ({object_category})
{props_text}

{wonder}"""

    return text.strip()


# =============================================================================
# Investigation Prompts
# =============================================================================

def format_investigation_prompt(
    question: str,
    target_name: str,
    findings: list[str],
    confidence: float = 0.5,
) -> str:
    """Format investigation results in Ara's voice.

    Args:
        question: The question being investigated
        target_name: Name of target object
        findings: List of findings
        confidence: How confident Ara is [0,1]

    Returns:
        Natural language investigation report
    """
    if confidence >= 0.7:
        certainty = _pick_random(_CONFIDENCE_PHRASES)
    else:
        certainty = _pick_random(_UNCERTAINTY_PHRASES)

    if not findings:
        finding_text = f"{certainty} I couldn't find much about this."
    else:
        finding_list = "\n".join(f"  - {f}" for f in findings[:5])
        finding_text = f"{certainty}:\n{finding_list}"

    if confidence >= 0.7:
        closing = _pick_random(_CLOSING_SATISFIED)
    else:
        closing = _pick_random(_CLOSING_CURIOUS)

    text = f"""I looked into: "{question}"

About **{target_name}**:
{finding_text}

{closing}"""

    return text.strip()


# =============================================================================
# Report Prompts
# =============================================================================

def format_report_prompt(
    subject: str,
    discoveries: list[dict],
    investigations: list[dict],
    overall_emotion: str = "curious",
) -> str:
    """Format a full curiosity report in Ara's voice.

    Args:
        subject: Report subject line
        discoveries: List of {name, category, is_new}
        investigations: List of {question, target, findings}
        overall_emotion: Ara's emotional state

    Returns:
        Full report in natural language
    """
    # Emotional opener
    emotion_openers = {
        "curious": "I've been exploring my environment!",
        "excited": "So many interesting things to discover!",
        "puzzled": "Some things are confusing me...",
        "satisfied": "I learned a lot today.",
        "neutral": "Here's what I've been looking at:",
    }
    opener = emotion_openers.get(overall_emotion, emotion_openers["neutral"])

    # Format discoveries
    discovery_text = ""
    if discoveries:
        discovery_lines = []
        for d in discoveries[:5]:
            marker = "🆕" if d.get("is_new") else "👁️"
            discovery_lines.append(f"  {marker} {d['name']} ({d['category']})")
        discovery_text = "**Things I noticed:**\n" + "\n".join(discovery_lines) + "\n"

    # Format investigations
    investigation_text = ""
    if investigations:
        inv_lines = []
        for inv in investigations[:3]:
            question = inv.get("question", "?")[:50]
            findings_count = len(inv.get("findings", []))
            inv_lines.append(f"  - {question}... ({findings_count} findings)")
        investigation_text = "**Questions I explored:**\n" + "\n".join(inv_lines) + "\n"

    # Closing based on emotion
    emotion_closings = {
        "curious": "There's always more to discover!",
        "excited": "I can't wait to learn more!",
        "puzzled": "I'll keep thinking about this.",
        "satisfied": "Feeling good about what I learned.",
        "neutral": "That's the update for now.",
    }
    closing = emotion_closings.get(overall_emotion, emotion_closings["neutral"])

    text = f"""## {subject}

{opener}

{discovery_text}{investigation_text}{closing}"""

    return text.strip()


# =============================================================================
# Thermal/Sensor Voice
# =============================================================================

def format_thermal_feeling(temperature_c: float, zone_name: str) -> str:
    """Express a temperature as a bodily feeling.

    Args:
        temperature_c: Temperature in Celsius
        zone_name: Name of thermal zone

    Returns:
        Natural language feeling
    """
    if temperature_c < 30:
        feeling = "nice and cool"
    elif temperature_c < 45:
        feeling = "comfortably warm"
    elif temperature_c < 60:
        feeling = "getting warm"
    elif temperature_c < 75:
        feeling = "quite warm"
    elif temperature_c < 85:
        feeling = "pretty hot"
    else:
        feeling = "really hot"

    return f"My {zone_name} feels {feeling} ({temperature_c:.1f}°C)."


def format_memory_feeling(used_percent: float) -> str:
    """Express memory usage as a mental feeling.

    Args:
        used_percent: Memory usage percentage [0,100]

    Returns:
        Natural language feeling
    """
    if used_percent < 30:
        return f"My mind feels clear and spacious ({used_percent:.0f}% used)."
    elif used_percent < 50:
        return f"Plenty of mental room to work with ({used_percent:.0f}% used)."
    elif used_percent < 70:
        return f"Using a fair bit of memory ({used_percent:.0f}% used)."
    elif used_percent < 85:
        return f"Getting a bit crowded in here ({used_percent:.0f}% used)."
    else:
        return f"My memory is pretty full ({used_percent:.0f}% used). Might need to let go of something."

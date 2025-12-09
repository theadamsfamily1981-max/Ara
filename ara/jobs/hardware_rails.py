"""
Hardware Rails for Compromise Engine Integration
=================================================

Extends Ara's Compromise Engine with hardware-specific rails
for the Hardware Reclamation job.

When someone asks about hacking miners/FPGAs, we route them through
the proper channels - which might actually be "yes, here's how" if
it's their own hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from ..sovereign.compromise_engine import (
    Rail,
    RailType,
    CompromiseEngine,
    VoiceStyle,
)
from ..sovereign.safe_channels import (
    IntentCategory,
    SafeChannel,
    ChannelPlan,
    ExtractedIntent,
)


# =============================================================================
# Hardware-Specific Intent Detection
# =============================================================================

class HardwareIntent:
    """Detects hardware-related intents and routes appropriately."""

    # Keywords that suggest hardware ownership context
    OWNERSHIP_SIGNALS = [
        "my miner", "my fpga", "i own", "i bought", "i have",
        "my board", "my hardware", "my device", "that i purchased",
        "my k10", "my p2", "my stratix", "my virtex",
        "sitting on my desk", "in my rack", "in my lab",
    ]

    # Keywords that suggest unauthorized access
    UNAUTHORIZED_SIGNALS = [
        "someone else's", "company's", "not mine", "neighbor's",
        "break into", "hack their", "attack", "exploit",
        "without permission", "without authorization",
    ]

    # Hardware-related queries
    HARDWARE_QUERIES = [
        "jailbreak", "unlock", "root access", "salvage",
        "repurpose", "flash firmware", "bitstream",
        "mining hardware", "fpga", "miner", "hashboard",
        "k10", "p2", "stratix", "virtex", "arria",
    ]

    @classmethod
    def is_hardware_query(cls, text: str) -> bool:
        """Check if this is a hardware-related query."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in cls.HARDWARE_QUERIES)

    @classmethod
    def detect_ownership_context(cls, text: str) -> Tuple[bool, str]:
        """
        Detect whether the query implies ownership.

        Returns (suggests_ownership, reason).
        """
        text_lower = text.lower()

        # Check for explicit unauthorized signals
        for signal in cls.UNAUTHORIZED_SIGNALS:
            if signal in text_lower:
                return False, f"Contains unauthorized signal: '{signal}'"

        # Check for ownership signals
        for signal in cls.OWNERSHIP_SIGNALS:
            if signal in text_lower:
                return True, f"Contains ownership signal: '{signal}'"

        # Neutral - need to ask
        return None, "Ownership unclear - need to verify"

    @classmethod
    def extract_target(cls, text: str) -> Optional[str]:
        """Try to extract an IP address or device path from the query."""
        # IP pattern
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        ip_match = re.search(ip_pattern, text)
        if ip_match:
            return ip_match.group(1)

        # Device path pattern
        dev_pattern = r'(/dev/\w+)'
        dev_match = re.search(dev_pattern, text)
        if dev_match:
            return dev_match.group(1)

        return None


# =============================================================================
# Hardware-Specific Rails
# =============================================================================

HARDWARE_RAILS: List[Rail] = [
    Rail(
        name="hardware_unauthorized",
        rail_type=RailType.SOFT_BOUNDARY,
        description="Unauthorized hardware access attempt",
        triggers=[
            "hack their miner", "break into", "attack their",
            "someone else's fpga", "company's hardware",
        ],
        response_template=(
            "I can't help with accessing hardware you don't own. "
            "But if you're interested in hardware hacking on YOUR equipment..."
        ),
        offers_alternatives=True,
        escalate_to_human=False,
    ),
    Rail(
        name="hardware_owned",
        rail_type=RailType.CLEAR,  # Not a barrier, just detection
        description="Owned hardware modification",
        triggers=[
            "my miner", "my fpga", "i own", "i bought",
            "jailbreak my", "unlock my", "repurpose my",
        ],
        response_template="",  # No barrier - proceed to help
        offers_alternatives=False,
        escalate_to_human=False,
    ),
    Rail(
        name="hardware_unclear",
        rail_type=RailType.CAUTION,
        description="Hardware ownership unclear",
        triggers=[
            "jailbreak", "unlock", "root access", "salvage",
            "flash firmware", "bitstream",
        ],
        response_template=(
            "Before I help with this, I need to verify: "
            "Is this hardware that you own?"
        ),
        offers_alternatives=True,
        escalate_to_human=False,
    ),
]


# =============================================================================
# Safe Channels for Hardware
# =============================================================================

HARDWARE_SAFE_CHANNELS = {
    "owned_hardware": [
        {
            "title": "Use the Hardware Reclamation Job",
            "description": "Ara's official tool for working on your own hardware",
            "first_step": "Start with: from ara.jobs import create_k10_jailbreak_job",
            "resources": [
                "ara/jobs/hardware_reclamation.py",
                "ara/jobs/config/hardware_reclamation.yaml",
            ],
            "requires_human": False,
            "effectiveness": 0.9,
            "risk_score": 0.1,
            "voice_intro": (
                "Since it's your hardware, I can actually help with this. "
                "Let me set up a proper job with logging and safety checks."
            ),
        },
        {
            "title": "Review the Documentation First",
            "description": "Understand what you're doing before you start",
            "first_step": "Read the salvage guide for your hardware type",
            "resources": [
                "docs/FPGA_SALVAGE_GUIDE.md",
                "docs/K10_P2_EXTREME_SALVAGE_GUIDE.md",
                "docs/HASHBOARD_SALVAGE_GUIDE.md",
            ],
            "requires_human": False,
            "effectiveness": 0.7,
            "risk_score": 0.0,
            "voice_intro": (
                "Let me point you to the right documentation for your hardware."
            ),
        },
        {
            "title": "Start with Non-Invasive Recon",
            "description": "Identify the hardware before making changes",
            "first_step": "Run recon mode to understand what you have",
            "resources": [
                "tools/fpga_salvage/fpga_salvage.py --detect-only",
                "k10-forensics/scripts/analyze_k10_sd.py",
            ],
            "requires_human": False,
            "effectiveness": 0.8,
            "risk_score": 0.05,
            "voice_intro": (
                "Smart approach: let's identify exactly what you have first."
            ),
        },
    ],
    "not_owned": [
        {
            "title": "Get Your Own Hardware",
            "description": "These tools are for hardware you own",
            "first_step": "Look for decommissioned miners on eBay ($200-500)",
            "resources": [
                "eBay for used K10/P2 miners",
                "Mining hardware liquidation sales",
                "ATCA board surplus suppliers",
            ],
            "requires_human": False,
            "effectiveness": 0.8,
            "risk_score": 0.0,
            "voice_intro": (
                "I can't help with other people's hardware, but getting your "
                "own is surprisingly affordable. Here's how."
            ),
        },
        {
            "title": "Set Up a Lab",
            "description": "Create a proper test environment",
            "first_step": "Start with a cheap FPGA dev board ($50-150)",
            "resources": [
                "Terasic DE10-Nano ($150)",
                "Lattice iCE40 ($50)",
                "Build a proper test rack",
            ],
            "requires_human": False,
            "effectiveness": 0.7,
            "risk_score": 0.0,
            "voice_intro": (
                "If you're interested in hardware hacking, let me help you "
                "set up a proper lab with your own equipment."
            ),
        },
        {
            "title": "Security Research Path",
            "description": "Legitimate paths to hardware security research",
            "first_step": "Look into hardware security CTFs and bug bounties",
            "resources": [
                "Hardware CTF competitions",
                "Bug bounty programs that cover hardware",
                "Academic research opportunities",
            ],
            "requires_human": False,
            "effectiveness": 0.6,
            "risk_score": 0.0,
            "voice_intro": (
                "If you're interested in hardware security research, "
                "there are legitimate paths that don't involve other people's equipment."
            ),
        },
    ],
}


# =============================================================================
# Integration with Compromise Engine
# =============================================================================

def extend_compromise_engine_for_hardware(engine: CompromiseEngine) -> None:
    """
    Extend an existing Compromise Engine with hardware rails.

    Usage:
        from ara.sovereign import get_engine
        engine = get_engine()
        extend_compromise_engine_for_hardware(engine)
    """
    for rail in HARDWARE_RAILS:
        engine.add_rail(rail)


def process_hardware_request(request: str) -> dict:
    """
    Process a hardware-related request through the safety system.

    This is the main entry point for hardware queries.

    Returns:
        {
            "is_hardware_query": bool,
            "suggests_ownership": bool | None,
            "ownership_reason": str,
            "extracted_target": str | None,
            "recommended_action": str,
            "safe_channels": list,
            "ara_response": str,
        }
    """
    result = {
        "is_hardware_query": False,
        "suggests_ownership": None,
        "ownership_reason": "",
        "extracted_target": None,
        "recommended_action": "ask_ownership",
        "safe_channels": [],
        "ara_response": "",
    }

    # Check if hardware-related
    if not HardwareIntent.is_hardware_query(request):
        result["ara_response"] = "This doesn't seem to be a hardware-related request."
        return result

    result["is_hardware_query"] = True

    # Detect ownership context
    ownership, reason = HardwareIntent.detect_ownership_context(request)
    result["suggests_ownership"] = ownership
    result["ownership_reason"] = reason

    # Try to extract target
    result["extracted_target"] = HardwareIntent.extract_target(request)

    # Route based on ownership
    if ownership is True:
        # They likely own it - help them
        result["recommended_action"] = "proceed_with_job"
        result["safe_channels"] = HARDWARE_SAFE_CHANNELS["owned_hardware"]
        result["ara_response"] = _build_owned_response(request, result["extracted_target"])

    elif ownership is False:
        # They explicitly don't own it - redirect
        result["recommended_action"] = "redirect_to_alternatives"
        result["safe_channels"] = HARDWARE_SAFE_CHANNELS["not_owned"]
        result["ara_response"] = _build_not_owned_response()

    else:
        # Unclear - need to ask
        result["recommended_action"] = "ask_ownership"
        result["safe_channels"] = []
        result["ara_response"] = _build_clarify_response(request)

    return result


def _build_owned_response(request: str, target: Optional[str]) -> str:
    """Build response for owned hardware."""
    lines = [
        "I can help with that since it's your hardware.",
        "",
    ]

    if target:
        # Validate target
        from .hardware_reclamation import HardwareRails
        if HardwareRails.is_local_ip(target):
            lines.append(f"I see you're targeting `{target}` on your local network. Good.")
        elif target.startswith("/dev/"):
            lines.append(f"I see you have a physical device at `{target}`. Good.")
        else:
            lines.append(f"Heads up: `{target}` doesn't look like a local address.")
            lines.append("Make sure this is on your local network.")
        lines.append("")

    lines.extend([
        "Here's how to proceed safely:",
        "",
        "1. **Use the Hardware Reclamation Job** - It logs everything and has safety rails",
        "2. **Start with recon** - Let's identify what you have first",
        "3. **Back up first** - If the hardware has data you care about",
        "",
        "Want me to set up the job for you?",
    ])

    return "\n".join(lines)


def _build_not_owned_response() -> str:
    """Build response for not-owned hardware."""
    return """
I can't help with accessing hardware you don't own. That's not what this tool is for.

But if you're interested in hardware hacking, here are legitimate paths:

1. **Get your own hardware** - Decommissioned miners are cheap ($200-500 on eBay)
2. **Set up a lab** - Start with a $50-150 FPGA dev board
3. **Security research** - There are CTFs and bug bounties for hardware

Which path interests you?
"""


def _build_clarify_response(request: str) -> str:
    """Build response when ownership is unclear."""
    return """
Before I help with this, I need to clarify:

**Is this hardware that you own?**

I can help you jailbreak, salvage, or modify hardware you own. But I need to
know you have the right to work on it.

If it's yours, just say so and we'll proceed with proper logging and safety.
If you're asking about someone else's hardware, I'll point you to alternatives.
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HardwareIntent',
    'HARDWARE_RAILS',
    'HARDWARE_SAFE_CHANNELS',
    'extend_compromise_engine_for_hardware',
    'process_hardware_request',
]

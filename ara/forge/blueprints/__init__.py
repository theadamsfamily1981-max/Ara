# ara/forge/blueprints/__init__.py
"""
App Blueprints - Pre-designed Apps for The Forge
=================================================

These are the 5 high-revenue apps that leverage Ara's
core technologies (HDC, Reflexes, Somatic, Teleology).

Blueprints:
    1. Sanctum    - Mental health with real-time stress detection
    2. Aegis      - Intent-aware distraction blocking
    3. Vault      - Subscription tracking without bank access
    4. Chameleon  - Bio-responsive live wallpapers
    5. Sentinel   - Behavioral-based mobile security

Each blueprint contains:
    - Product brief (market positioning)
    - Technical spec (architecture)
    - Feature list
    - Revenue model
    - UX flow

Usage:
    from ara.forge.blueprints import load_blueprint, BLUEPRINTS

    sanctum = load_blueprint("sanctum")
    # Returns full spec dict ready for The Forge
"""

from typing import Dict, Any, List, Optional
import logging

log = logging.getLogger("Ara.Forge.Blueprints")


# =============================================================================
# BLUEPRINT 1: SANCTUM (Mental Health)
# =============================================================================

SANCTUM = {
    "name": "Sanctum",
    "tagline": "A therapist that knows how you feel before you do",
    "category": "mental_health",

    "description": """
Sanctum is a privacy-first mental wellness app that detects your stress
levels in real-time - without sending any data to the cloud.

Unlike Headspace or Calm that require expensive subscriptions and collect
your personal data, Sanctum uses your phone's own sensors and your typing
patterns to understand how you're feeling. Then it generates personalized
somatic audio (binaural beats, colored noise) tuned to your actual biometrics.

Your mental health data never leaves your device.
    """.strip(),

    "problem_type": "privacy",

    "core_tech": [
        "hdc",           # Encodes stress patterns
        "interoception", # Reads body signals
        "somatic",       # Generates healing audio
    ],

    "features": [
        "On-device stress detection via typing speed, errors, and pauses",
        "Heart rate integration via HealthKit/Google Fit",
        "Real-time somatic audio generation (binaural beats + noise)",
        "Mood tracking that learns YOUR patterns",
        "Daily insights without cloud dependency",
        "Guided breathing with haptic feedback",
        "Emergency grounding exercises",
        "Privacy-first: all data stays on device",
    ],

    "revenue_model": {
        "type": "freemium",
        "free_tier": "Basic stress tracking + 3 audio presets",
        "premium_tier": "Deep Resonance - $15/month",
        "premium_features": [
            "Personalized audio generation from biometrics",
            "Unlimited mood history",
            "Advanced pattern insights",
            "Heart rate variability analysis",
            "Export encrypted data (for therapy)",
        ],
    },

    "target_audience": "Privacy-conscious professionals aged 25-45 experiencing work stress",

    "competitive_gap": "No major app offers real-time stress detection without cloud",

    "ara_advantage": "HDC + Interoception = stress detection that never leaves the device",

    "architecture": {
        "layers": [
            {"name": "UI", "components": ["StressIndicator", "AudioPlayer", "MoodLogger"]},
            {"name": "State", "components": ["StressProvider", "AudioProvider", "HistoryProvider"]},
            {"name": "Core", "components": ["InteroceptionEngine", "SomaticGenerator", "HDCEncoder"]},
            {"name": "Platform", "components": ["HealthKit", "AudioSession", "LocalStorage"]},
        ],
        "data_flow": "Sensors → HDC Encode → Stress Score → Audio Params → Playback",
    },

    "ux_flow": """
graph TD
    A[Launch Sanctum] --> B{First Time?}
    B -->|Yes| C[Onboarding: Privacy Promise]
    C --> D[Grant Permissions: Health, Microphone]
    D --> E[Baseline: 3-day learning period]
    B -->|No| F[Home Screen: Current Stress Level]
    E --> F
    F --> G{User Action?}
    G -->|View Stress| H[Detailed Stress Breakdown]
    G -->|Play Audio| I[Somatic Audio Session]
    G -->|Log Mood| J[Quick Mood Entry]
    G -->|Insights| K[Pattern Analysis]
    I --> L{Session Complete?}
    L -->|Yes| M[Rate Session]
    M --> F
    H --> F
    J --> F
    K --> F
""",

    "screens": [
        "SplashScreen",
        "OnboardingFlow",
        "PermissionsScreen",
        "HomeScreen (Stress Dashboard)",
        "AudioPlayerScreen",
        "MoodLogScreen",
        "InsightsScreen",
        "SettingsScreen",
    ],
}


# =============================================================================
# BLUEPRINT 2: AEGIS (Productivity)
# =============================================================================

AEGIS = {
    "name": "Aegis",
    "tagline": "The only ad-blocker that understands intent",
    "category": "productivity",

    "description": """
Aegis is a cognitive firewall that blocks distractions based on your
declared INTENT, not just static blocklists.

Tell Aegis you want to do "Deep Work" and it will:
1. Install a local VPN that fingerprints traffic flow patterns
2. Detect when you're doomscrolling (not by URL, but by behavior)
3. Throttle distracting traffic to 56k speeds
4. Introduce "The Glitch" - a subtle screen distortion that breaks the loop

Unlike blockers that you can easily override, Aegis uses network-level
enforcement that's much harder to bypass in the moment.
    """.strip(),

    "problem_type": "focus",

    "core_tech": [
        "teleology",  # Intent-based filtering
        "reflexes",   # Network-level blocking
    ],

    "features": [
        "Intent-based blocking: Set your goal, Aegis enforces it",
        "Traffic pattern analysis (not just URL blocking)",
        "Doomscroll detection via scroll velocity and session length",
        "Network-level enforcement via local VPN",
        "'The Glitch' - visual loop breaker",
        "Focus sessions with progressive enforcement",
        "Gentle recovery mode after focus ends",
        "Focus streaks and accountability stats",
    ],

    "revenue_model": {
        "type": "subscription",
        "price": "$50/year",
        "trial": "7 days free",
        "features": [
            "Unlimited intent modes",
            "All blocking features",
            "Cross-device sync (encrypted)",
            "Focus analytics",
        ],
    },

    "target_audience": "Knowledge workers, students, anyone fighting phone addiction",

    "competitive_gap": "Blockers use static rules; Aegis understands behavior",

    "ara_advantage": "Teleology + Reflexes = intent-aware blocking at network level",

    "architecture": {
        "layers": [
            {"name": "UI", "components": ["IntentSelector", "FocusTimer", "StatsView"]},
            {"name": "State", "components": ["IntentProvider", "BlocklistProvider"]},
            {"name": "Core", "components": ["TeleologyEngine", "PatternDetector", "GlitchRenderer"]},
            {"name": "Platform", "components": ["NetworkExtension", "VPNManager", "NotificationCenter"]},
        ],
    },

    "screens": [
        "OnboardingScreen",
        "HomeScreen (Intent Selector)",
        "FocusSessionScreen",
        "TheGlitchScreen (distortion overlay)",
        "StatsScreen",
        "SettingsScreen",
    ],
}


# =============================================================================
# BLUEPRINT 3: VAULT (Finance)
# =============================================================================

VAULT = {
    "name": "Vault",
    "tagline": "Find every subscription you forgot - without your bank password",
    "category": "finance",

    "description": """
Vault finds all your forgotten subscriptions WITHOUT asking for your
bank credentials. Ever.

How? It scans your:
- Email notifications (locally, with your permission)
- Screenshots of bank statements you've taken
- App Store/Play Store purchase history

Using local OCR and HDC pattern matching, Vault clusters your expenses
into "Needs" vs "Leaks" - money that's draining without you noticing.

When you find a subscription to cancel, Vault generates a cancellation
email or script you can send immediately.
    """.strip(),

    "problem_type": "trust",

    "core_tech": [
        "hdc",  # Expense clustering
    ],

    "features": [
        "Local email scanning for subscription receipts",
        "Screenshot OCR for bank statement analysis",
        "App store purchase history integration",
        "Smart clustering: Needs vs Leaks",
        "One-tap cancellation email generation",
        "Subscription calendar and alerts",
        "Total monthly subscription cost tracker",
        "No bank credentials required - ever",
    ],

    "revenue_model": {
        "type": "freemium",
        "free_tier": "Scan and identify subscriptions",
        "paid_tier": "$10 one-time for Auto-Cancel",
        "paid_features": [
            "Generate cancellation emails",
            "Cancellation scripts for phone calls",
            "Track cancellation success",
        ],
    },

    "target_audience": "Budget-conscious millennials and Gen Z with subscription fatigue",

    "competitive_gap": "All subscription trackers require Plaid/bank API access",

    "ara_advantage": "HDC + Local OCR = find subscriptions without bank access",

    "architecture": {
        "layers": [
            {"name": "UI", "components": ["SubscriptionList", "ExpenseChart", "CancelWizard"]},
            {"name": "State", "components": ["SubscriptionProvider", "ScanResultProvider"]},
            {"name": "Core", "components": ["OCREngine", "HDCCluster", "EmailParser"]},
            {"name": "Platform", "components": ["EmailAccess", "PhotoLibrary", "AppStoreReceipts"]},
        ],
    },

    "screens": [
        "OnboardingScreen",
        "PermissionsScreen (Email, Photos)",
        "ScanProgressScreen",
        "SubscriptionListScreen",
        "SubscriptionDetailScreen",
        "CancelWizardScreen",
        "SettingsScreen",
    ],
}


# =============================================================================
# BLUEPRINT 4: CHAMELEON (Personalization)
# =============================================================================

CHAMELEON = {
    "name": "Chameleon",
    "tagline": "A home screen that breathes with you",
    "category": "personalization",

    "description": """
Chameleon generates live wallpapers that respond to your actual state.

It reads:
- Battery level and temperature
- CPU load (are you gaming or idle?)
- Time of day and ambient light
- Your motion (walking, sitting, sleeping)
- Optional: heart rate from wearables

Based on these signals, Chameleon renders real-time shaders:
- High stress/heat → Cool blues, slow motion (calming)
- Low battery/evening → Dim, warm embers (preservation)
- Flow state/focus → Sharp, golden geometry (clarity)

It's not just a wallpaper - it's visual biofeedback.
    """.strip(),

    "problem_type": "personalization",

    "core_tech": [
        "somatic",        # Bio-feedback visuals
        "interoception",  # State sensing
    ],

    "features": [
        "Real-time shader rendering based on device state",
        "Battery-aware color temperature",
        "Motion-responsive geometry",
        "Time-of-day color shifting",
        "Optional heart rate integration",
        "Shader marketplace for custom designs",
        "Low power mode for battery preservation",
        "Widget showing current 'aura' state",
    ],

    "revenue_model": {
        "type": "one_time_plus_marketplace",
        "base_price": "$5 one-time",
        "marketplace": "Shader packs $1-3 each",
        "creator_split": "70% to shader creators",
    },

    "target_audience": "Android enthusiasts, creators, anyone who personalizes their device",

    "competitive_gap": "No home screen reacts to user's actual biometrics",

    "ara_advantage": "Somatic Shaders = visuals that breathe with your state",

    "architecture": {
        "layers": [
            {"name": "UI", "components": ["ShaderPreview", "ShaderPicker", "MarketplaceBrowser"]},
            {"name": "State", "components": ["SensorProvider", "ShaderProvider"]},
            {"name": "Core", "components": ["ShaderEngine", "SensorFusion", "InteroceptionReader"]},
            {"name": "Platform", "components": ["WallpaperService", "SensorManager", "HealthConnect"]},
        ],
    },

    "screens": [
        "OnboardingScreen",
        "ShaderGalleryScreen",
        "ShaderPreviewScreen",
        "SensorCalibrationScreen",
        "MarketplaceScreen",
        "SettingsScreen",
    ],
}


# =============================================================================
# BLUEPRINT 5: SENTINEL (Security)
# =============================================================================

SENTINEL = {
    "name": "Sentinel",
    "tagline": "Antivirus is dead. This is an Immune System.",
    "category": "security",

    "description": """
Sentinel is a behavioral-based mobile security app that learns what
"normal" looks like for YOUR phone, then detects when something's wrong.

Traditional antivirus uses signature matching - it only catches known
threats. Sentinel uses proprioception (self-monitoring) to detect
ANOMALIES - behaviors that don't match the baseline.

It learns for 3 days:
- Which apps use network
- How much battery each app consumes
- Background data patterns
- Sensor access patterns

Then it watches for deviations. When a rogue app (tracker, malware)
does something unusual, Sentinel kills its background data immediately.

For enterprises: Collaborative Immunity. When one user detects a new
threat pattern, the anonymized pattern is shared to the fleet.
    """.strip(),

    "problem_type": "anomaly",

    "core_tech": [
        "proprioception",  # Self-monitoring
        "hdc",             # Baseline encoding
    ],

    "features": [
        "3-day baseline learning of normal phone behavior",
        "Real-time anomaly detection",
        "Instant background data kill for rogue apps",
        "Battery drain alerts",
        "Unusual sensor access detection",
        "Network traffic analysis (local, no VPN needed)",
        "Collaborative Immunity (enterprise)",
        "Weekly security reports",
    ],

    "revenue_model": {
        "type": "enterprise",
        "consumer_price": "Free (ad-supported)",
        "enterprise_price": "$100/year per device",
        "enterprise_features": [
            "Collaborative Immunity across fleet",
            "Admin dashboard",
            "Policy enforcement",
            "Threat intelligence sharing",
        ],
    },

    "target_audience": "Security-conscious individuals and enterprise IT teams",

    "competitive_gap": "No mobile security app uses behavioral baselines",

    "ara_advantage": "Proprioception = learns your phone's normal, detects deviations",

    "architecture": {
        "layers": [
            {"name": "UI", "components": ["SecurityDashboard", "ThreatList", "AppAuditor"]},
            {"name": "State", "components": ["BaselineProvider", "ThreatProvider"]},
            {"name": "Core", "components": ["ProprioceptionEngine", "AnomalyDetector", "HDCBaseline"]},
            {"name": "Platform", "components": ["UsageStats", "NetworkStats", "BatteryStats"]},
        ],
    },

    "screens": [
        "OnboardingScreen",
        "BaselineLearningScreen (3-day progress)",
        "SecurityDashboardScreen",
        "ThreatDetailScreen",
        "AppAuditScreen",
        "ReportScreen",
        "SettingsScreen",
    ],
}


# =============================================================================
# Registry
# =============================================================================

BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "sanctum": SANCTUM,
    "aegis": AEGIS,
    "vault": VAULT,
    "chameleon": CHAMELEON,
    "sentinel": SENTINEL,
}


def load_blueprint(name: str) -> Dict[str, Any]:
    """
    Load a blueprint by name.

    Args:
        name: Blueprint name (sanctum, aegis, vault, chameleon, sentinel)

    Returns:
        Blueprint spec dict

    Raises:
        ValueError: If blueprint not found
    """
    name = name.lower()
    if name not in BLUEPRINTS:
        available = ", ".join(BLUEPRINTS.keys())
        raise ValueError(f"Unknown blueprint '{name}'. Available: {available}")

    log.info("Loaded blueprint: %s", name)
    return BLUEPRINTS[name].copy()


def list_blueprints() -> List[str]:
    """List all available blueprint names."""
    return list(BLUEPRINTS.keys())


def get_blueprint_summary(name: str) -> Dict[str, str]:
    """Get a brief summary of a blueprint."""
    bp = load_blueprint(name)
    return {
        "name": bp["name"],
        "tagline": bp["tagline"],
        "category": bp["category"],
        "revenue_model": bp["revenue_model"].get("type", "unknown"),
    }


def get_all_summaries() -> List[Dict[str, str]]:
    """Get summaries of all blueprints."""
    return [get_blueprint_summary(name) for name in BLUEPRINTS.keys()]


__all__ = [
    'BLUEPRINTS',
    'SANCTUM',
    'AEGIS',
    'VAULT',
    'CHAMELEON',
    'SENTINEL',
    'load_blueprint',
    'list_blueprints',
    'get_blueprint_summary',
    'get_all_summaries',
]

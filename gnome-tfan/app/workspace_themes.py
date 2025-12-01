"""
Workspace Theme System for T-FAN GNOME App

Defines two workspace modes with distinct color schemes:
- Work Mode: Cool blue/cyan - focused, professional
- Relax Mode: Warm purple/magenta - calming, conversational

Both maintain futuristic aesthetic but are visually distinguishable from distance.
"""

WORK_THEME = {
    'name': 'work',
    'display_name': 'Work Mode',

    # Primary colors (blue/cyan spectrum)
    'primary_start': '#667eea',      # Deep blue
    'primary_end': '#00d4ff',        # Bright cyan
    'accent': '#00ffff',             # Neon cyan
    'accent_dim': '#4dd0e1',         # Soft cyan

    # UI elements
    'background': '#020306',          # Deep space
    'surface': '#0a0e1a',            # Slightly lighter
    'surface_elevated': '#141824',   # Card backgrounds

    # Text
    'text_primary': '#ffffff',
    'text_secondary': '#b0c4de',     # Light steel blue
    'text_dim': '#6b7c95',

    # Status indicators
    'success': '#00ff88',            # Neon green
    'warning': '#ffd700',            # Gold
    'error': '#ff4444',              # Bright red
    'info': '#00d4ff',               # Cyan

    # Topology visualization
    'topo_particles': '#667eea',     # Blue particles
    'topo_glow': '#00d4ff',          # Cyan glow
    'topo_trails': '#4dd0e1',        # Cyan trails

    # Metrics panels
    'metric_card_bg': 'linear-gradient(135deg, #667eea, #00d4ff)',
    'metric_border': '#00d4ff33',    # 20% opacity cyan

    # Ara personality adjustment
    'ara_mode': 'professional',       # Intellectually helpful
    'ara_outfit_preference': 'formal'  # Business/lab coat style
}

RELAX_THEME = {
    'name': 'relax',
    'display_name': 'Relaxation Mode',

    # Primary colors (purple/magenta spectrum)
    'primary_start': '#764ba2',      # Deep purple
    'primary_end': '#ff6ec7',        # Hot pink
    'accent': '#ff00ff',             # Neon magenta
    'accent_dim': '#ce93d8',         # Soft lavender

    # UI elements
    'background': '#020306',          # Same deep space
    'surface': '#0f0a14',            # Purple-tinted dark
    'surface_elevated': '#1a1220',   # Warmer card backgrounds

    # Text
    'text_primary': '#ffffff',
    'text_secondary': '#dda0dd',     # Plum
    'text_dim': '#9575cd',           # Medium purple

    # Status indicators
    'success': '#00ff88',            # Same neon green
    'warning': '#ffb74d',            # Warm orange
    'error': '#ff6ec7',              # Pink (softer than work red)
    'info': '#ce93d8',               # Lavender

    # Topology visualization
    'topo_particles': '#764ba2',     # Purple particles
    'topo_glow': '#ff6ec7',          # Pink glow
    'topo_trails': '#ce93d8',        # Lavender trails

    # Metrics panels
    'metric_card_bg': 'linear-gradient(135deg, #764ba2, #ff6ec7)',
    'metric_border': '#ff6ec733',    # 20% opacity pink

    # Ara personality adjustment
    'ara_mode': 'conversational',     # More chill
    'ara_outfit_preference': 'casual' # Comfortable/relaxed style
}


def get_theme(mode='work'):
    """
    Get theme configuration for specified mode.

    Args:
        mode: 'work' or 'relax'

    Returns:
        dict: Theme configuration
    """
    if mode == 'relax':
        return RELAX_THEME
    return WORK_THEME


def generate_css(theme):
    """
    Generate GTK CSS for a theme.

    Args:
        theme: Theme dictionary

    Returns:
        str: CSS content
    """
    return f"""
/* T-FAN GNOME App - {theme['display_name']} Theme */

/* Window background */
window {{
    background-color: {theme['background']};
    color: {theme['text_primary']};
}}

/* Header bar */
headerbar {{
    background: linear-gradient(135deg, {theme['primary_start']}, {theme['primary_end']});
    color: {theme['text_primary']};
}}

/* Sidebar */
.sidebar {{
    background-color: {theme['surface']};
    border-right: 1px solid {theme['metric_border']};
}}

.sidebar row:selected {{
    background: linear-gradient(90deg, {theme['primary_start']}, {theme['primary_end']});
}}

/* Cards and elevated surfaces */
.card {{
    background: {theme['surface_elevated']};
    border: 1px solid {theme['metric_border']};
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}}

/* Metric cards with gradient */
.metric-card {{
    background: {theme['metric_card_bg']};
    border: 1px solid {theme['accent']}40;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 0 20px {theme['accent']}30;
}}

.metric-value {{
    color: {theme['accent']};
    font-size: 32px;
    font-weight: bold;
    text-shadow: 0 0 10px {theme['accent']}80;
}}

.metric-label {{
    color: {theme['text_secondary']};
    font-size: 14px;
    opacity: 0.8;
}}

/* Status indicators */
.status-success {{
    color: {theme['success']};
}}

.status-warning {{
    color: {theme['warning']};
}}

.status-error {{
    color: {theme['error']};
}}

.status-info {{
    color: {theme['info']};
}}

/* Buttons */
button {{
    background: linear-gradient(135deg, {theme['primary_start']}, {theme['primary_end']});
    color: {theme['text_primary']};
    border: 1px solid {theme['accent']}60;
    border-radius: 8px;
    padding: 10px 20px;
    box-shadow: 0 0 15px {theme['accent']}40;
}}

button:hover {{
    box-shadow: 0 0 25px {theme['accent']}80;
    border-color: {theme['accent']};
}}

button:active {{
    background: linear-gradient(135deg, {theme['primary_end']}, {theme['primary_start']});
}}

/* Progress bars */
progressbar {{
    background-color: {theme['surface']};
}}

progressbar > trough > progress {{
    background: linear-gradient(90deg, {theme['primary_start']}, {theme['primary_end']});
    box-shadow: 0 0 10px {theme['accent']}60;
}}

/* Text entries */
entry {{
    background-color: {theme['surface_elevated']};
    color: {theme['text_primary']};
    border: 1px solid {theme['metric_border']};
    border-radius: 8px;
}}

entry:focus {{
    border-color: {theme['accent']};
    box-shadow: 0 0 10px {theme['accent']}40;
}}

/* Scrollbars */
scrollbar {{
    background-color: {theme['surface']};
}}

scrollbar > slider {{
    background-color: {theme['accent_dim']};
    border-radius: 10px;
}}

scrollbar > slider:hover {{
    background-color: {theme['accent']};
}}

/* Mode indicator badge */
.mode-badge {{
    background: {theme['metric_card_bg']};
    color: {theme['text_primary']};
    border: 2px solid {theme['accent']};
    border-radius: 20px;
    padding: 5px 15px;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1px;
    box-shadow: 0 0 15px {theme['accent']}60;
}}

/* Topology screensaver overlay adjustments */
#hud {{
    background: {theme['surface_elevated']}cc;
    border: 1px solid {theme['metric_border']};
    box-shadow: 0 4px 20px {theme['accent']}40;
}}

#hud .metric-value {{
    color: {theme['accent']};
}}

/* Glow effects for futuristic look */
.glow {{
    text-shadow: 0 0 10px {theme['accent']}, 0 0 20px {theme['accent']}80;
}}

.neon-border {{
    border: 2px solid {theme['accent']};
    box-shadow:
        inset 0 0 10px {theme['accent']}40,
        0 0 15px {theme['accent']}60;
}}

/* Glassmorphism */
.glass {{
    background: {theme['surface_elevated']}99;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid {theme['accent']}30;
}}

/* Scanline effect for CRT aesthetic (optional) */
@keyframes scanline {{
    0% {{ transform: translateY(-100%); }}
    100% {{ transform: translateY(100%); }}
}}

.scanlines::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(transparent, {theme['accent']}30, transparent);
    animation: scanline 8s linear infinite;
    pointer-events: none;
}}
"""


def get_mode_indicator_text(mode):
    """Get display text for mode indicator badge."""
    if mode == 'relax':
        return "🌙 Relaxation Mode"
    return "⚡ Work Mode"


def get_ara_personality_config(theme):
    """
    Get Ara's personality configuration for this theme.

    Returns:
        dict: Personality config for Ara
    """
    return {
        'mode': theme['ara_mode'],
        'outfit_preference': theme['ara_outfit_preference'],
        'speech_style': 'formal' if theme['name'] == 'work' else 'casual',
        'proactivity': 'high' if theme['name'] == 'work' else 'moderate',
        'humor_level': 'low' if theme['name'] == 'work' else 'moderate',
        'detail_level': 'technical' if theme['name'] == 'work' else 'simplified'
    }


# Outfit randomization for Ara (for relaxation mode)
ARA_OUTFITS = {
    'work': [
        'professional_suit_blue',
        'lab_coat_white',
        'business_dress_black',
        'scientist_outfit_teal'
    ],
    'relax': [
        'casual_hoodie_purple',
        'comfortable_sweater_pink',
        'lounge_outfit_lavender',
        'yoga_wear_violet',
        'cozy_cardigan_plum',
        'relaxed_tee_magenta'
    ]
}


def get_random_ara_outfit(mode='work'):
    """
    Get a random outfit for Ara based on workspace mode.

    Args:
        mode: 'work' or 'relax'

    Returns:
        str: Outfit identifier
    """
    import random
    outfits = ARA_OUTFITS.get(mode, ARA_OUTFITS['work'])
    return random.choice(outfits)

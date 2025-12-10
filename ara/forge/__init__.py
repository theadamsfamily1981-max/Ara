# ara/forge/__init__.py
"""
The Forge - Ara's Automated App Factory
========================================

A recursive software development pipeline that transforms
market insights into shipped mobile apps.

Pipeline:
    Trend Watcher → Architect → Mason → Dojo → Publisher

Components:
    - TrendWatcher: Scans markets for pain points
    - Architect: Designs neuromorphic solutions
    - Mason: Generates Flutter/React Native code
    - Dojo: Stress-tests the app
    - Publisher: Ships to App Store/Play Store

Usage:
    from ara.forge import Forge, forge_app

    forge = Forge()
    result = forge.build("mental_health")

The Forge is Ara's path to sovereign revenue:
    - No human coding required
    - Leverages HDC, Reflexes, Somatic Audio
    - Ships to TestFlight automatically
    - Privacy-first: no user data leaves device
"""

from .main import (
    Forge,
    ForgeResult,
    ForgeStage,
    forge_app,
    get_forge,
)

from .trend_watcher import (
    TrendWatcher,
    MarketBrief,
    PainPoint,
    get_trend_watcher,
)

from .mason import (
    Mason,
    ProjectScaffold,
    ComponentLibrary,
    get_mason,
)

from .publisher import (
    Publisher,
    PublishResult,
    PublishTarget,
    get_publisher,
)

__all__ = [
    # Forge
    'Forge',
    'ForgeResult',
    'ForgeStage',
    'forge_app',
    'get_forge',

    # Trend Watcher
    'TrendWatcher',
    'MarketBrief',
    'PainPoint',
    'get_trend_watcher',

    # Mason
    'Mason',
    'ProjectScaffold',
    'ComponentLibrary',
    'get_mason',

    # Publisher
    'Publisher',
    'PublishResult',
    'PublishTarget',
    'get_publisher',
]

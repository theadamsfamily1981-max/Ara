# ara/forge/main.py
"""
The Forge - Main Orchestrator
==============================

The recursive software development pipeline that builds
apps using Ara's core technologies.

Pipeline Flow:
    1. Trend Watcher analyzes market → ProductBrief
    2. Architect designs solution → Spec + Architecture
    3. Mason scaffolds code → Flutter/React Native project
    4. Dojo stress-tests → Bug reports
    5. Mason fixes bugs → Iterate until passing
    6. Publisher ships → TestFlight/Play Store

Core Tech Integration:
    - HDC (Hyperdimensional Computing) for local AI
    - Reflexes (TCAM/eBPF) for instant response
    - Somatic Audio for biofeedback
    - Teleology for intent-aware behavior
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

log = logging.getLogger("Ara.Forge")


# =============================================================================
# Types
# =============================================================================

class ForgeStage(str, Enum):
    """Stages in the forge pipeline."""
    ANALYZE = "analyze"         # Trend Watcher
    DESIGN = "design"           # Architect
    SCAFFOLD = "scaffold"       # Mason creates project
    ITERATE = "iterate"         # Mason + Dojo loop
    PUBLISH = "publish"         # Publisher ships
    COMPLETE = "complete"


class CoreTech(str, Enum):
    """Ara's core technologies available to apps."""
    HDC = "hdc"                 # Hyperdimensional Computing
    REFLEXES = "reflexes"       # TCAM/eBPF fast path
    SOMATIC = "somatic"         # Audio biofeedback
    TELEOLOGY = "teleology"     # Intent-aware behavior
    INTEROCEPTION = "interoception"  # Body state sensing
    PROPRIOCEPTION = "proprioception"  # Self-monitoring


class AppCategory(str, Enum):
    """Target app categories."""
    MENTAL_HEALTH = "mental_health"
    PRODUCTIVITY = "productivity"
    FINANCE = "finance"
    SECURITY = "security"
    PERSONALIZATION = "personalization"


@dataclass
class ForgeResult:
    """Result of a forge build."""
    app_name: str
    category: AppCategory
    stage: ForgeStage
    success: bool

    # Artifacts
    brief: Optional[Dict[str, Any]] = None
    spec: Optional[Dict[str, Any]] = None
    project_path: Optional[Path] = None

    # Metrics
    iterations: int = 0
    dojo_reports: List[Dict[str, Any]] = field(default_factory=list)

    # Publishing
    testflight_url: Optional[str] = None
    playstore_url: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "category": self.category.value,
            "stage": self.stage.value,
            "success": self.success,
            "iterations": self.iterations,
            "duration_seconds": self.duration_seconds,
            "testflight_url": self.testflight_url,
            "playstore_url": self.playstore_url,
        }


# =============================================================================
# Forge
# =============================================================================

class Forge:
    """
    The Automated App Factory.

    Coordinates:
        - TrendWatcher: Market analysis
        - Architect: Solution design
        - Mason: Code generation
        - Dojo: Quality assurance
        - Publisher: App Store submission

    Usage:
        forge = Forge()
        result = await forge.build("mental_health", blueprint="sanctum")
    """

    MAX_ITERATIONS = 10  # Max bug-fix cycles

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        target_platform: str = "flutter",  # flutter | react_native
    ):
        """
        Initialize the Forge.

        Args:
            output_dir: Where to create app projects
            target_platform: Flutter or React Native
        """
        self.output_dir = output_dir or Path.home() / ".ara" / "forge" / "projects"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_platform = target_platform

        # Lazy-load components
        self._trend_watcher = None
        self._architect = None
        self._mason = None
        self._dojo = None
        self._publisher = None

        log.info("Forge initialized: platform=%s, output=%s",
                 target_platform, self.output_dir)

    @property
    def trend_watcher(self):
        if self._trend_watcher is None:
            from .trend_watcher import TrendWatcher
            self._trend_watcher = TrendWatcher()
        return self._trend_watcher

    @property
    def architect(self):
        if self._architect is None:
            from ara.academy.skills.architect import get_architect
            self._architect = get_architect()
        return self._architect

    @property
    def mason(self):
        if self._mason is None:
            from .mason import Mason
            self._mason = Mason(platform=self.target_platform)
        return self._mason

    @property
    def dojo(self):
        if self._dojo is None:
            from ara.academy.dojo import get_dojo
            self._dojo = get_dojo()
        return self._dojo

    @property
    def publisher(self):
        if self._publisher is None:
            from .publisher import Publisher
            self._publisher = Publisher()
        return self._publisher

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    async def build(
        self,
        category: str,
        blueprint: Optional[str] = None,
        auto_publish: bool = False,
    ) -> ForgeResult:
        """
        Run the full forge pipeline.

        Args:
            category: App category (mental_health, productivity, etc.)
            blueprint: Specific blueprint name (sanctum, aegis, etc.)
            auto_publish: If True, publish to TestFlight automatically

        Returns:
            ForgeResult with all artifacts and status
        """
        cat = AppCategory(category) if isinstance(category, str) else category

        result = ForgeResult(
            app_name=blueprint or f"ara_{category}_app",
            category=cat,
            stage=ForgeStage.ANALYZE,
            success=False,
            started_at=datetime.utcnow(),
        )

        try:
            # Stage 1: Analyze Market
            log.info("Forge: Stage 1 - Analyzing market for %s", category)
            result.stage = ForgeStage.ANALYZE

            if blueprint:
                brief = await self._load_blueprint(blueprint)
            else:
                brief = await self.trend_watcher.analyze_market(category)

            result.brief = brief

            # Stage 2: Design Solution
            log.info("Forge: Stage 2 - Designing solution")
            result.stage = ForgeStage.DESIGN

            spec = await self._design_solution(brief)
            result.spec = spec

            # Stage 3: Scaffold Project
            log.info("Forge: Stage 3 - Scaffolding project")
            result.stage = ForgeStage.SCAFFOLD

            project_path = await self.mason.scaffold_project(
                spec,
                output_dir=self.output_dir / result.app_name,
            )
            result.project_path = project_path

            # Stage 4: Iterate (Dojo + Fix loop)
            log.info("Forge: Stage 4 - Iterating (Dojo + Mason)")
            result.stage = ForgeStage.ITERATE

            passed = False
            for i in range(self.MAX_ITERATIONS):
                result.iterations = i + 1

                # Run Dojo stress test
                report = await self._stress_test(project_path)
                result.dojo_reports.append(report)

                if report.get("passed", False):
                    log.info("Forge: Dojo PASSED on iteration %d", i + 1)
                    passed = True
                    break

                # Fix bugs
                log.info("Forge: Dojo found issues, fixing (iteration %d)", i + 1)
                await self.mason.fix_bugs(project_path, report)

            if not passed:
                log.warning("Forge: Max iterations reached without Dojo pass")
                result.completed_at = datetime.utcnow()
                return result

            # Stage 5: Publish
            if auto_publish:
                log.info("Forge: Stage 5 - Publishing")
                result.stage = ForgeStage.PUBLISH

                pub_result = await self.publisher.ship_to_testflight(project_path)
                result.testflight_url = pub_result.get("testflight_url")

            # Complete
            result.stage = ForgeStage.COMPLETE
            result.success = True
            result.completed_at = datetime.utcnow()

            log.info("Forge: Build complete! App=%s, Duration=%.1fs",
                     result.app_name, result.duration_seconds)

            return result

        except Exception as e:
            log.error("Forge: Build failed at stage %s: %s", result.stage, e)
            result.completed_at = datetime.utcnow()
            return result

    # =========================================================================
    # Stage Implementations
    # =========================================================================

    async def _load_blueprint(self, name: str) -> Dict[str, Any]:
        """Load a predefined app blueprint."""
        from .blueprints import load_blueprint
        return load_blueprint(name)

    async def _design_solution(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Use Architect to design the solution."""
        # Determine which core tech to use based on brief
        problem_type = brief.get("problem_type", "general")
        category = brief.get("category", "")

        core_tech = []
        if "mental" in category or "stress" in problem_type:
            core_tech.extend([CoreTech.SOMATIC, CoreTech.INTEROCEPTION])
        if "privacy" in problem_type or "security" in category:
            core_tech.extend([CoreTech.HDC, CoreTech.REFLEXES])
        if "intent" in problem_type or "focus" in problem_type:
            core_tech.append(CoreTech.TELEOLOGY)
        if "anomaly" in problem_type or "monitoring" in problem_type:
            core_tech.append(CoreTech.PROPRIOCEPTION)

        if not core_tech:
            core_tech = [CoreTech.HDC]

        spec = {
            "name": brief.get("suggested_name", "ara_app"),
            "description": brief.get("description", ""),
            "category": brief.get("category", "general"),
            "core_tech": [t.value for t in core_tech],
            "features": brief.get("features", []),
            "revenue_model": brief.get("revenue_model", "freemium"),
            "privacy_first": True,  # Always true for Ara apps
            "platforms": ["ios", "android"],
            "ux_flow": self._generate_ux_flow(brief),
            "architecture": self._generate_architecture(brief, core_tech),
        }

        return spec

    def _generate_ux_flow(self, brief: Dict[str, Any]) -> str:
        """Generate Mermaid UX flow diagram."""
        name = brief.get("suggested_name", "App")

        flow = f"""
graph TD
    A[Launch {name}] --> B{{First Time?}}
    B -->|Yes| C[Onboarding]
    B -->|No| D[Main Screen]
    C --> E[Explain Privacy]
    E --> F[Grant Permissions]
    F --> D
    D --> G[Core Feature]
    G --> H{{Success?}}
    H -->|Yes| I[Show Result]
    H -->|No| J[Retry/Help]
    J --> G
    I --> D
"""
        return flow.strip()

    def _generate_architecture(
        self,
        brief: Dict[str, Any],
        core_tech: List[CoreTech],
    ) -> Dict[str, Any]:
        """Generate architecture specification."""
        arch = {
            "layers": [
                {
                    "name": "UI",
                    "framework": "Flutter" if self.target_platform == "flutter" else "React Native",
                    "components": ["Screens", "Widgets", "Themes"],
                },
                {
                    "name": "State",
                    "framework": "Riverpod" if self.target_platform == "flutter" else "Redux",
                    "components": ["Providers", "Actions", "Selectors"],
                },
                {
                    "name": "Core",
                    "framework": "Ara SDK",
                    "components": [t.value for t in core_tech],
                },
                {
                    "name": "Platform",
                    "framework": "Native",
                    "components": ["HealthKit", "NetworkExtension", "LocalAuth"],
                },
            ],
            "data_flow": "unidirectional",
            "privacy": {
                "data_on_device": True,
                "cloud_sync": False,
                "encryption": "AES-256",
            },
        }

        return arch

    async def _stress_test(self, project_path: Path) -> Dict[str, Any]:
        """Run Dojo stress test on the project."""
        # For now, simulate stress testing
        # Real implementation would:
        # 1. Build the app
        # 2. Run in headless emulator
        # 3. Fuzz test with random inputs
        # 4. Check for crashes, memory leaks, UI jank

        report = {
            "passed": True,  # Start optimistic
            "issues": [],
            "coverage": 0.0,
            "memory_peak_mb": 0,
            "startup_ms": 0,
        }

        # Check if project has required files
        required_files = [
            "pubspec.yaml" if self.target_platform == "flutter" else "package.json",
            "lib/main.dart" if self.target_platform == "flutter" else "src/App.js",
        ]

        for req in required_files:
            if not (project_path / req).exists():
                report["passed"] = False
                report["issues"].append({
                    "type": "missing_file",
                    "file": req,
                    "severity": "critical",
                })

        return report

    # =========================================================================
    # Quick Build
    # =========================================================================

    async def quick_build(self, blueprint: str) -> ForgeResult:
        """
        Quick build from a predefined blueprint.

        Args:
            blueprint: Blueprint name (sanctum, aegis, vault, chameleon, sentinel)

        Returns:
            ForgeResult
        """
        brief = await self._load_blueprint(blueprint)
        category = brief.get("category", "general")
        return await self.build(category, blueprint=blueprint)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_forge: Optional[Forge] = None


def get_forge() -> Forge:
    """Get the default forge instance."""
    global _default_forge
    if _default_forge is None:
        _default_forge = Forge()
    return _default_forge


async def forge_app(
    category: str,
    blueprint: Optional[str] = None,
    auto_publish: bool = False,
) -> ForgeResult:
    """
    Convenience function to forge an app.

    Args:
        category: App category
        blueprint: Optional blueprint name
        auto_publish: Auto-publish to TestFlight

    Returns:
        ForgeResult
    """
    forge = get_forge()
    return await forge.build(category, blueprint, auto_publish)


def forge_blueprint(blueprint: str) -> ForgeResult:
    """
    Synchronous convenience to forge from blueprint.

    Usage:
        result = forge_blueprint("sanctum")
    """
    return asyncio.run(get_forge().quick_build(blueprint))


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Forge - Automated App Factory")
    parser.add_argument("blueprint", help="Blueprint to build (sanctum, aegis, vault, chameleon, sentinel)")
    parser.add_argument("--publish", action="store_true", help="Auto-publish to TestFlight")
    parser.add_argument("--platform", default="flutter", choices=["flutter", "react_native"])

    args = parser.parse_args()

    forge = Forge(target_platform=args.platform)
    result = asyncio.run(forge.quick_build(args.blueprint))

    print(f"\n{'='*60}")
    print(f"Forge Result: {result.app_name}")
    print(f"{'='*60}")
    print(f"Stage: {result.stage.value}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    if result.project_path:
        print(f"Project: {result.project_path}")
    if result.testflight_url:
        print(f"TestFlight: {result.testflight_url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

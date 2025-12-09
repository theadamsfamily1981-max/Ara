"""
Self-Publishing Service
========================

Orchestrates the full publishing pipeline:
1. Research → Generate research brief
2. Prototype → Build content assets
3. Production → Polish and validate
4. Ship → Deploy to platforms

Usage:
    from ara.jobs.publishing_service import SelfPublishingService

    service = SelfPublishingService()
    result = service.execute(topic="Spiking neural networks for Ara")

Or via CLI:
    python -m ara.jobs.publishing_service --topic "Your topic here"
"""

from __future__ import annotations

import os
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from ara.enterprise.publishing import (
    PublishingFactory,
    ContentBundle,
    ResearchBrief,
)
from ara.enterprise.shipping import (
    PublishingDispatcher,
    ShippingReport,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Job Configuration
# =============================================================================

@dataclass
class PublishingJobConfig:
    """Configuration for a publishing job."""
    job_id: str
    topic: str

    # Targets
    repo_owner: str = "theadamsfamily1981-max"
    repo_name: str = "Ara"
    targets: List[str] = field(default_factory=lambda: ["local"])

    # Paths
    artifacts_dir: str = "artifacts"

    # Options
    auto_ship: bool = False  # If True, ships without confirmation
    skip_production: bool = False  # Skip production stage (for quick tests)

    @classmethod
    def from_yaml(cls, path: str) -> "PublishingJobConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            job_id=data.get("job_id", f"pub_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
            topic=data.get("inputs", {}).get("topic", ""),
            repo_owner=data.get("inputs", {}).get("repo_owner", "theadamsfamily1981-max"),
            repo_name=data.get("inputs", {}).get("repo_name", "Ara"),
            targets=data.get("targets", ["local"]),
            artifacts_dir=data.get("artifacts_dir", "artifacts"),
            auto_ship=data.get("auto_ship", False),
            skip_production=data.get("skip_production", False),
        )


# =============================================================================
# Job Result
# =============================================================================

@dataclass
class PublishingJobResult:
    """Result of a publishing job."""
    job_id: str
    topic: str
    success: bool

    # Stage outputs
    research_brief_path: Optional[str] = None
    prototypes_dir: Optional[str] = None
    production_dir: Optional[str] = None
    shipping_report: Optional[ShippingReport] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Errors
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "success": self.success,
            "research_brief_path": self.research_brief_path,
            "prototypes_dir": self.prototypes_dir,
            "production_dir": self.production_dir,
            "shipping_success": self.shipping_report.all_successful if self.shipping_report else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


# =============================================================================
# Self-Publishing Service
# =============================================================================

class SelfPublishingService:
    """
    Orchestrates the self-publishing pipeline.

    Full pipeline:
    1. Research: Generate research brief from topic
    2. Prototype: Build content assets (thread, newsletter, blog)
    3. Production: Polish and validate content
    4. Ship: Deploy to configured platforms
    """

    def __init__(self, config: Optional[PublishingJobConfig] = None):
        self.config = config
        self.factory = PublishingFactory()
        self.dispatcher = PublishingDispatcher()

    def execute(
        self,
        topic: str,
        config: Optional[PublishingJobConfig] = None,
    ) -> PublishingJobResult:
        """
        Execute the full publishing pipeline.

        Args:
            topic: The topic to publish about
            config: Optional config override

        Returns:
            PublishingJobResult with all outputs
        """
        cfg = config or self.config or PublishingJobConfig(
            job_id=f"pub_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            topic=topic,
        )

        result = PublishingJobResult(
            job_id=cfg.job_id,
            topic=topic,
            success=False,
        )

        try:
            # Ensure directories
            research_dir = os.path.join(cfg.artifacts_dir, "research")
            prototypes_dir = os.path.join(cfg.artifacts_dir, "prototypes")
            production_dir = os.path.join(cfg.artifacts_dir, "production")
            shipped_dir = os.path.join(cfg.artifacts_dir, "shipped")

            for d in [research_dir, prototypes_dir, production_dir, shipped_dir]:
                os.makedirs(d, exist_ok=True)

            # Stage 1: Research
            logger.info(f"[{cfg.job_id}] Stage 1: Research")
            project = self.factory.create_content_project(topic=topic)
            research_brief = self.factory.generate_research(
                project_id=project.id,
                output_dir=research_dir,
            )
            result.research_brief_path = os.path.join(research_dir, f"{project.id}_research.md")

            if not research_brief:
                raise RuntimeError("Research generation failed")

            # Advance pipeline stage
            self.factory.advance_stage(project.id, notes="Research complete", force=True)

            # Stage 2: Prototype
            logger.info(f"[{cfg.job_id}] Stage 2: Prototype")
            bundle = self.factory.build_prototypes(
                project_id=project.id,
                research_brief=research_brief,
                output_dir=prototypes_dir,
            )
            result.prototypes_dir = prototypes_dir

            self.factory.advance_stage(project.id, notes="Prototypes built", force=True)

            # Stage 3: Production
            if not cfg.skip_production:
                logger.info(f"[{cfg.job_id}] Stage 3: Production")
                bundle = self.factory.productionize(
                    project_id=project.id,
                    bundle=bundle,
                    output_dir=production_dir,
                )
                result.production_dir = production_dir

                self.factory.advance_stage(project.id, notes="Production ready", force=True)

            # Stage 4: Ship
            if cfg.auto_ship or "local" in cfg.targets:
                logger.info(f"[{cfg.job_id}] Stage 4: Ship")
                shipping_report = self.dispatcher.ship_content(
                    bundle=bundle,
                    targets=cfg.targets,
                    repo_owner=cfg.repo_owner,
                    repo_name=cfg.repo_name,
                    output_dir=shipped_dir,
                )
                result.shipping_report = shipping_report

                if shipping_report.all_successful:
                    self.factory.ship(project.id, version="1.0.0", notes="Published via SelfPublishingService")

            result.success = True
            result.completed_at = datetime.utcnow()

            logger.info(f"[{cfg.job_id}] Pipeline complete in {result.duration_seconds:.1f}s")

        except Exception as e:
            logger.error(f"[{cfg.job_id}] Pipeline failed: {e}")
            result.success = False
            result.error = str(e)
            result.completed_at = datetime.utcnow()

        return result

    def execute_from_yaml(self, config_path: str) -> PublishingJobResult:
        """Execute pipeline from YAML config file."""
        config = PublishingJobConfig.from_yaml(config_path)
        return self.execute(topic=config.topic, config=config)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Self-Publishing Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic to publish about",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--auto-ship",
        action="store_true",
        help="Automatically ship without confirmation",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["local"],
        help="Shipping targets (local, github, twitter, email)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Execute
    service = SelfPublishingService()

    if args.config:
        result = service.execute_from_yaml(args.config)
    elif args.topic:
        config = PublishingJobConfig(
            job_id=f"cli_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            topic=args.topic,
            targets=args.targets,
            auto_ship=args.auto_ship,
        )
        result = service.execute(topic=args.topic, config=config)
    else:
        parser.print_help()
        return

    # Report
    print("\n" + "=" * 60)
    print("PUBLISHING JOB RESULT")
    print("=" * 60)
    print(f"Job ID: {result.job_id}")
    print(f"Topic: {result.topic}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    if result.research_brief_path:
        print(f"Research: {result.research_brief_path}")
    if result.prototypes_dir:
        print(f"Prototypes: {result.prototypes_dir}")
    if result.production_dir:
        print(f"Production: {result.production_dir}")
    if result.shipping_report:
        print(f"Shipped: {result.shipping_report.success_count} assets")

    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    main()

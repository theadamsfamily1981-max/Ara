"""
Publishing Dispatcher (Shipping)
=================================

Ships content bundles to various platforms.

Platforms:
- GitHub: Push to content repo
- Twitter: (TODO) Post threads via API
- Email: (TODO) Send newsletters via email service
- Blog: (TODO) Deploy to static site / CMS
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .publishing import ContentBundle, ContentAsset

# Covenant integration
try:
    from ara.utils.covenant import get_covenant, AutomationLevel, CovenantViolation
    COVENANT_AVAILABLE = True
except ImportError:
    COVENANT_AVAILABLE = False
    get_covenant = None
    AutomationLevel = None
    CovenantViolation = None

logger = logging.getLogger(__name__)


# =============================================================================
# Shipping Results
# =============================================================================

@dataclass
class ShipmentResult:
    """Result of shipping a single asset."""
    asset_type: str
    platform: str
    success: bool
    destination: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    covenant_violations: List[Any] = field(default_factory=list)  # CovenantViolation
    blocked_by_covenant: bool = False


@dataclass
class ShippingReport:
    """Full shipping report for a content bundle."""
    bundle_id: str
    shipped_at: datetime = field(default_factory=datetime.utcnow)
    results: List[ShipmentResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def all_successful(self) -> bool:
        return self.failure_count == 0

    @property
    def covenant_blocked_count(self) -> int:
        return sum(1 for r in self.results if r.blocked_by_covenant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "shipped_at": self.shipped_at.isoformat(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "covenant_blocked_count": self.covenant_blocked_count,
            "results": [
                {
                    "asset_type": r.asset_type,
                    "platform": r.platform,
                    "success": r.success,
                    "destination": r.destination,
                    "error": r.error,
                    "blocked_by_covenant": r.blocked_by_covenant,
                    "covenant_violations": [
                        {"rule_id": v.rule_id, "message": v.message, "severity": v.severity}
                        for v in r.covenant_violations
                    ] if r.covenant_violations else [],
                }
                for r in self.results
            ],
        }


# =============================================================================
# Publishing Dispatcher
# =============================================================================

class PublishingDispatcher:
    """
    Ships content to various platforms.

    For v1, primarily ships to:
    - GitHub (push to content repo)
    - Local (write to output directory)

    Future platforms:
    - Twitter/X
    - Email (via SendGrid, SES, etc.)
    - Blog (GitHub Pages, Vercel, etc.)

    Covenant Integration:
    - Checks automation level before shipping
    - Runs content through covenant guardrails
    - Blocks shipping if covenant violations detected
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # GitHub integration
        self._github_client = None

        # Covenant
        self._covenant = None
        if COVENANT_AVAILABLE:
            try:
                self._covenant = get_covenant()
            except Exception as e:
                logger.warning(f"Could not load covenant: {e}")

    @property
    def github(self):
        """Lazy-load GitHub client."""
        if self._github_client is None:
            try:
                from ara.integrations.github_api import get_github_client
                self._github_client = get_github_client()
            except ImportError:
                logger.warning("GitHub client not available")
        return self._github_client

    @property
    def covenant(self):
        """Get covenant instance."""
        return self._covenant

    # =========================================================================
    # Covenant Checks
    # =========================================================================

    def check_covenant(
        self,
        asset: ContentAsset,
        automation_level: int = 0,
    ) -> tuple:
        """
        Check if asset passes covenant guardrails.

        Args:
            asset: Content to check
            automation_level: Current automation level (0, 1, 2)

        Returns:
            (passes, violations) tuple
        """
        if not self._covenant:
            # No covenant loaded - pass by default
            return True, []

        # Run content checks
        passes, violations = self._covenant.check_content(
            text=asset.content,
            content_type=asset.asset_type,
        )

        # If automation level 2 (auto-ship), be stricter
        if automation_level == 2 and violations:
            # Any violation at auto-ship level is blocking
            passes = False

        return passes, violations

    def get_automation_level(self, content_type: str) -> int:
        """Get automation level for a content type."""
        if not self._covenant:
            return 0  # Default to most restrictive
        return int(self._covenant.get_automation_level(content_type))

    # =========================================================================
    # Main Shipping Method
    # =========================================================================

    def ship_content(
        self,
        bundle: ContentBundle,
        targets: Optional[List[str]] = None,
        repo_owner: Optional[str] = None,
        repo_name: Optional[str] = None,
        output_dir: str = "artifacts/shipped",
    ) -> ShippingReport:
        """
        Ship a content bundle to target platforms.

        Args:
            bundle: Content bundle to ship
            targets: List of platforms to ship to (default: all)
            repo_owner: GitHub repo owner (for github target)
            repo_name: GitHub repo name (for github target)
            output_dir: Local output directory

        Returns:
            ShippingReport with results
        """
        report = ShippingReport(bundle_id=bundle.bundle_id)

        # Default targets
        if targets is None:
            targets = ["local"]  # Safe default
            if repo_owner and repo_name:
                targets.append("github")

        os.makedirs(output_dir, exist_ok=True)

        for asset in bundle.assets:
            # Get automation level for this asset type
            auto_level = self.get_automation_level(asset.asset_type)
            logger.debug(f"Asset {asset.asset_type} automation level: {auto_level}")

            for target in targets:
                result = self._ship_asset(
                    asset=asset,
                    target=target,
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    output_dir=output_dir,
                    bundle=bundle,
                    automation_level=auto_level,
                )
                report.results.append(result)

        logger.info(
            f"PublishingDispatcher: shipped bundle {bundle.bundle_id} "
            f"({report.success_count} success, {report.failure_count} failed)"
        )

        # Write shipping report
        import json
        report_path = os.path.join(output_dir, f"{bundle.bundle_id}_shipping_report.json")
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

    def _ship_asset(
        self,
        asset: ContentAsset,
        target: str,
        repo_owner: Optional[str],
        repo_name: Optional[str],
        output_dir: str,
        bundle: ContentBundle,
        automation_level: int = 0,
    ) -> ShipmentResult:
        """Ship a single asset to a target."""

        # Run covenant checks
        passes_covenant, violations = self.check_covenant(asset, automation_level)

        if not passes_covenant:
            blocking_violations = [v for v in violations if v.blocking]
            logger.warning(
                f"Asset {asset.asset_type} blocked by covenant: "
                f"{len(blocking_violations)} blocking violations"
            )
            return ShipmentResult(
                asset_type=asset.asset_type,
                platform=target,
                success=False,
                error="Blocked by covenant guardrails",
                covenant_violations=violations,
                blocked_by_covenant=True,
            )

        # Log non-blocking violations as warnings
        if violations:
            for v in violations:
                logger.warning(f"Covenant warning for {asset.asset_type}: {v.message}")

        if target == "local":
            return self._ship_to_local(asset, output_dir)

        elif target == "github":
            if not repo_owner or not repo_name:
                return ShipmentResult(
                    asset_type=asset.asset_type,
                    platform="github",
                    success=False,
                    error="repo_owner and repo_name required for GitHub shipping",
                )
            return self._ship_to_github(asset, repo_owner, repo_name, bundle)

        elif target == "twitter":
            return self._ship_to_twitter(asset)

        elif target == "email":
            return self._ship_to_email(asset)

        else:
            return ShipmentResult(
                asset_type=asset.asset_type,
                platform=target,
                success=False,
                error=f"Unknown target: {target}",
            )

    # =========================================================================
    # Platform-Specific Shippers
    # =========================================================================

    def _ship_to_local(self, asset: ContentAsset, output_dir: str) -> ShipmentResult:
        """Ship to local filesystem."""
        try:
            # Determine filename
            ext = ".md" if asset.platform in ("blog", "email") else ".txt"
            filename = f"{asset.asset_type}{ext}"
            dest_path = os.path.join(output_dir, filename)

            with open(dest_path, 'w') as f:
                f.write(asset.content)

            return ShipmentResult(
                asset_type=asset.asset_type,
                platform="local",
                success=True,
                destination=dest_path,
            )
        except Exception as e:
            return ShipmentResult(
                asset_type=asset.asset_type,
                platform="local",
                success=False,
                error=str(e),
            )

    def _ship_to_github(
        self,
        asset: ContentAsset,
        repo_owner: str,
        repo_name: str,
        bundle: ContentBundle,
    ) -> ShipmentResult:
        """Ship to GitHub repository."""
        if not self.github or not self.github.is_available:
            return ShipmentResult(
                asset_type=asset.asset_type,
                platform="github",
                success=False,
                error="GitHub client not available or no token",
            )

        try:
            # Determine path in repo
            ext = ".md" if asset.platform in ("blog", "email") else ".txt"
            repo_path = f"content/published/{bundle.bundle_id}/{asset.asset_type}{ext}"

            result = self.github.create_or_update_file(
                owner=repo_owner,
                repo=repo_name,
                path=repo_path,
                content=asset.content,
                message=f"Publish: {asset.title}",
            )

            if result:
                return ShipmentResult(
                    asset_type=asset.asset_type,
                    platform="github",
                    success=True,
                    destination=f"https://github.com/{repo_owner}/{repo_name}/blob/main/{repo_path}",
                    metadata={"commit_sha": result.get("commit", {}).get("sha")},
                )
            else:
                return ShipmentResult(
                    asset_type=asset.asset_type,
                    platform="github",
                    success=False,
                    error="GitHub API returned no result",
                )

        except Exception as e:
            return ShipmentResult(
                asset_type=asset.asset_type,
                platform="github",
                success=False,
                error=str(e),
            )

    def _ship_to_twitter(self, asset: ContentAsset) -> ShipmentResult:
        """Ship to Twitter/X. TODO: Implement."""
        return ShipmentResult(
            asset_type=asset.asset_type,
            platform="twitter",
            success=False,
            error="Twitter shipping not implemented (Phase 2)",
        )

    def _ship_to_email(self, asset: ContentAsset) -> ShipmentResult:
        """Ship via email. TODO: Implement."""
        return ShipmentResult(
            asset_type=asset.asset_type,
            platform="email",
            success=False,
            error="Email shipping not implemented (Phase 2)",
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ShipmentResult',
    'ShippingReport',
    'PublishingDispatcher',
]

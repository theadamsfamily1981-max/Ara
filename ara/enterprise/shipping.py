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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "shipped_at": self.shipped_at.isoformat(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "results": [
                {
                    "asset_type": r.asset_type,
                    "platform": r.platform,
                    "success": r.success,
                    "destination": r.destination,
                    "error": r.error,
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
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # GitHub integration
        self._github_client = None

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
            for target in targets:
                result = self._ship_asset(
                    asset=asset,
                    target=target,
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    output_dir=output_dir,
                    bundle=bundle,
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
    ) -> ShipmentResult:
        """Ship a single asset to a target."""

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

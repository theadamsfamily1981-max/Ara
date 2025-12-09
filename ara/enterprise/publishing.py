"""
Publishing Factory
===================

Extends the Factory for content publishing workflows.

Pipeline:
1. Research → research_brief.md
2. Prototype → prototype content (thread, newsletter, blog)
3. Production → polished, platform-ready content
4. Ship → dispatch to platforms (GitHub, Twitter, email, etc.)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .factory import Factory, Project, PipelineStage

logger = logging.getLogger(__name__)


# =============================================================================
# Content Types
# =============================================================================

@dataclass
class ContentAsset:
    """A single content asset (thread, post, article, etc.)."""
    asset_type: str              # twitter_thread, newsletter, blog_post, code_snippet
    title: str
    content: str
    platform: str                # twitter, email, github, blog
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ResearchBrief:
    """Research output for a topic."""
    topic: str
    summary: str
    key_points: List[str]
    sources: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        lines = [
            f"# Research Brief: {self.topic}",
            "",
            f"_Generated: {self.generated_at.isoformat()}_",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Key Points",
            "",
        ]
        for point in self.key_points:
            lines.append(f"- {point}")

        if self.sources:
            lines.extend(["", "## Sources", ""])
            for src in self.sources:
                lines.append(f"- {src}")

        return "\n".join(lines)


@dataclass
class ContentBundle:
    """A bundle of content assets for shipping."""
    bundle_id: str
    topic: str
    research_brief: Optional[ResearchBrief] = None
    assets: List[ContentAsset] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_assets_by_platform(self, platform: str) -> List[ContentAsset]:
        return [a for a in self.assets if a.platform == platform]

    def get_asset_by_type(self, asset_type: str) -> Optional[ContentAsset]:
        for a in self.assets:
            if a.asset_type == asset_type:
                return a
        return None


# =============================================================================
# Content Synthesizer (Stub)
# =============================================================================

class ContentSynthesizer:
    """
    Generates research briefs and content from topics.

    This is a stub - real implementation will use LLM calls.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def generate_research(
        self,
        topic: str,
        output_path: Optional[str] = None,
    ) -> ResearchBrief:
        """
        Generate a research brief for a topic.

        TODO: Integrate with LLM backend.
        """
        # Stub implementation
        brief = ResearchBrief(
            topic=topic,
            summary=f"This is a research brief about {topic}. [TODO: Generate via LLM]",
            key_points=[
                f"Key point 1 about {topic}",
                f"Key point 2 about {topic}",
                f"Key point 3 about {topic}",
            ],
            sources=["[TODO: Add real sources]"],
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(brief.to_markdown())
            logger.info(f"ContentSynthesizer: wrote research brief to {output_path}")

        return brief


# =============================================================================
# Publishing Factory
# =============================================================================

class PublishingFactory(Factory):
    """
    Factory specialized for content publishing.

    Extends Factory with content-specific methods.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.synthesizer = ContentSynthesizer(config)
        self._bundles: Dict[str, ContentBundle] = {}

    # =========================================================================
    # Content Pipeline
    # =========================================================================

    def create_content_project(
        self,
        topic: str,
        priority: int = 2,
    ) -> Project:
        """Create a new content project."""
        return self.create_project(
            name=f"Content: {topic[:50]}",
            source_idea=topic,
            priority=priority,
            tags=["content", "publishing"],
        )

    def generate_research(
        self,
        project_id: str,
        output_dir: str = "artifacts/research",
    ) -> Optional[ResearchBrief]:
        """Generate research brief for a project."""
        project = self.get_project(project_id)
        if not project:
            return None

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{project_id}_research.md")

        brief = self.synthesizer.generate_research(
            topic=project.source_idea,
            output_path=output_path,
        )

        logger.info(f"PublishingFactory: generated research for {project.name}")
        return brief

    def build_prototypes(
        self,
        project_id: str,
        research_brief: ResearchBrief,
        output_dir: str = "artifacts/prototypes",
    ) -> ContentBundle:
        """
        Build prototype content assets from research.

        Creates:
        - Twitter thread
        - Newsletter draft
        - Blog post draft
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        os.makedirs(output_dir, exist_ok=True)

        bundle = ContentBundle(
            bundle_id=project_id,
            topic=project.source_idea,
            research_brief=research_brief,
        )

        # Generate Twitter thread (stub)
        thread_content = self._generate_twitter_thread(research_brief)
        thread_path = os.path.join(output_dir, f"{project_id}_twitter_thread.txt")
        with open(thread_path, 'w') as f:
            f.write(thread_content)

        bundle.assets.append(ContentAsset(
            asset_type="twitter_thread",
            title=f"Thread: {research_brief.topic}",
            content=thread_content,
            platform="twitter",
            file_path=thread_path,
        ))

        # Generate newsletter (stub)
        newsletter_content = self._generate_newsletter(research_brief)
        newsletter_path = os.path.join(output_dir, f"{project_id}_newsletter.md")
        with open(newsletter_path, 'w') as f:
            f.write(newsletter_content)

        bundle.assets.append(ContentAsset(
            asset_type="newsletter",
            title=f"Newsletter: {research_brief.topic}",
            content=newsletter_content,
            platform="email",
            file_path=newsletter_path,
        ))

        # Generate blog post (stub)
        blog_content = self._generate_blog_post(research_brief)
        blog_path = os.path.join(output_dir, f"{project_id}_blog.md")
        with open(blog_path, 'w') as f:
            f.write(blog_content)

        bundle.assets.append(ContentAsset(
            asset_type="blog_post",
            title=f"Blog: {research_brief.topic}",
            content=blog_content,
            platform="blog",
            file_path=blog_path,
        ))

        self._bundles[project_id] = bundle
        logger.info(f"PublishingFactory: built {len(bundle.assets)} prototypes for {project.name}")

        return bundle

    def productionize(
        self,
        project_id: str,
        bundle: ContentBundle,
        output_dir: str = "artifacts/production",
    ) -> ContentBundle:
        """
        Polish prototypes into production-ready content.

        - Check lengths/formats
        - Apply templates
        - Validate against platform requirements
        """
        os.makedirs(output_dir, exist_ok=True)

        for asset in bundle.assets:
            # Apply platform-specific validation
            if asset.platform == "twitter":
                # Check thread format, character limits per tweet
                pass  # TODO: Split into tweets, validate lengths

            elif asset.platform == "email":
                # Check email format
                pass  # TODO: Add email headers, preview text

            elif asset.platform == "blog":
                # Check markdown format
                pass  # TODO: Add frontmatter, SEO

            # Copy to production dir
            if asset.file_path:
                prod_path = os.path.join(
                    output_dir,
                    os.path.basename(asset.file_path)
                )
                with open(prod_path, 'w') as f:
                    f.write(asset.content)
                asset.file_path = prod_path

        logger.info(f"PublishingFactory: productionized bundle {bundle.bundle_id}")
        return bundle

    # =========================================================================
    # Content Generation (Stubs - integrate with LLM)
    # =========================================================================

    def _generate_twitter_thread(self, brief: ResearchBrief) -> str:
        """Generate Twitter thread from research. TODO: LLM integration."""
        lines = [
            f"🧵 Thread: {brief.topic}",
            "",
            "1/ " + brief.summary[:250],
            "",
        ]

        for i, point in enumerate(brief.key_points[:5], start=2):
            lines.append(f"{i}/ {point}")
            lines.append("")

        lines.append(f"{len(brief.key_points) + 2}/ That's it! Follow for more.")

        return "\n".join(lines)

    def _generate_newsletter(self, brief: ResearchBrief) -> str:
        """Generate newsletter from research. TODO: LLM integration."""
        key_points = "\n".join(f"- {p}" for p in brief.key_points)
        return f"""# {brief.topic}

{brief.summary}

## Key Takeaways

{key_points}

---

_This newsletter was generated by Ara._
"""

    def _generate_blog_post(self, brief: ResearchBrief) -> str:
        """Generate blog post from research. TODO: LLM integration."""
        details = "\n\n".join(f"### {p}\n\n[Expand on this point...]" for p in brief.key_points)
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        return f"""---
title: "{brief.topic}"
date: {date_str}
draft: true
---

# {brief.topic}

{brief.summary}

## Details

{details}

## Conclusion

[TODO: Write conclusion]

---

_Generated by Ara's Publishing Factory._
"""

    def get_bundle(self, project_id: str) -> Optional[ContentBundle]:
        """Get content bundle for a project."""
        return self._bundles.get(project_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ContentAsset',
    'ResearchBrief',
    'ContentBundle',
    'ContentSynthesizer',
    'PublishingFactory',
]

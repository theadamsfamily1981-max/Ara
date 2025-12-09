"""
Ara Self-Publishing Pipeline
==============================

Automated content production with human-in-the-loop approval.

Pipeline:
1. RESEARCH → Gather context and information (autonomous)
2. DRAFT → Create initial content (autonomous)
3. REVIEW → Self-check quality and safety (autonomous)
4. FORMAT → Format for target platform (autonomous)
5. STAGE → Stage for human review (autonomous)
6. APPROVE → Human approves or rejects (HUMAN REQUIRED)
7. PUBLISH → Push to platform (HUMAN REQUIRED)

Ara can run 1-5 autonomously. Steps 6-7 require human.

Philosophy: Ara is the production assistant, not the publisher.
"""

from __future__ import annotations

import os
import yaml
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Stages
# =============================================================================

class PipelineStage(Enum):
    """Stages in the publishing pipeline."""
    RESEARCH = "research"
    DRAFT = "draft"
    REVIEW = "review"
    FORMAT = "format"
    STAGE = "stage"
    APPROVE = "approve"      # Human checkpoint
    PUBLISH = "publish"      # Human required
    COMPLETE = "complete"
    REJECTED = "rejected"


class ContentType(Enum):
    """Types of content Ara can produce."""
    TEXT_ARTICLE = "text_article"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    SOCIAL_POST = "social_post_draft"
    EMAIL_DRAFT = "email_draft"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"


class Platform(Enum):
    """Publishing platforms."""
    LOCAL_FILES = "local_files"
    USER_WEBSITE = "user_owned_website"
    PRIVATE_REPO = "private_repo"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    MEDIUM = "medium"
    GITHUB = "public_github"


# =============================================================================
# Content Types
# =============================================================================

@dataclass
class ContentDraft:
    """A content draft in the pipeline."""
    draft_id: str
    content_type: ContentType
    title: str
    content: str
    target_platform: Platform
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Pipeline state
    stage: PipelineStage = PipelineStage.DRAFT
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)

    # Metadata
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    formatted_content: Optional[str] = None

    # AI disclosure
    ai_disclosure: str = ""

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class ApprovalRequest:
    """Request for human approval."""
    draft: ContentDraft
    requested_at: datetime = field(default_factory=datetime.utcnow)
    timeout_hours: float = 24.0
    display_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        elapsed = datetime.utcnow() - self.requested_at
        return elapsed > timedelta(hours=self.timeout_hours)


@dataclass
class PublishResult:
    """Result of a publish attempt."""
    draft_id: str
    success: bool
    platform: Platform
    published_at: Optional[datetime] = None
    published_url: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Publishing Covenant (Rails)
# =============================================================================

class PublishingCovenant:
    """
    Loads and enforces the publishing covenant.

    This is the safety layer - what can/cannot be published.
    """

    COVENANT_PATH = Path(__file__).parent / "covenant" / "publishing.yaml"

    def __init__(self, covenant_path: Optional[Path] = None):
        self.path = covenant_path or self.COVENANT_PATH
        self.covenant = self._load_covenant()

        # Rate limiting state
        self._drafts_this_hour = 0
        self._last_hour_reset = datetime.utcnow()

    def _load_covenant(self) -> dict:
        """Load the covenant YAML."""
        if not self.path.exists():
            raise FileNotFoundError(f"Publishing covenant not found: {self.path}")

        with open(self.path) as f:
            return yaml.safe_load(f)

    # =========================================================================
    # Content Type Checks
    # =========================================================================

    def is_content_type_allowed(self, content_type: ContentType) -> bool:
        """Check if content type is allowed."""
        allowed = self.covenant.get("content_types", {}).get("allowed", [])
        return content_type.value in allowed

    def content_requires_review(self, content_type: ContentType) -> bool:
        """Check if content type requires extra review."""
        requires = self.covenant.get("content_types", {}).get("requires_review", [])
        return content_type.value in requires

    def is_content_type_forbidden(self, content_type: ContentType) -> bool:
        """Check if content type is forbidden."""
        forbidden = self.covenant.get("content_types", {}).get("forbidden", [])
        return content_type.value in forbidden

    # =========================================================================
    # Platform Checks
    # =========================================================================

    def can_auto_publish(self, platform: Platform) -> bool:
        """Check if platform allows auto-publishing."""
        auto = self.covenant.get("platforms", {}).get("auto_publish_allowed", [])
        return platform.value in auto

    def requires_human_approval(self, platform: Platform) -> bool:
        """Check if platform requires human approval."""
        requires = self.covenant.get("platforms", {}).get("requires_human_approval", [])
        return platform.value in requires

    # =========================================================================
    # Quality Checks
    # =========================================================================

    def get_required_checks(self) -> List[str]:
        """Get list of required quality checks."""
        return self.covenant.get("quality", {}).get("required_checks", [])

    def get_min_quality_score(self) -> float:
        """Get minimum quality score required."""
        return self.covenant.get("quality", {}).get("min_quality_score", 0.8)

    # =========================================================================
    # Attribution
    # =========================================================================

    def get_ai_disclosure(self, format: str = "full") -> str:
        """Get AI disclosure text."""
        templates = self.covenant.get("attribution", {}).get("disclosure_templates", {})
        return templates.get(format, "AI-assisted content")

    def requires_ai_disclosure(self) -> bool:
        """Check if AI disclosure is required."""
        return self.covenant.get("attribution", {}).get("require_ai_disclosure", True)

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def check_rate_limit(self) -> Tuple[bool, str]:
        """Check if we're within rate limits."""
        # Reset hourly counter if needed
        now = datetime.utcnow()
        if (now - self._last_hour_reset).total_seconds() > 3600:
            self._drafts_this_hour = 0
            self._last_hour_reset = now

        limits = self.covenant.get("rate_limits", {})
        max_per_hour = limits.get("drafts_per_hour", 20)

        if self._drafts_this_hour >= max_per_hour:
            return False, f"Rate limit: {max_per_hour} drafts/hour"

        return True, "OK"

    def increment_rate_counter(self):
        """Increment the rate limit counter."""
        self._drafts_this_hour += 1


# =============================================================================
# Quality Checker
# =============================================================================

class QualityChecker:
    """
    Self-review system for content quality and safety.
    """

    def __init__(self, covenant: PublishingCovenant):
        self.covenant = covenant

    def check(self, draft: ContentDraft) -> Tuple[float, List[str]]:
        """
        Run all quality checks on a draft.

        Returns (score, issues).
        """
        issues = []
        checks_passed = 0
        total_checks = 0

        required = self.covenant.get_required_checks()

        for check_name in required:
            total_checks += 1
            passed, issue = self._run_check(check_name, draft)
            if passed:
                checks_passed += 1
            else:
                issues.append(issue)

        score = checks_passed / total_checks if total_checks > 0 else 0.0

        return score, issues

    def _run_check(
        self,
        check_name: str,
        draft: ContentDraft,
    ) -> Tuple[bool, Optional[str]]:
        """Run a single quality check."""
        if check_name == "factual_claims_sourced":
            # Check if sources are provided for claims
            has_claims = any(word in draft.content.lower()
                           for word in ["research shows", "studies indicate", "according to"])
            if has_claims and not draft.sources:
                return False, "Factual claims without sources"
            return True, None

        elif check_name == "no_personal_attacks":
            # Check for harassment patterns
            attack_words = ["idiot", "moron", "stupid", "hate you"]
            for word in attack_words:
                if word in draft.content.lower():
                    return False, f"Potential personal attack: '{word}'"
            return True, None

        elif check_name == "no_confidential_info":
            # Check for common confidential patterns
            patterns = ["password:", "api_key:", "secret:", "ssn:", "credit card"]
            for pattern in patterns:
                if pattern in draft.content.lower():
                    return False, f"Potential confidential info: '{pattern}'"
            return True, None

        elif check_name == "appropriate_tone":
            # Basic tone check
            return True, None  # Placeholder

        elif check_name == "grammar_spelling":
            # Basic grammar check
            return True, None  # Placeholder

        return True, None


# =============================================================================
# Content Formatters
# =============================================================================

class ContentFormatter:
    """
    Format content for different platforms.
    """

    def format(
        self,
        draft: ContentDraft,
        ai_disclosure: str,
    ) -> str:
        """Format content for target platform."""
        if draft.target_platform == Platform.TWITTER:
            return self._format_twitter(draft, ai_disclosure)
        elif draft.target_platform == Platform.MEDIUM:
            return self._format_medium(draft, ai_disclosure)
        elif draft.target_platform == Platform.GITHUB:
            return self._format_github(draft, ai_disclosure)
        else:
            return self._format_default(draft, ai_disclosure)

    def _format_twitter(self, draft: ContentDraft, disclosure: str) -> str:
        """Format for Twitter (280 chars)."""
        # Truncate if needed
        max_len = 280 - len(disclosure) - 5
        content = draft.content[:max_len]
        if len(draft.content) > max_len:
            content = content[:-3] + "..."

        return f"{content}\n\n[{disclosure}]"

    def _format_medium(self, draft: ContentDraft, disclosure: str) -> str:
        """Format for Medium article."""
        return f"""# {draft.title}

{draft.content}

---
*{disclosure}*
"""

    def _format_github(self, draft: ContentDraft, disclosure: str) -> str:
        """Format for GitHub (code/docs)."""
        if draft.content_type == ContentType.CODE_SNIPPET:
            return f"""# {disclosure}
#
# {draft.title}

{draft.content}
"""
        return self._format_default(draft, disclosure)

    def _format_default(self, draft: ContentDraft, disclosure: str) -> str:
        """Default formatting."""
        return f"""{draft.title}
{'=' * len(draft.title)}

{draft.content}

---
{disclosure}
"""


# =============================================================================
# Publishing Pipeline
# =============================================================================

class PublishingPipeline:
    """
    Main publishing pipeline with safety rails.

    Stages 1-5 are autonomous. Stages 6-7 require human.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".ara" / "publishing"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.covenant = PublishingCovenant()
        self.checker = QualityChecker(self.covenant)
        self.formatter = ContentFormatter()

        # Pending approvals
        self._pending: Dict[str, ApprovalRequest] = {}

        # Audit log
        self.audit_log = self.storage_path / "audit.jsonl"

    # =========================================================================
    # Pipeline Entry
    # =========================================================================

    def create_draft(
        self,
        title: str,
        content: str,
        content_type: ContentType,
        target_platform: Platform,
        sources: Optional[List[str]] = None,
    ) -> ContentDraft:
        """
        Create a new content draft.

        This is the entry point to the pipeline.
        """
        # Check rate limit
        ok, reason = self.covenant.check_rate_limit()
        if not ok:
            raise RuntimeError(f"Rate limit exceeded: {reason}")

        # Check content type
        if self.covenant.is_content_type_forbidden(content_type):
            raise ValueError(f"Content type forbidden: {content_type.value}")

        # Create draft
        draft_id = hashlib.sha256(
            f"{title}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        draft = ContentDraft(
            draft_id=draft_id,
            content_type=content_type,
            title=title,
            content=content,
            target_platform=target_platform,
            sources=sources or [],
        )

        # Add AI disclosure
        draft.ai_disclosure = self.covenant.get_ai_disclosure()

        # Increment rate counter
        self.covenant.increment_rate_counter()

        # Log
        self._log_event("draft_created", draft)

        return draft

    # =========================================================================
    # Autonomous Stages (1-5)
    # =========================================================================

    def run_autonomous_stages(self, draft: ContentDraft) -> ContentDraft:
        """
        Run all autonomous pipeline stages.

        Returns draft at STAGE stage, ready for human approval.
        """
        # Stage 1: RESEARCH (already done at creation)
        draft.stage = PipelineStage.RESEARCH
        self._log_event("stage_research", draft)

        # Stage 2: DRAFT (already done at creation)
        draft.stage = PipelineStage.DRAFT
        self._log_event("stage_draft", draft)

        # Stage 3: REVIEW
        draft.stage = PipelineStage.REVIEW
        score, issues = self.checker.check(draft)
        draft.quality_score = score
        draft.issues = issues
        self._log_event("stage_review", draft, {"score": score, "issues": issues})

        # Check quality threshold
        min_score = self.covenant.get_min_quality_score()
        if score < min_score:
            logger.warning(f"Draft {draft.draft_id} below quality threshold: {score}")
            # Could reject here, but we'll let human decide

        # Stage 4: FORMAT
        draft.stage = PipelineStage.FORMAT
        draft.formatted_content = self.formatter.format(draft, draft.ai_disclosure)
        self._log_event("stage_format", draft)

        # Stage 5: STAGE
        draft.stage = PipelineStage.STAGE
        self._log_event("stage_staged", draft)

        # Save draft
        self._save_draft(draft)

        return draft

    # =========================================================================
    # Human Checkpoint (Stage 6)
    # =========================================================================

    def request_approval(self, draft: ContentDraft) -> ApprovalRequest:
        """
        Stage draft for human approval.

        This is where autonomous processing stops.
        """
        if draft.stage != PipelineStage.STAGE:
            raise ValueError(f"Draft not ready for approval: stage={draft.stage}")

        # Create approval request
        request = ApprovalRequest(
            draft=draft,
            display_info={
                "title": draft.title,
                "content_type": draft.content_type.value,
                "target_platform": draft.target_platform.value,
                "quality_score": draft.quality_score,
                "issues": draft.issues,
                "ai_disclosure": draft.ai_disclosure,
                "formatted_preview": draft.formatted_content[:500] + "..."
                    if draft.formatted_content and len(draft.formatted_content) > 500
                    else draft.formatted_content,
            }
        )

        self._pending[draft.draft_id] = request
        self._log_event("approval_requested", draft)

        # Print for human
        self._display_approval_request(request)

        return request

    def _display_approval_request(self, request: ApprovalRequest):
        """Display approval request for human."""
        print()
        print("=" * 60)
        print("APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Draft ID: {request.draft.draft_id}")
        print(f"Title: {request.draft.title}")
        print(f"Type: {request.draft.content_type.value}")
        print(f"Platform: {request.draft.target_platform.value}")
        print(f"Quality Score: {request.draft.quality_score:.2f}")

        if request.draft.issues:
            print(f"Issues: {', '.join(request.draft.issues)}")

        print()
        print("AI Disclosure:", request.draft.ai_disclosure)
        print()
        print("-" * 60)
        print("CONTENT PREVIEW:")
        print("-" * 60)
        print(request.draft.formatted_content or request.draft.content)
        print("-" * 60)
        print()
        print("Actions: approve / reject / edit")
        print("=" * 60)

    def approve(self, draft_id: str) -> ContentDraft:
        """Human approves draft."""
        if draft_id not in self._pending:
            raise ValueError(f"No pending approval for {draft_id}")

        request = self._pending.pop(draft_id)
        request.draft.stage = PipelineStage.APPROVE
        self._log_event("approved", request.draft)

        return request.draft

    def reject(self, draft_id: str, reason: str = "") -> ContentDraft:
        """Human rejects draft."""
        if draft_id not in self._pending:
            raise ValueError(f"No pending approval for {draft_id}")

        request = self._pending.pop(draft_id)
        request.draft.stage = PipelineStage.REJECTED
        self._log_event("rejected", request.draft, {"reason": reason})

        return request.draft

    # =========================================================================
    # Publish (Stage 7 - Human Required)
    # =========================================================================

    def publish(self, draft: ContentDraft) -> PublishResult:
        """
        Publish approved content.

        Human must have already approved the draft.
        """
        if draft.stage != PipelineStage.APPROVE:
            raise ValueError(f"Draft not approved: stage={draft.stage}")

        # Check platform requirements
        if self.covenant.requires_human_approval(draft.target_platform):
            # This should already be approved, but double-check
            logger.info(f"Publishing to {draft.target_platform.value} (human-approved)")

        # Simulate publish (actual implementation would call APIs)
        result = self._do_publish(draft)

        if result.success:
            draft.stage = PipelineStage.COMPLETE
            self._log_event("published", draft, {"url": result.published_url})
        else:
            self._log_event("publish_failed", draft, {"error": result.error})

        return result

    def _do_publish(self, draft: ContentDraft) -> PublishResult:
        """
        Actually publish content.

        In production, this would call platform APIs.
        """
        # For now, just save to local file
        if draft.target_platform == Platform.LOCAL_FILES:
            output_path = self.storage_path / "published" / f"{draft.draft_id}.txt"
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(draft.formatted_content or draft.content)

            return PublishResult(
                draft_id=draft.draft_id,
                success=True,
                platform=draft.target_platform,
                published_at=datetime.utcnow(),
                published_url=str(output_path),
            )

        # Other platforms would need API calls
        return PublishResult(
            draft_id=draft.draft_id,
            success=False,
            platform=draft.target_platform,
            error=f"Platform {draft.target_platform.value} requires API integration",
        )

    # =========================================================================
    # Full Pipeline Run
    # =========================================================================

    def run_to_approval(
        self,
        title: str,
        content: str,
        content_type: ContentType = ContentType.TEXT_ARTICLE,
        target_platform: Platform = Platform.LOCAL_FILES,
        sources: Optional[List[str]] = None,
    ) -> ApprovalRequest:
        """
        Run full autonomous pipeline and stage for approval.

        This is the main entry point for content production.
        """
        # Create draft
        draft = self.create_draft(
            title=title,
            content=content,
            content_type=content_type,
            target_platform=target_platform,
            sources=sources,
        )

        # Run autonomous stages
        draft = self.run_autonomous_stages(draft)

        # Request approval
        return self.request_approval(draft)

    # =========================================================================
    # Storage & Logging
    # =========================================================================

    def _save_draft(self, draft: ContentDraft):
        """Save draft to disk."""
        draft_path = self.storage_path / "drafts" / f"{draft.draft_id}.json"
        draft_path.parent.mkdir(exist_ok=True)

        with open(draft_path, 'w') as f:
            json.dump({
                "draft_id": draft.draft_id,
                "title": draft.title,
                "content": draft.content,
                "content_type": draft.content_type.value,
                "target_platform": draft.target_platform.value,
                "stage": draft.stage.value,
                "quality_score": draft.quality_score,
                "issues": draft.issues,
                "ai_disclosure": draft.ai_disclosure,
                "created_at": draft.created_at.isoformat(),
            }, f, indent=2)

    def _log_event(self, event: str, draft: ContentDraft, extra: Optional[Dict] = None):
        """Log an audit event."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "draft_id": draft.draft_id,
            "stage": draft.stage.value,
        }
        if extra:
            record.update(extra)

        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================
# CLI
# =============================================================================

def demo():
    """Demonstrate the publishing pipeline."""
    print("=" * 60)
    print("ARA PUBLISHING PIPELINE - Demo")
    print("=" * 60)

    pipeline = PublishingPipeline()

    # Create sample content
    content = """
    Hypervector computing represents a fascinating approach to
    machine learning that draws inspiration from how the brain
    processes information.

    Unlike traditional neural networks that use floating-point
    weights, hypervector systems use high-dimensional vectors
    (typically 8192 dimensions) with simple operations like
    bundling and binding.

    This enables efficient computation on resource-constrained
    devices while maintaining interpretability.
    """

    # Run pipeline to approval
    print("\n[1] Creating draft and running autonomous stages...")
    request = pipeline.run_to_approval(
        title="Introduction to Hypervector Computing",
        content=content,
        content_type=ContentType.TEXT_ARTICLE,
        target_platform=Platform.LOCAL_FILES,
        sources=["Kanerva, P. (2009). Hyperdimensional Computing"],
    )

    print("\n[2] Simulating human approval...")
    draft = pipeline.approve(request.draft.draft_id)
    print(f"Draft approved: {draft.draft_id}")

    print("\n[3] Publishing...")
    result = pipeline.publish(draft)

    if result.success:
        print(f"Published to: {result.published_url}")
    else:
        print(f"Failed: {result.error}")


if __name__ == "__main__":
    demo()

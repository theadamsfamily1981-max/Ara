"""
Ara Publishing Pipeline
========================

Automated content production with human-in-the-loop approval.

Pipeline Stages:
1. RESEARCH → Gather context (autonomous)
2. DRAFT → Create content (autonomous)
3. REVIEW → Quality check (autonomous)
4. FORMAT → Platform formatting (autonomous)
5. STAGE → Ready for review (autonomous)
6. APPROVE → Human approval (HUMAN REQUIRED)
7. PUBLISH → Push to platform (HUMAN REQUIRED)

Usage:
    from ara.publishing import PublishingPipeline, ContentType, Platform

    pipeline = PublishingPipeline()

    # Run to approval stage
    request = pipeline.run_to_approval(
        title="My Article",
        content="Article content...",
        content_type=ContentType.TEXT_ARTICLE,
        target_platform=Platform.LOCAL_FILES,
    )

    # Human reviews and approves
    draft = pipeline.approve(request.draft.draft_id)

    # Publish
    result = pipeline.publish(draft)

IMPORTANT: Ara can run stages 1-5 autonomously.
Human approval is REQUIRED for stages 6-7.
"""

from .pipeline import (
    # Pipeline
    PublishingPipeline,
    PublishingCovenant,
    QualityChecker,
    ContentFormatter,

    # Enums
    PipelineStage,
    ContentType,
    Platform,

    # Types
    ContentDraft,
    ApprovalRequest,
    PublishResult,
)


__all__ = [
    # Pipeline
    'PublishingPipeline',
    'PublishingCovenant',
    'QualityChecker',
    'ContentFormatter',

    # Enums
    'PipelineStage',
    'ContentType',
    'Platform',

    # Types
    'ContentDraft',
    'ApprovalRequest',
    'PublishResult',
]

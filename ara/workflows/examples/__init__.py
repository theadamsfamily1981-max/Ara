# ara/workflows/examples/__init__.py
"""
Example Self-Guided Workflows
=============================

These examples show how Ara guides users through workflows
while the engine handles the actual execution.

Examples:
    - onboarding: New user onboarding flow
    - grant_application: Grant application workflow
    - project_setup: New project initialization
"""

from .onboarding import (
    create_onboarding_workflow,
    OnboardingActivities,
)

__all__ = [
    "create_onboarding_workflow",
    "OnboardingActivities",
]

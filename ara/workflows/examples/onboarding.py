# ara/workflows/examples/onboarding.py
"""
Onboarding Workflow Example
===========================

A self-guided workflow that onboards new users.
Demonstrates Ara's three roles:

1. Ara-as-Companion: Explains each step to the user
2. Ara-as-Director: Decides what to do based on user responses
3. Ara-as-Historian: Records the experience for learning

The workflow:
    1. Welcome → Collect user goals
    2. Assessment → Understand current state
    3. Configuration → Set up based on goals
    4. Tutorial → Guide through first use
    5. Complete → Celebrate and suggest next steps

Usage:
    from ara.workflows.examples.onboarding import create_onboarding_workflow
    from ara.workflows import AraSelfGuidedOrchestrator
    from ara.workflows.adapters.temporal import TemporalAdapter

    # Create and register workflow
    adapter = TemporalAdapter()
    workflow = create_onboarding_workflow()
    adapter.register_workflow(workflow)

    # Run with Ara guiding
    orchestrator = AraSelfGuidedOrchestrator(engine=adapter)
    result = await orchestrator.run_workflow(
        workflow_id="onboarding",
        initial_state={"user_id": "123"},
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..adapters.base import (
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from ..adapters.temporal import activity

log = logging.getLogger("Ara.Workflows.Examples.Onboarding")


# =============================================================================
# Workflow Definition
# =============================================================================

def create_onboarding_workflow() -> WorkflowDefinition:
    """
    Create the onboarding workflow definition.

    This defines WHAT steps exist.
    Ara decides WHICH to run and HOW to explain them.
    """
    return WorkflowDefinition(
        id="onboarding",
        name="New User Onboarding",
        description="Guide new users through their first Ara experience",
        version="1.0.0",
        steps=[
            # Step 1: Welcome and collect goals
            StepDefinition(
                id="welcome",
                name="Welcome User",
                type=StepType.HUMAN,
                description="Greet the user and understand their goals",
                prompt_template="""
Welcome! I'm Ara, your AI companion.

I'd like to understand what brings you here today.
What are you hoping to accomplish?

Pick one or tell me in your own words:
""",
                branches=[
                    "Build an app",
                    "Learn AI/ML",
                    "Automate tasks",
                    "Explore capabilities",
                    "Other",
                ],
            ),

            # Step 2: Assess current state
            StepDefinition(
                id="assess_experience",
                name="Assess Experience Level",
                type=StepType.HUMAN,
                description="Understand user's technical background",
                requires=["welcome"],
                prompt_template="""
Great! Now I'd like to understand your background so I can
tailor the experience for you.

How would you describe your technical experience?
""",
                branches=[
                    "Beginner - New to programming",
                    "Intermediate - Some coding experience",
                    "Advanced - Professional developer",
                    "Expert - Deep AI/ML experience",
                ],
            ),

            # Step 3: Profile creation
            StepDefinition(
                id="create_profile",
                name="Create User Profile",
                type=StepType.ACTIVITY,
                description="Set up the user profile based on responses",
                requires=["welcome", "assess_experience"],
                meta={"activity": "create_user_profile"},
            ),

            # Step 4: Choose path (decision)
            StepDefinition(
                id="choose_path",
                name="Choose Learning Path",
                type=StepType.DECISION,
                description="Select the appropriate onboarding path",
                requires=["create_profile"],
                branches=["beginner_path", "advanced_path"],
                condition="state.get('experience_level', 'beginner') in ['Beginner', 'Intermediate']",
            ),

            # Step 5a: Beginner tutorial
            StepDefinition(
                id="beginner_tutorial",
                name="Beginner Tutorial",
                type=StepType.ACTIVITY,
                description="Walk through basics for new users",
                requires=["choose_path"],
                meta={"activity": "run_beginner_tutorial"},
            ),

            # Step 5b: Advanced overview
            StepDefinition(
                id="advanced_overview",
                name="Advanced Overview",
                type=StepType.ACTIVITY,
                description="Quick overview for experienced users",
                requires=["choose_path"],
                meta={"activity": "run_advanced_overview"},
            ),

            # Step 6: First project suggestion
            StepDefinition(
                id="suggest_project",
                name="Suggest First Project",
                type=StepType.ACTIVITY,
                description="Recommend a starter project based on goals",
                requires=["beginner_tutorial", "advanced_overview"],
                meta={"activity": "suggest_first_project"},
            ),

            # Step 7: Confirm next steps
            StepDefinition(
                id="confirm_next",
                name="Confirm Next Steps",
                type=StepType.HUMAN,
                description="Confirm the user is ready to proceed",
                requires=["suggest_project"],
                prompt_template="""
I've suggested some things to try.

Would you like to:
""",
                branches=[
                    "Start the suggested project",
                    "Explore on my own first",
                    "Ask a question",
                    "Take a break and come back later",
                ],
            ),

            # Step 8: Complete onboarding
            StepDefinition(
                id="complete",
                name="Complete Onboarding",
                type=StepType.ACTIVITY,
                description="Mark onboarding complete and record preferences",
                requires=["confirm_next"],
                meta={"activity": "complete_onboarding"},
            ),
        ],
        entry_step="welcome",
        exit_steps=["complete"],
        meta={
            "category": "onboarding",
            "estimated_duration_minutes": 10,
        },
    )


# =============================================================================
# Activities (the actual work)
# =============================================================================

class OnboardingActivities:
    """
    Activities that perform the actual onboarding work.

    These are the "muscles" - Ara decides when to invoke them.
    In production, these would be Temporal activities.
    """

    @staticmethod
    @activity(
        name="create_user_profile",
        timeout_seconds=30,
        description="Create or update user profile from onboarding responses",
    )
    async def create_user_profile(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create user profile from collected information.

        In production: saves to database, initializes preferences.
        """
        user_id = state.get("user_id", "unknown")
        goal = state.get("welcome_response", "explore")
        experience = state.get("assess_experience_response", "beginner")

        # Map experience to level
        experience_map = {
            "Beginner - New to programming": "beginner",
            "Intermediate - Some coding experience": "intermediate",
            "Advanced - Professional developer": "advanced",
            "Expert - Deep AI/ML experience": "expert",
        }
        experience_level = experience_map.get(experience, "beginner")

        # Map goal to category
        goal_map = {
            "Build an app": "builder",
            "Learn AI/ML": "learner",
            "Automate tasks": "automator",
            "Explore capabilities": "explorer",
            "Other": "custom",
        }
        goal_category = goal_map.get(goal, "explorer")

        profile = {
            "user_id": user_id,
            "goal": goal,
            "goal_category": goal_category,
            "experience": experience,
            "experience_level": experience_level,
            "created_at": datetime.utcnow().isoformat(),
            "onboarding_status": "in_progress",
        }

        log.info("Created profile for user %s: goal=%s, level=%s",
                 user_id, goal_category, experience_level)

        return {
            "profile": profile,
            "experience_level": experience_level,
            "goal_category": goal_category,
        }

    @staticmethod
    @activity(
        name="run_beginner_tutorial",
        timeout_seconds=60,
        description="Run the beginner tutorial flow",
    )
    async def run_beginner_tutorial(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute beginner tutorial.

        In production: would launch interactive tutorial UI.
        """
        log.info("Running beginner tutorial")

        # Simulate tutorial steps
        tutorial_steps = [
            "Introduction to Ara",
            "Understanding the interface",
            "Your first command",
            "Getting help",
        ]

        return {
            "tutorial_completed": True,
            "tutorial_path": "beginner",
            "steps_shown": tutorial_steps,
        }

    @staticmethod
    @activity(
        name="run_advanced_overview",
        timeout_seconds=30,
        description="Show quick overview for advanced users",
    )
    async def run_advanced_overview(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute advanced overview.

        Shorter path for experienced users.
        """
        log.info("Running advanced overview")

        overview_topics = [
            "Architecture overview",
            "Extension points",
            "Advanced features",
        ]

        return {
            "tutorial_completed": True,
            "tutorial_path": "advanced",
            "steps_shown": overview_topics,
        }

    @staticmethod
    @activity(
        name="suggest_first_project",
        timeout_seconds=30,
        description="Suggest a starter project based on user goals",
    )
    async def suggest_first_project(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate project suggestion based on profile.
        """
        goal_category = state.get("goal_category", "explorer")
        experience_level = state.get("experience_level", "beginner")

        # Project suggestions by goal
        projects = {
            "builder": {
                "beginner": {
                    "name": "Hello World App",
                    "description": "Build a simple app that responds to your voice",
                    "estimated_time": "30 minutes",
                },
                "intermediate": {
                    "name": "Personal Assistant Bot",
                    "description": "Create a bot that manages your calendar",
                    "estimated_time": "2 hours",
                },
                "advanced": {
                    "name": "Custom Skill Integration",
                    "description": "Build a skill that integrates with your workflow",
                    "estimated_time": "4 hours",
                },
                "expert": {
                    "name": "Forge Blueprint",
                    "description": "Design and implement a new app blueprint for The Forge",
                    "estimated_time": "1 day",
                },
            },
            "learner": {
                "beginner": {
                    "name": "AI Basics Tutorial",
                    "description": "Learn how AI systems work through hands-on exercises",
                    "estimated_time": "1 hour",
                },
                "intermediate": {
                    "name": "HDC Exploration",
                    "description": "Explore hyperdimensional computing concepts",
                    "estimated_time": "2 hours",
                },
                "advanced": {
                    "name": "Custom Codec Development",
                    "description": "Create a new codec for the HDC system",
                    "estimated_time": "1 day",
                },
                "expert": {
                    "name": "Contribute to Academy",
                    "description": "Help build new learning materials",
                    "estimated_time": "Ongoing",
                },
            },
            "automator": {
                "beginner": {
                    "name": "Daily Briefing Automation",
                    "description": "Set up automatic morning briefings",
                    "estimated_time": "30 minutes",
                },
                "intermediate": {
                    "name": "Workflow Builder",
                    "description": "Create a custom automation workflow",
                    "estimated_time": "1 hour",
                },
                "advanced": {
                    "name": "Self-Guided Workflow",
                    "description": "Build an Ara-guided workflow for your team",
                    "estimated_time": "4 hours",
                },
                "expert": {
                    "name": "Custom Engine Adapter",
                    "description": "Connect Ara to your existing workflow engine",
                    "estimated_time": "1 day",
                },
            },
            "explorer": {
                "beginner": {
                    "name": "Guided Tour",
                    "description": "Take a tour of Ara's capabilities",
                    "estimated_time": "20 minutes",
                },
                "intermediate": {
                    "name": "Feature Deep Dive",
                    "description": "Explore a feature area in depth",
                    "estimated_time": "1 hour",
                },
                "advanced": {
                    "name": "Architecture Study",
                    "description": "Understand how Ara works under the hood",
                    "estimated_time": "2 hours",
                },
                "expert": {
                    "name": "Contribution Guide",
                    "description": "Learn how to contribute to Ara",
                    "estimated_time": "Variable",
                },
            },
        }

        # Get suggestion
        goal_projects = projects.get(goal_category, projects["explorer"])
        suggestion = goal_projects.get(experience_level, goal_projects["beginner"])

        log.info("Suggested project: %s", suggestion["name"])

        return {
            "suggested_project": suggestion,
            "alternative_projects": [
                p for level, p in goal_projects.items()
                if level != experience_level
            ][:2],
        }

    @staticmethod
    @activity(
        name="complete_onboarding",
        timeout_seconds=30,
        description="Mark onboarding complete and save preferences",
    )
    async def complete_onboarding(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize onboarding.

        In production: updates database, triggers follow-up actions.
        """
        user_id = state.get("user_id", "unknown")
        next_action = state.get("confirm_next_response", "explore")

        # Record completion
        completion = {
            "user_id": user_id,
            "completed_at": datetime.utcnow().isoformat(),
            "next_action": next_action,
            "tutorial_path": state.get("tutorial_path"),
            "suggested_project": state.get("suggested_project", {}).get("name"),
        }

        log.info("Onboarding complete for user %s", user_id)

        return {
            "onboarding_complete": True,
            "completion_record": completion,
        }


# =============================================================================
# Demo Runner
# =============================================================================

async def demo_onboarding():
    """
    Demonstrate the onboarding workflow with Ara guiding.

    Run with: python -m ara.workflows.examples.onboarding
    """
    from ..adapters.temporal import TemporalAdapter
    from ..orchestrator import AraSelfGuidedOrchestrator

    print("\n" + "=" * 60)
    print("Ara Self-Guided Onboarding Demo")
    print("=" * 60 + "\n")

    # Set up adapter and register workflow
    adapter = TemporalAdapter()
    workflow = create_onboarding_workflow()
    adapter.register_workflow(workflow)

    # Create orchestrator
    orchestrator = AraSelfGuidedOrchestrator(engine=adapter)

    # Register guidance callback
    def on_guidance(msg):
        print(f"\n[Ara]: {msg.message}")
        if msg.options:
            for i, opt in enumerate(msg.options, 1):
                print(f"  {i}. {opt}")

    orchestrator.on_guidance(on_guidance)

    # Simulate user input
    responses = iter([
        "Build an app",
        "Intermediate - Some coding experience",
        "Start the suggested project",
    ])

    async def mock_user_input(prompt: str) -> str:
        try:
            response = next(responses)
            print(f"\n[User]: {response}")
            return response
        except StopIteration:
            return "Continue"

    # Run workflow
    result = await orchestrator.run_workflow(
        workflow_id="onboarding",
        initial_state={"user_id": "demo_user"},
        user_input_handler=mock_user_input,
    )

    # Show result
    print("\n" + "=" * 60)
    print("Onboarding Result")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Steps Completed: {result.steps_completed}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"User Interactions: {result.user_interactions}")
    if result.output.get("suggested_project"):
        print(f"Suggested Project: {result.output['suggested_project']['name']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_onboarding())

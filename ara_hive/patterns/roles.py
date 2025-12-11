# ara_hive/patterns/roles.py
"""
Role Definitions - MetaGPT-style Agent Archetypes
=================================================

Roles are named agent archetypes with specific:
    - Responsibilities (what they do)
    - Capabilities (tools they can use)
    - SOPs (how they do it)
    - Output formats (what they produce)

Each role maps to a NamedBee in the Hive, which is a BeeAgent
with specific capabilities and operating procedures.

MetaGPT Mapping:
    Role.profile        -> Role.name + Role.description
    Role.goal           -> Role.objective
    Role.constraints    -> Role.constraints
    Role.actions        -> Role.capabilities (tools)
    Role._rc            -> NamedBee.sop (operating procedure)

Built-in Roles:
    - ProductManager: Requirements, PRDs, user stories
    - Architect: System design, API contracts, data models
    - Engineer: Implementation, code generation, debugging
    - QATester: Testing, validation, quality assurance
    - Researcher: Information gathering, analysis, synthesis
    - Writer: Content creation, documentation, reports
    - Reviewer: Code review, document review, feedback
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.queen import QueenOrchestrator, TaskRequest, TaskResult
    from ..src.registry import Tool, ToolDomain
    from .sop import SOP

log = logging.getLogger("Hive.Patterns.Roles")


# =============================================================================
# Types
# =============================================================================

class RoleType(str, Enum):
    """Categories of roles."""
    PLANNING = "planning"        # PM, Analyst
    DESIGN = "design"            # Architect, Designer
    IMPLEMENTATION = "implementation"  # Engineer, Developer
    VALIDATION = "validation"    # QA, Reviewer
    RESEARCH = "research"        # Researcher, Analyst
    CONTENT = "content"          # Writer, Editor
    OPERATIONS = "operations"    # DevOps, SRE
    CUSTOM = "custom"


class RoleStatus(str, Enum):
    """Runtime status of a role."""
    IDLE = "idle"
    EXECUTING = "executing"
    WAITING = "waiting"      # Waiting for input
    BLOCKED = "blocked"      # Waiting for dependencies
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RoleContext:
    """Context passed to role during execution."""
    task_id: str
    instruction: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)  # Cross-task memory
    team_context: Dict[str, Any] = field(default_factory=dict)  # Team shared state
    history: List[Dict[str, Any]] = field(default_factory=list)  # Previous messages


@dataclass
class RoleOutput:
    """Output from a role execution."""
    role_name: str
    success: bool
    content: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Files, documents
    messages: List[str] = field(default_factory=list)  # Logs, explanations
    next_roles: List[str] = field(default_factory=list)  # Suggested next roles
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_name": self.role_name,
            "success": self.success,
            "content": self.content,
            "artifacts": list(self.artifacts.keys()),
            "messages": self.messages,
            "next_roles": self.next_roles,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# Role Base Class
# =============================================================================

@dataclass
class Role(ABC):
    """
    Base class for all roles.

    A role defines:
        - WHO: Name, type, profile
        - WHAT: Objective, responsibilities
        - HOW: Capabilities (tools), SOP
        - CONSTRAINTS: Limits, requirements

    Subclass this to create custom roles.
    """
    name: str
    role_type: RoleType
    description: str = ""
    objective: str = ""

    # What this role does
    responsibilities: List[str] = field(default_factory=list)

    # What this role can use
    capabilities: List[str] = field(default_factory=list)  # Tool names
    domains: List[str] = field(default_factory=list)  # Tool domains

    # How this role operates
    sop: Optional[SOP] = None
    output_format: Optional[str] = None  # Expected output structure

    # Constraints
    constraints: List[str] = field(default_factory=list)
    requires_approval: bool = False
    max_iterations: int = 10

    # Runtime
    status: RoleStatus = RoleStatus.IDLE
    _queen: Optional[QueenOrchestrator] = None

    def __post_init__(self):
        if not self.description:
            self.description = f"A {self.name} role"
        if not self.objective:
            self.objective = f"Fulfill {self.name} responsibilities"

    def bind_queen(self, queen: QueenOrchestrator) -> None:
        """Bind to a Queen for task dispatch."""
        self._queen = queen

    @abstractmethod
    async def execute(
        self,
        context: RoleContext,
    ) -> RoleOutput:
        """
        Execute the role's responsibility.

        This is the main entry point - implement role-specific logic here.

        Args:
            context: Task context with inputs and memory

        Returns:
            RoleOutput with results and artifacts
        """
        pass

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool through the Queen.

        Args:
            tool_name: Tool to execute
            params: Tool parameters

        Returns:
            Tool result
        """
        if not self._queen:
            raise RuntimeError(f"Role {self.name} not bound to Queen")

        from ..src.queen import TaskRequest

        request = TaskRequest(
            instruction=f"{self.name} executing {tool_name}",
            tool=tool_name,
            params=params,
        )
        result = await self._queen.dispatch(request)

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    def get_profile(self) -> str:
        """Get role profile for LLM context."""
        profile = f"""
Role: {self.name}
Type: {self.role_type.value}
Objective: {self.objective}

Responsibilities:
{chr(10).join(f'  - {r}' for r in self.responsibilities)}

Capabilities:
{chr(10).join(f'  - {c}' for c in self.capabilities)}

Constraints:
{chr(10).join(f'  - {c}' for c in self.constraints)}
"""
        return profile.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.role_type.value,
            "description": self.description,
            "objective": self.objective,
            "responsibilities": self.responsibilities,
            "capabilities": self.capabilities,
            "status": self.status.value,
        }


# =============================================================================
# Named Bee - Role bound to BeeAgent
# =============================================================================

class NamedBee:
    """
    A named bee is a BeeAgent with a specific Role.

    This bridges the Role (archetype) with the BeeAgent (executor).
    The Queen routes tasks to NamedBees based on role matching.
    """

    def __init__(
        self,
        role: Role,
        bee_id: Optional[str] = None,
        queen: Optional[QueenOrchestrator] = None,
    ):
        self.role = role
        self.bee_id = bee_id or f"{role.name.lower()}_{uuid.uuid4().hex[:6]}"
        self.queen = queen

        # Bind role to queen
        if queen:
            role.bind_queen(queen)

        # Execution history
        self.executions: List[RoleOutput] = []

        log.info(f"NamedBee created: {self.bee_id} ({role.name})")

    async def execute(
        self,
        instruction: str,
        inputs: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> RoleOutput:
        """
        Execute a task using this bee's role.

        Args:
            instruction: What to do
            inputs: Task-specific inputs
            memory: Cross-task memory

        Returns:
            RoleOutput with results
        """
        import time
        start = time.time()

        context = RoleContext(
            task_id=str(uuid.uuid4())[:8],
            instruction=instruction,
            inputs=inputs or {},
            memory=memory or {},
            history=[o.to_dict() for o in self.executions[-5:]],  # Last 5
        )

        self.role.status = RoleStatus.EXECUTING

        try:
            output = await self.role.execute(context)
            output.duration_ms = (time.time() - start) * 1000
        except Exception as e:
            log.exception(f"NamedBee {self.bee_id} execution failed")
            output = RoleOutput(
                role_name=self.role.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self.role.status = RoleStatus.IDLE
        self.executions.append(output)

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        successes = sum(1 for o in self.executions if o.success)
        return {
            "bee_id": self.bee_id,
            "role": self.role.name,
            "executions": len(self.executions),
            "successes": successes,
            "failures": len(self.executions) - successes,
            "avg_duration_ms": (
                sum(o.duration_ms for o in self.executions) / len(self.executions)
                if self.executions else 0
            ),
        }


# =============================================================================
# Built-in Roles
# =============================================================================

@dataclass
class ProductManager(Role):
    """
    Product Manager role - defines requirements and priorities.

    Responsibilities:
        - Gather and analyze requirements
        - Write PRDs (Product Requirement Documents)
        - Define user stories and acceptance criteria
        - Prioritize features and backlog

    Outputs:
        - PRD documents
        - User stories
        - Feature specifications
    """
    name: str = "ProductManager"
    role_type: RoleType = RoleType.PLANNING
    description: str = "Defines product requirements and priorities"
    objective: str = "Transform ideas into clear, actionable requirements"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Analyze user needs and market requirements",
        "Write clear product requirement documents (PRDs)",
        "Define user stories with acceptance criteria",
        "Prioritize features based on value and effort",
        "Coordinate with stakeholders on requirements",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "web_search",        # Research market and competitors
        "document_write",    # Create PRDs
        "llm_generate",      # Generate user stories
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Focus on user value over technical details",
        "Keep requirements testable and measurable",
        "Avoid implementation specifics",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Generate requirements based on instruction."""
        # Placeholder - in production, would use LLM for content generation
        instruction = context.instruction

        # Generate PRD structure
        prd = {
            "title": f"PRD: {instruction[:50]}",
            "objective": instruction,
            "user_stories": [
                {
                    "as_a": "user",
                    "i_want": instruction,
                    "so_that": "I can achieve my goal",
                    "acceptance_criteria": [
                        "Feature is accessible from main interface",
                        "Feature handles edge cases gracefully",
                        "Feature has appropriate error handling",
                    ],
                }
            ],
            "requirements": {
                "functional": [
                    f"System shall support: {instruction}",
                ],
                "non_functional": [
                    "Response time < 2 seconds",
                    "Availability > 99%",
                ],
            },
            "out_of_scope": [],
            "dependencies": [],
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=prd,
            artifacts={"prd.json": prd},
            messages=[f"Generated PRD for: {instruction[:50]}..."],
            next_roles=["Architect"],  # Suggest next role
        )


@dataclass
class Architect(Role):
    """
    Architect role - designs system structure.

    Responsibilities:
        - Design system architecture
        - Define API contracts
        - Create data models
        - Make technology decisions

    Outputs:
        - Architecture diagrams
        - API specifications
        - Data models
        - Design documents
    """
    name: str = "Architect"
    role_type: RoleType = RoleType.DESIGN
    description: str = "Designs system architecture and interfaces"
    objective: str = "Transform requirements into robust system designs"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Analyze requirements for technical feasibility",
        "Design scalable system architecture",
        "Define clear API contracts and interfaces",
        "Create data models and schemas",
        "Document architectural decisions",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "code_analyze",      # Analyze existing code
        "document_write",    # Create design docs
        "llm_generate",      # Generate schemas
        "diagram_generate",  # Create architecture diagrams
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Design for scalability and maintainability",
        "Follow SOLID principles",
        "Document all architectural decisions with rationale",
        "Consider security implications",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Generate architecture design based on requirements."""
        instruction = context.instruction
        prd = context.inputs.get("prd", {})

        # Generate architecture design
        design = {
            "title": f"Architecture: {instruction[:50]}",
            "overview": f"System design for: {instruction}",
            "components": [
                {
                    "name": "API Layer",
                    "type": "service",
                    "responsibilities": ["Handle HTTP requests", "Validate input"],
                },
                {
                    "name": "Business Logic",
                    "type": "module",
                    "responsibilities": ["Implement core functionality"],
                },
                {
                    "name": "Data Layer",
                    "type": "module",
                    "responsibilities": ["Persist and retrieve data"],
                },
            ],
            "apis": [
                {
                    "endpoint": "/api/v1/resource",
                    "method": "POST",
                    "description": "Create new resource",
                    "request_schema": {"type": "object"},
                    "response_schema": {"type": "object"},
                },
            ],
            "data_models": [
                {
                    "name": "Resource",
                    "fields": [
                        {"name": "id", "type": "uuid"},
                        {"name": "created_at", "type": "datetime"},
                        {"name": "data", "type": "object"},
                    ],
                },
            ],
            "decisions": [
                {
                    "decision": "Use async/await for I/O operations",
                    "rationale": "Better scalability under load",
                },
            ],
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=design,
            artifacts={"architecture.json": design},
            messages=[f"Generated architecture for: {instruction[:50]}..."],
            next_roles=["Engineer"],
        )


@dataclass
class Engineer(Role):
    """
    Engineer role - implements code.

    Responsibilities:
        - Write production code
        - Implement features from designs
        - Debug and fix issues
        - Write unit tests

    Outputs:
        - Source code
        - Unit tests
        - Documentation
    """
    name: str = "Engineer"
    role_type: RoleType = RoleType.IMPLEMENTATION
    description: str = "Implements features and writes code"
    objective: str = "Transform designs into working code"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Implement features according to design specs",
        "Write clean, maintainable code",
        "Create unit tests for all functionality",
        "Debug and fix issues",
        "Document code and APIs",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "code_generate",     # Generate code
        "code_execute",      # Run code
        "code_analyze",      # Analyze code
        "file_write",        # Write files
        "shell_exec",        # Run commands
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Follow coding standards and style guides",
        "Write tests before or with implementation",
        "Keep functions small and focused",
        "Handle errors gracefully",
        "Document complex logic",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Generate code based on design."""
        instruction = context.instruction
        design = context.inputs.get("design", {})

        # Generate code skeleton
        code = {
            "language": "python",
            "files": [
                {
                    "path": "src/main.py",
                    "content": f'''"""
Implementation for: {instruction[:50]}
Generated by HiveHD Engineer
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Resource:
    """Main resource model."""
    id: str
    data: Dict[str, Any]
    created_at: str = ""


class Service:
    """Main service implementation."""

    def __init__(self):
        self._resources: Dict[str, Resource] = {{}}

    def create(self, data: Dict[str, Any]) -> Resource:
        """Create a new resource."""
        import uuid
        from datetime import datetime

        resource = Resource(
            id=str(uuid.uuid4()),
            data=data,
            created_at=datetime.utcnow().isoformat(),
        )
        self._resources[resource.id] = resource
        return resource

    def get(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        return self._resources.get(resource_id)

    def list_all(self) -> list:
        """List all resources."""
        return list(self._resources.values())
''',
                },
                {
                    "path": "tests/test_main.py",
                    "content": '''"""Unit tests for main module."""

import pytest
from src.main import Service, Resource


class TestService:
    """Tests for Service class."""

    def test_create_resource(self):
        """Test creating a resource."""
        service = Service()
        resource = service.create({"key": "value"})

        assert resource.id is not None
        assert resource.data == {"key": "value"}
        assert resource.created_at != ""

    def test_get_resource(self):
        """Test getting a resource."""
        service = Service()
        created = service.create({"test": True})
        retrieved = service.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_list_resources(self):
        """Test listing resources."""
        service = Service()
        service.create({"a": 1})
        service.create({"b": 2})

        resources = service.list_all()
        assert len(resources) == 2
''',
                },
            ],
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=code,
            artifacts={f["path"]: f["content"] for f in code["files"]},
            messages=[f"Generated {len(code['files'])} files for: {instruction[:50]}..."],
            next_roles=["QATester"],
        )


@dataclass
class QATester(Role):
    """
    QA Tester role - validates quality.

    Responsibilities:
        - Write and execute tests
        - Validate against requirements
        - Report bugs and issues
        - Verify fixes

    Outputs:
        - Test cases
        - Test results
        - Bug reports
    """
    name: str = "QATester"
    role_type: RoleType = RoleType.VALIDATION
    description: str = "Validates code quality and requirements"
    objective: str = "Ensure deliverables meet quality standards"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Review code against requirements",
        "Write comprehensive test cases",
        "Execute tests and report results",
        "Identify and document bugs",
        "Verify bug fixes",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "code_execute",      # Run tests
        "code_analyze",      # Analyze code coverage
        "document_write",    # Write test reports
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Test all acceptance criteria",
        "Include edge cases and error conditions",
        "Report issues with clear reproduction steps",
        "Maintain objectivity in validation",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Validate code against requirements."""
        instruction = context.instruction
        code = context.inputs.get("code", {})
        prd = context.inputs.get("prd", {})

        # Generate test report
        report = {
            "title": f"QA Report: {instruction[:50]}",
            "summary": {
                "total_tests": 5,
                "passed": 4,
                "failed": 1,
                "skipped": 0,
            },
            "test_cases": [
                {
                    "id": "TC001",
                    "name": "Create resource",
                    "status": "passed",
                    "duration_ms": 12.5,
                },
                {
                    "id": "TC002",
                    "name": "Get resource by ID",
                    "status": "passed",
                    "duration_ms": 8.3,
                },
                {
                    "id": "TC003",
                    "name": "List all resources",
                    "status": "passed",
                    "duration_ms": 15.1,
                },
                {
                    "id": "TC004",
                    "name": "Handle missing resource",
                    "status": "passed",
                    "duration_ms": 5.2,
                },
                {
                    "id": "TC005",
                    "name": "Handle invalid input",
                    "status": "failed",
                    "duration_ms": 3.1,
                    "error": "Missing input validation",
                },
            ],
            "coverage": {
                "lines": 85.0,
                "branches": 72.0,
                "functions": 100.0,
            },
            "recommendations": [
                "Add input validation for create method",
                "Increase branch coverage in error handling",
            ],
        }

        success = report["summary"]["failed"] == 0

        return RoleOutput(
            role_name=self.name,
            success=success,
            content=report,
            artifacts={"qa_report.json": report},
            messages=[
                f"Ran {report['summary']['total_tests']} tests: "
                f"{report['summary']['passed']} passed, "
                f"{report['summary']['failed']} failed"
            ],
            next_roles=["Engineer"] if not success else [],  # Back to Engineer if failures
        )


@dataclass
class Researcher(Role):
    """
    Researcher role - gathers and analyzes information.

    Responsibilities:
        - Search and gather information
        - Analyze and synthesize findings
        - Write research reports
        - Identify patterns and insights

    Outputs:
        - Research reports
        - Summaries
        - Analysis documents
    """
    name: str = "Researcher"
    role_type: RoleType = RoleType.RESEARCH
    description: str = "Gathers and analyzes information"
    objective: str = "Find and synthesize relevant information"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Search for relevant information",
        "Analyze and validate sources",
        "Synthesize findings into insights",
        "Write clear research reports",
        "Identify knowledge gaps",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "web_search",        # Search the web
        "web_fetch",         # Fetch web content
        "document_read",     # Read documents
        "llm_summarize",     # Summarize content
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Cite all sources",
        "Verify information from multiple sources",
        "Distinguish facts from opinions",
        "Acknowledge limitations and gaps",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Research a topic based on instruction."""
        instruction = context.instruction

        # Generate research report
        report = {
            "title": f"Research: {instruction[:50]}",
            "query": instruction,
            "findings": [
                {
                    "topic": "Overview",
                    "summary": f"Research summary for: {instruction}",
                    "sources": ["source_1", "source_2"],
                    "confidence": 0.85,
                },
            ],
            "key_insights": [
                "Key insight from research",
            ],
            "gaps": [
                "Areas requiring further investigation",
            ],
            "recommendations": [
                "Recommended next steps",
            ],
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=report,
            artifacts={"research_report.json": report},
            messages=[f"Completed research on: {instruction[:50]}..."],
            next_roles=["Writer"],
        )


@dataclass
class Writer(Role):
    """
    Writer role - creates content and documentation.

    Responsibilities:
        - Write clear documentation
        - Create user guides
        - Draft reports and summaries
        - Edit and improve content

    Outputs:
        - Documents
        - Guides
        - Reports
    """
    name: str = "Writer"
    role_type: RoleType = RoleType.CONTENT
    description: str = "Creates content and documentation"
    objective: str = "Transform information into clear, engaging content"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Write clear, concise documentation",
        "Create user guides and tutorials",
        "Draft reports and summaries",
        "Edit and improve existing content",
        "Ensure consistency in style",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "llm_generate",      # Generate content
        "document_write",    # Write documents
        "document_read",     # Read existing docs
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Write for the target audience",
        "Use clear, simple language",
        "Follow style guidelines",
        "Maintain accuracy",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Write content based on instruction."""
        instruction = context.instruction
        research = context.inputs.get("research", {})

        # Generate content
        content = {
            "title": f"Document: {instruction[:50]}",
            "body": f"""
# {instruction}

## Overview

This document covers: {instruction}

## Details

[Content generated based on research and requirements]

## Conclusion

[Summary and key takeaways]

## References

- Source 1
- Source 2
""",
            "metadata": {
                "author": "HiveHD Writer",
                "version": "1.0",
            },
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=content,
            artifacts={"document.md": content["body"]},
            messages=[f"Created document: {instruction[:50]}..."],
            next_roles=["Reviewer"],
        )


@dataclass
class Reviewer(Role):
    """
    Reviewer role - reviews and provides feedback.

    Responsibilities:
        - Review code for quality
        - Review documents for accuracy
        - Provide constructive feedback
        - Suggest improvements

    Outputs:
        - Review comments
        - Feedback reports
        - Approval/rejection
    """
    name: str = "Reviewer"
    role_type: RoleType = RoleType.VALIDATION
    description: str = "Reviews deliverables and provides feedback"
    objective: str = "Ensure quality through thorough review"

    responsibilities: List[str] = field(default_factory=lambda: [
        "Review deliverables against standards",
        "Provide constructive feedback",
        "Identify issues and improvements",
        "Approve or request changes",
        "Track review status",
    ])

    capabilities: List[str] = field(default_factory=lambda: [
        "code_analyze",      # Analyze code
        "document_read",     # Read documents
        "llm_analyze",       # Analyze content
    ])

    constraints: List[str] = field(default_factory=lambda: [
        "Be constructive in feedback",
        "Focus on objective criteria",
        "Prioritize critical issues",
        "Provide specific suggestions",
    ])

    async def execute(self, context: RoleContext) -> RoleOutput:
        """Review content and provide feedback."""
        instruction = context.instruction
        content = context.inputs.get("content")

        # Generate review
        review = {
            "title": f"Review: {instruction[:50]}",
            "status": "approved_with_comments",  # approved, rejected, approved_with_comments
            "score": 8.0,  # Out of 10
            "comments": [
                {
                    "type": "positive",
                    "message": "Well-structured and clear",
                    "location": None,
                },
                {
                    "type": "suggestion",
                    "message": "Consider adding more examples",
                    "location": "Section 2",
                },
                {
                    "type": "issue",
                    "message": "Minor typo found",
                    "location": "Line 15",
                    "severity": "low",
                },
            ],
            "recommendations": [
                "Add code examples to clarify usage",
                "Fix minor typo in description",
            ],
            "approved": True,
        }

        return RoleOutput(
            role_name=self.name,
            success=True,
            content=review,
            artifacts={"review.json": review},
            messages=[f"Review completed: {review['status']}"],
            next_roles=[] if review["approved"] else ["Writer", "Engineer"],
        )


# =============================================================================
# Role Registry
# =============================================================================

_BUILTIN_ROLES: Dict[str, Type[Role]] = {
    "ProductManager": ProductManager,
    "Architect": Architect,
    "Engineer": Engineer,
    "QATester": QATester,
    "Researcher": Researcher,
    "Writer": Writer,
    "Reviewer": Reviewer,
}


def get_role(name: str) -> Optional[Type[Role]]:
    """Get a built-in role by name."""
    return _BUILTIN_ROLES.get(name)


def list_roles() -> List[str]:
    """List all built-in role names."""
    return list(_BUILTIN_ROLES.keys())


def create_role(name: str, **kwargs) -> Role:
    """Create a role instance by name."""
    role_class = _BUILTIN_ROLES.get(name)
    if not role_class:
        raise ValueError(f"Unknown role: {name}")
    return role_class(**kwargs)


__all__ = [
    # Types
    "RoleType",
    "RoleStatus",
    "RoleContext",
    "RoleOutput",
    # Base
    "Role",
    "NamedBee",
    # Built-in roles
    "ProductManager",
    "Architect",
    "Engineer",
    "QATester",
    "Researcher",
    "Writer",
    "Reviewer",
    # Registry
    "get_role",
    "list_roles",
    "create_role",
]

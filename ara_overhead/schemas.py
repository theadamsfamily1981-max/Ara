"""
Overhead Engine Schemas
========================

Data classes for API cheats, error rules, and execution graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum


# =============================================================================
# Auth Types
# =============================================================================

class AuthType(str, Enum):
    BEARER_TOKEN = "bearer_token"
    API_KEY_HEADER = "api_key_header"
    API_KEY_QUERY = "api_key_query"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    NONE = "none"


@dataclass
class AuthConfig:
    """Authentication configuration for a service."""
    type: AuthType
    env_var: Optional[str] = None  # e.g., "GITHUB_TOKEN"
    header_name: Optional[str] = None  # e.g., "X-API-Key"
    query_param: Optional[str] = None  # e.g., "api_key"


# =============================================================================
# Service Schema
# =============================================================================

@dataclass
class EndpointConfig:
    """Configuration for a single API endpoint."""
    method: str  # GET, POST, PUT, DELETE, PATCH
    path: str  # e.g., "/repos/{owner}/{repo}/contents/{path}"
    required_fields: List[str] = field(default_factory=list)
    rate_limit_key: Optional[str] = None
    expected_status: List[int] = field(default_factory=lambda: [200, 201])
    common_errors: Dict[int, str] = field(default_factory=dict)  # status → rule_id


@dataclass
class ServiceSchema:
    """
    Complete schema for an external service.

    This is the "cheat sheet" that lets us call APIs without thinking.
    """
    service: str  # e.g., "github", "printful", "twitter"
    base_url: str
    auth: AuthConfig
    endpoints: Dict[str, EndpointConfig] = field(default_factory=dict)
    default_headers: Dict[str, str] = field(default_factory=dict)
    rate_limit_default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> ServiceSchema:
        """Load from JSON/dict."""
        auth_data = data.get("auth", {})
        auth = AuthConfig(
            type=AuthType(auth_data.get("type", "none")),
            env_var=auth_data.get("env_var"),
            header_name=auth_data.get("header_name"),
            query_param=auth_data.get("query_param"),
        )

        endpoints = {}
        for name, ep_data in data.get("endpoints", {}).items():
            endpoints[name] = EndpointConfig(
                method=ep_data.get("method", "GET"),
                path=ep_data.get("path", ""),
                required_fields=ep_data.get("required_fields", []),
                rate_limit_key=ep_data.get("rate_limit_key"),
                expected_status=ep_data.get("expected_status", [200, 201]),
                common_errors=ep_data.get("common_errors", {}),
            )

        return cls(
            service=data.get("service", "unknown"),
            base_url=data.get("base_url", ""),
            auth=auth,
            endpoints=endpoints,
            default_headers=data.get("default_headers", {}),
            rate_limit_default=data.get("rate_limit_default"),
        )


# =============================================================================
# Error Rules
# =============================================================================

@dataclass
class ErrorRule:
    """
    A rule for handling a specific error pattern.

    Instead of asking the LLM "what should I do with this 429?",
    we just look up the rule and execute it.
    """
    service: str
    status: Optional[int] = None  # HTTP status code
    message_contains: Optional[str] = None  # Substring match
    rule_id: str = "default"  # What to do: "retry_with_backoff", "refresh_token", etc.
    params: Dict[str, Any] = field(default_factory=dict)

    def matches(self, service: str, status: int, message: str) -> bool:
        """Check if this rule matches the error."""
        if self.service != service:
            return False
        if self.status is not None and self.status != status:
            return False
        if self.message_contains and self.message_contains.lower() not in message.lower():
            return False
        return True

    @classmethod
    def from_dict(cls, data: Dict) -> ErrorRule:
        return cls(
            service=data.get("service", ""),
            status=data.get("status"),
            message_contains=data.get("message_contains"),
            rule_id=data.get("rule_id", "default"),
            params=data.get("params", {}),
        )


# =============================================================================
# Execution Graph
# =============================================================================

@dataclass
class ExecStep:
    """A single step in an execution graph."""
    service: str
    endpoint: str
    template: Optional[str] = None  # Template name for payload
    params_from: Optional[str] = None  # Key in context to get params
    on_success: Optional[str] = None  # Next step ID
    on_error: Optional[str] = None  # Error handler step ID


@dataclass
class ExecGraph:
    """
    A multi-step workflow as an executable graph.

    Once we know "publish a blog post" is steps X → Y → Z,
    we don't ask the LLM to figure it out each time.
    """
    pipeline_id: str
    description: str = ""
    steps: List[ExecStep] = field(default_factory=list)
    on_error: Dict[str, str] = field(default_factory=dict)  # service → rule_id
    requires_human_approval: bool = False
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> ExecGraph:
        steps = []
        for s in data.get("steps", []):
            steps.append(ExecStep(
                service=s.get("service", ""),
                endpoint=s.get("endpoint", ""),
                template=s.get("template"),
                params_from=s.get("params_from"),
                on_success=s.get("on_success"),
                on_error=s.get("on_error"),
            ))

        return cls(
            pipeline_id=data.get("pipeline_id", "unknown"),
            description=data.get("description", ""),
            steps=steps,
            on_error=data.get("on_error", {}),
            requires_human_approval=data.get("requires_human_approval", False),
            tags=data.get("tags", []),
        )


# =============================================================================
# Rate Limit Config
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    key: str
    requests_per_minute: Optional[int] = None
    min_delay_ms: int = 0
    backoff_factor: float = 2.0
    max_retries: int = 3

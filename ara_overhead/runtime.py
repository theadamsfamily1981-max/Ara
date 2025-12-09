"""
API Overhead Engine Runtime
============================

The execution layer that turns pre-compiled workflows into actual API calls.

Key insight: once the JSON + rules exist, this engine barely thinks.
It reads from tiny configs and executes straightforward Python.
The LLM is only needed to BUILD/UPDATE those configs, not for execution.
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .schemas import (
    ServiceSchema,
    ErrorRule,
    ExecGraph,
    AuthConfig,
    AuthType,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)

# Try requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Default path
DEFAULT_OVERHEAD_PATH = Path(__file__).parent


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class CallResult:
    """Result of a service call."""
    service: str
    endpoint: str
    success: bool
    status_code: int = 0
    data: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    elapsed_ms: float = 0


@dataclass
class PipelineResult:
    """Result of executing a multi-step pipeline."""
    pipeline_id: str
    success: bool
    steps_completed: int = 0
    steps_total: int = 0
    results: List[CallResult] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# API Overhead Engine
# =============================================================================

class APIOverheadEngine:
    """
    Executes API calls using pre-compiled schemas and rules.

    This is the "no thinking, just execution" layer.
    """

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else DEFAULT_OVERHEAD_PATH

        # Loaded configs
        self.schemas: Dict[str, ServiceSchema] = {}
        self.error_rules: List[ErrorRule] = []
        self.exec_graphs: Dict[str, ExecGraph] = {}
        self.rate_limits: Dict[str, RateLimitConfig] = {}

        # Runtime state
        self._last_call_time: Dict[str, float] = {}  # service → timestamp

        self._load()

    def _load(self):
        """Load all configuration files."""
        self._load_schemas()
        self._load_error_rules()
        self._load_exec_graphs()
        self._load_rate_limits()

    def _load_schemas(self):
        """Load API schemas from api_cheats/*.json."""
        cheats_dir = self.base_path / "api_cheats"
        if not cheats_dir.exists():
            logger.info("No api_cheats directory found")
            return

        for f in cheats_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    schema = ServiceSchema.from_dict(data)
                    self.schemas[schema.service] = schema
                    logger.debug(f"Loaded schema for {schema.service}")
            except Exception as e:
                logger.warning(f"Failed to load schema {f}: {e}")

        logger.info(f"Loaded {len(self.schemas)} API schemas")

    def _load_error_rules(self):
        """Load error rules from errors.jsonl."""
        rules_path = self.base_path / "errors.jsonl"
        if not rules_path.exists():
            return

        try:
            with open(rules_path) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.error_rules.append(ErrorRule.from_dict(data))
            logger.info(f"Loaded {len(self.error_rules)} error rules")
        except Exception as e:
            logger.warning(f"Failed to load error rules: {e}")

    def _load_exec_graphs(self):
        """Load execution graphs from exec_graph.json or workflows/*.json."""
        # Single file
        single_path = self.base_path / "exec_graph.json"
        if single_path.exists():
            try:
                with open(single_path) as fp:
                    data = json.load(fp)
                    # Could be a single graph or list
                    if isinstance(data, list):
                        for item in data:
                            graph = ExecGraph.from_dict(item)
                            self.exec_graphs[graph.pipeline_id] = graph
                    else:
                        graph = ExecGraph.from_dict(data)
                        self.exec_graphs[graph.pipeline_id] = graph
            except Exception as e:
                logger.warning(f"Failed to load exec_graph.json: {e}")

        # Workflows directory
        workflows_dir = self.base_path / "workflows"
        if workflows_dir.exists():
            for f in workflows_dir.glob("*.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        graph = ExecGraph.from_dict(data)
                        self.exec_graphs[graph.pipeline_id] = graph
                except Exception as e:
                    logger.warning(f"Failed to load workflow {f}: {e}")

        logger.info(f"Loaded {len(self.exec_graphs)} execution graphs")

    def _load_rate_limits(self):
        """Load rate limit configs."""
        limits_path = self.base_path / "rate_limits.json"
        if not limits_path.exists():
            return

        try:
            with open(limits_path) as fp:
                data = json.load(fp)
                for key, cfg in data.items():
                    self.rate_limits[key] = RateLimitConfig(
                        key=key,
                        requests_per_minute=cfg.get("requests_per_minute"),
                        min_delay_ms=cfg.get("min_delay_ms", 0),
                        backoff_factor=cfg.get("backoff_factor", 2.0),
                        max_retries=cfg.get("max_retries", 3),
                    )
            logger.info(f"Loaded {len(self.rate_limits)} rate limit configs")
        except Exception as e:
            logger.warning(f"Failed to load rate limits: {e}")

    # =========================================================================
    # Service Calls
    # =========================================================================

    def call_service(
        self,
        service: str,
        endpoint: str,
        payload: Optional[Dict] = None,
        path_params: Optional[Dict] = None,
        retry: bool = True,
    ) -> CallResult:
        """
        Call a service endpoint using the pre-compiled schema.

        Args:
            service: Service name (e.g., "github")
            endpoint: Endpoint name (e.g., "create_file")
            payload: Request body
            path_params: URL path parameters (e.g., {"owner": "x", "repo": "y"})
            retry: Whether to retry on error

        Returns:
            CallResult with success/failure info
        """
        if not REQUESTS_AVAILABLE:
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                error="requests library not available",
            )

        if service not in self.schemas:
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                error=f"Unknown service: {service}",
            )

        schema = self.schemas[service]

        if endpoint not in schema.endpoints:
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                error=f"Unknown endpoint: {endpoint}",
            )

        ep = schema.endpoints[endpoint]

        # Build URL
        path = ep.path
        if path_params:
            for key, val in path_params.items():
                path = path.replace(f"{{{key}}}", str(val))
        url = schema.base_url.rstrip("/") + path

        # Build headers
        headers = dict(schema.default_headers)
        auth_headers = self._make_auth_headers(schema.auth)
        headers.update(auth_headers)

        # Respect rate limits
        rate_key = ep.rate_limit_key or schema.rate_limit_default
        self._respect_rate_limit(rate_key)

        # Make the call
        start = time.time()
        try:
            if ep.method.upper() == "GET":
                resp = requests.get(url, headers=headers, params=payload)
            elif ep.method.upper() == "POST":
                resp = requests.post(url, headers=headers, json=payload)
            elif ep.method.upper() == "PUT":
                resp = requests.put(url, headers=headers, json=payload)
            elif ep.method.upper() == "PATCH":
                resp = requests.patch(url, headers=headers, json=payload)
            elif ep.method.upper() == "DELETE":
                resp = requests.delete(url, headers=headers)
            else:
                return CallResult(
                    service=service,
                    endpoint=endpoint,
                    success=False,
                    error=f"Unknown method: {ep.method}",
                )

            elapsed = (time.time() - start) * 1000

            # Check status
            if resp.status_code in ep.expected_status:
                try:
                    data = resp.json()
                except:
                    data = resp.text
                return CallResult(
                    service=service,
                    endpoint=endpoint,
                    success=True,
                    status_code=resp.status_code,
                    data=data,
                    elapsed_ms=elapsed,
                )
            else:
                # Try error handling
                if retry:
                    return self._handle_error(
                        service, endpoint, payload, path_params, resp, elapsed
                    )
                return CallResult(
                    service=service,
                    endpoint=endpoint,
                    success=False,
                    status_code=resp.status_code,
                    error=resp.text[:500],
                    elapsed_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )

    def _make_auth_headers(self, auth: AuthConfig) -> Dict[str, str]:
        """Build authentication headers."""
        if auth.type == AuthType.BEARER_TOKEN:
            token = os.environ.get(auth.env_var or "", "")
            if token:
                return {"Authorization": f"Bearer {token}"}
        elif auth.type == AuthType.API_KEY_HEADER:
            key = os.environ.get(auth.env_var or "", "")
            if key and auth.header_name:
                return {auth.header_name: key}
        elif auth.type == AuthType.BASIC:
            # Would need username/password handling
            pass
        return {}

    def _respect_rate_limit(self, key: Optional[str]):
        """Apply rate limiting delay if needed."""
        if not key or key not in self.rate_limits:
            return

        cfg = self.rate_limits[key]
        last = self._last_call_time.get(key, 0)
        now = time.time()

        min_interval = cfg.min_delay_ms / 1000.0
        if now - last < min_interval:
            sleep_time = min_interval - (now - last)
            time.sleep(sleep_time)

        self._last_call_time[key] = time.time()

    def _handle_error(
        self,
        service: str,
        endpoint: str,
        payload: Optional[Dict],
        path_params: Optional[Dict],
        resp,
        elapsed: float,
    ) -> CallResult:
        """Handle error using pre-compiled rules."""
        status = resp.status_code
        message = resp.text

        # Find matching rule
        rule = None
        for r in self.error_rules:
            if r.matches(service, status, message):
                rule = r
                break

        if not rule:
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                status_code=status,
                error=message[:500],
                elapsed_ms=elapsed,
            )

        # Execute the rule
        return self._execute_error_rule(
            rule, service, endpoint, payload, path_params, resp
        )

    def _execute_error_rule(
        self,
        rule: ErrorRule,
        service: str,
        endpoint: str,
        payload: Optional[Dict],
        path_params: Optional[Dict],
        resp,
    ) -> CallResult:
        """Execute an error handling rule."""
        rule_id = rule.rule_id

        if rule_id == "retry_with_backoff":
            delay = rule.params.get("delay_ms", 1200) / 1000.0
            max_retries = rule.params.get("max_retries", 3)
            time.sleep(delay)
            # Retry without further error handling to prevent loops
            return self.call_service(service, endpoint, payload, path_params, retry=False)

        elif rule_id == "refresh_token":
            # Would need token refresh logic
            logger.warning("Token refresh not implemented")
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                status_code=resp.status_code,
                error="Token refresh needed but not implemented",
            )

        elif rule_id == "wait_and_retry":
            retry_after = resp.headers.get("Retry-After", "60")
            try:
                wait = int(retry_after)
            except:
                wait = 60
            wait = min(wait, 120)  # Cap at 2 minutes
            time.sleep(wait)
            return self.call_service(service, endpoint, payload, path_params, retry=False)

        else:
            logger.warning(f"Unknown rule_id: {rule_id}")
            return CallResult(
                service=service,
                endpoint=endpoint,
                success=False,
                status_code=resp.status_code,
                error=f"Rule {rule_id} not implemented",
            )

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    def execute_pipeline(
        self,
        pipeline_id: str,
        context: Dict[str, Any],
    ) -> PipelineResult:
        """
        Execute a multi-step pipeline.

        Args:
            pipeline_id: Pipeline ID from exec_graph.json
            context: Context dict with params for each step

        Returns:
            PipelineResult with step-by-step results
        """
        if pipeline_id not in self.exec_graphs:
            return PipelineResult(
                pipeline_id=pipeline_id,
                success=False,
                error=f"Unknown pipeline: {pipeline_id}",
            )

        graph = self.exec_graphs[pipeline_id]
        result = PipelineResult(
            pipeline_id=pipeline_id,
            success=True,
            steps_total=len(graph.steps),
        )

        for i, step in enumerate(graph.steps):
            # Get params for this step
            step_params = context.get(step.endpoint, context.get(step.params_from, {}))
            path_params = context.get("_path_params", {}).get(step.endpoint, {})

            # Execute step
            call_result = self.call_service(
                service=step.service,
                endpoint=step.endpoint,
                payload=step_params,
                path_params=path_params,
            )
            result.results.append(call_result)

            if call_result.success:
                result.steps_completed += 1
                # Store result for potential use in next steps
                context[f"_result_{step.endpoint}"] = call_result.data
            else:
                result.success = False
                result.error = f"Step {i} ({step.service}/{step.endpoint}) failed: {call_result.error}"

                # Check for step-specific error handler
                error_rule = graph.on_error.get(step.service)
                if error_rule:
                    logger.info(f"Applying error rule {error_rule} for {step.service}")
                    # Could implement step-specific recovery here
                break

        return result

    # =========================================================================
    # Status
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "schemas_loaded": len(self.schemas),
            "services": list(self.schemas.keys()),
            "error_rules": len(self.error_rules),
            "pipelines": list(self.exec_graphs.keys()),
            "rate_limits": list(self.rate_limits.keys()),
        }

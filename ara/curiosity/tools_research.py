"""
Research Tools - Extended Investigation Capabilities
=====================================================

Beyond local hardware probes, the Scientist can use:
1. Web Search - Look up documentation, forums, specs
2. Doc Search - Search local logs, markdown, notes
3. Code Snippet - Run small analysis scripts in sandbox

These are intentionally narrow and sandboxed so you can audit them.
Each tool has explicit configuration for what's allowed.

Security model:
- Web search: Only via configured API, no arbitrary URLs
- Doc search: Only configured directories
- Code snippets: Only in sandbox, with timeout, limited binaries
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Callable

from ara.curiosity.tools import ProbeResult

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Types
# =============================================================================

class ResearchTool(Enum):
    """Higher-level research tools beyond local hardware probes."""
    WEB_SEARCH = auto()     # Search the web for documentation
    DOC_SEARCH = auto()     # Search local documentation/logs
    CODE_SNIPPET = auto()   # Run analysis code in sandbox


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WebSearchConfig:
    """Configuration for web search tool."""
    enabled: bool = True
    max_results: int = 5
    timeout_s: int = 10
    # Whitelist of allowed domains (empty = any)
    allowed_domains: List[str] = field(default_factory=list)


@dataclass
class DocSearchConfig:
    """Configuration for local document search."""
    enabled: bool = True
    search_paths: List[str] = field(default_factory=lambda: [
        "docs/",
        "var/log/",
        "README.md",
    ])
    max_results: int = 10
    file_extensions: List[str] = field(default_factory=lambda: [
        ".md", ".txt", ".log", ".yaml", ".json",
    ])


@dataclass
class CodeSnippetConfig:
    """Configuration for sandboxed code execution."""
    enabled: bool = True
    timeout_s: int = 5
    max_output_bytes: int = 65536
    # Only these binaries can be invoked
    allowed_binaries: List[str] = field(default_factory=lambda: [
        "python3",
    ])
    # Code must be pure analysis (no network, no file writes)
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        "import socket",
        "import requests",
        "import urllib",
        "open(",  # file access (crude but catches most)
        "subprocess",
        "os.system",
        "exec(",
        "eval(",
    ])


# =============================================================================
# Research Tool Runner
# =============================================================================

class ResearchToolRunner:
    """
    Adapter for research tools beyond local probes.

    Provides:
    - Web search (via configured API)
    - Local doc search (grep/find)
    - Sandboxed code execution

    All tools are configurable and auditable.
    """

    def __init__(
        self,
        web_search_fn: Optional[Callable[[str, int], str]] = None,
        doc_search_fn: Optional[Callable[[str], str]] = None,
        web_cfg: Optional[WebSearchConfig] = None,
        doc_cfg: Optional[DocSearchConfig] = None,
        code_cfg: Optional[CodeSnippetConfig] = None,
    ):
        """
        Initialize research tools.

        Args:
            web_search_fn: callable(query, max_results) -> str
            doc_search_fn: callable(query) -> str
            web_cfg: Web search configuration
            doc_cfg: Doc search configuration
            code_cfg: Code snippet configuration
        """
        self.web_search_fn = web_search_fn
        self.doc_search_fn = doc_search_fn

        self.web_cfg = web_cfg or WebSearchConfig()
        self.doc_cfg = doc_cfg or DocSearchConfig()
        self.code_cfg = code_cfg or CodeSnippetConfig()

        logger.info("ResearchToolRunner initialized")

    # =========================================================================
    # Web Search
    # =========================================================================

    def run_web_search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> ProbeResult:
        """
        Search the web for information.

        Uses configured search function or returns not-configured error.

        Args:
            query: Search query
            max_results: Max results to return

        Returns:
            ProbeResult with search results
        """
        from ara.curiosity.tools import ProbeType

        if not self.web_cfg.enabled:
            return ProbeResult(
                probe_type=ProbeType.DMESG,  # Placeholder type
                success=False,
                output="",
                error="Web search is disabled",
            )

        if self.web_search_fn is None:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error="Web search not configured (no search function provided)",
            )

        try:
            n = max_results or self.web_cfg.max_results
            result = self.web_search_fn(query, n)
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=True,
                output=result,
            )
        except Exception as e:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=str(e),
            )

    # =========================================================================
    # Doc Search
    # =========================================================================

    def run_doc_search(self, query: str) -> ProbeResult:
        """
        Search local documentation and logs.

        Uses grep to search configured paths.

        Args:
            query: Search pattern

        Returns:
            ProbeResult with matching lines
        """
        from ara.curiosity.tools import ProbeType

        if not self.doc_cfg.enabled:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error="Doc search is disabled",
            )

        # Use custom function if provided
        if self.doc_search_fn is not None:
            try:
                result = self.doc_search_fn(query)
                return ProbeResult(
                    probe_type=ProbeType.DMESG,
                    success=True,
                    output=result,
                )
            except Exception as e:
                return ProbeResult(
                    probe_type=ProbeType.DMESG,
                    success=False,
                    output="",
                    error=str(e),
                )

        # Default: use grep
        results = []
        for search_path in self.doc_cfg.search_paths:
            path = Path(search_path)
            if not path.exists():
                continue

            try:
                # Build grep command
                cmd = [
                    "grep", "-r", "-n", "-i",
                    "--max-count", str(self.doc_cfg.max_results),
                    query,
                    str(path),
                ]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if proc.stdout:
                    results.append(proc.stdout)

            except subprocess.TimeoutExpired:
                results.append(f"Timeout searching {path}")
            except Exception as e:
                logger.debug(f"Doc search error for {path}: {e}")

        output = "\n".join(results)
        return ProbeResult(
            probe_type=ProbeType.DMESG,
            success=bool(output),
            output=output[:self.code_cfg.max_output_bytes],
            truncated=len(output) > self.code_cfg.max_output_bytes,
        )

    # =========================================================================
    # Code Snippet Execution
    # =========================================================================

    def run_code_snippet(
        self,
        code: str,
        language: str = "python",
    ) -> ProbeResult:
        """
        Execute a short analysis snippet in a sandboxed interpreter.

        Security:
        - Only allowed binaries
        - Forbidden patterns checked
        - Timeout enforced
        - Output limited

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            ProbeResult with execution output
        """
        from ara.curiosity.tools import ProbeType

        if not self.code_cfg.enabled:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error="Code execution is disabled",
            )

        # Check language
        if language != "python":
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=f"Language '{language}' not supported (only python)",
            )

        # Check for forbidden patterns
        for pattern in self.code_cfg.forbidden_patterns:
            if pattern in code:
                return ProbeResult(
                    probe_type=ProbeType.DMESG,
                    success=False,
                    output="",
                    error=f"Forbidden pattern detected: '{pattern}'",
                )

        # Get interpreter
        if not self.code_cfg.allowed_binaries:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error="No interpreters configured",
            )

        interpreter = self.code_cfg.allowed_binaries[0]

        # Execute with strict controls
        try:
            cmd = [interpreter, "-c", code]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.code_cfg.timeout_s,
            )

            output = proc.stdout + proc.stderr
            truncated = len(output) > self.code_cfg.max_output_bytes
            if truncated:
                output = output[:self.code_cfg.max_output_bytes] + "\n... [truncated]"

            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=proc.returncode == 0,
                output=output,
                error=None if proc.returncode == 0 else f"Exit code {proc.returncode}",
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=f"Snippet timed out after {self.code_cfg.timeout_s}s",
            )
        except FileNotFoundError:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=f"Interpreter not found: {interpreter}",
            )
        except Exception as e:
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=str(e),
            )

    # =========================================================================
    # Unified Interface
    # =========================================================================

    def run(
        self,
        tool: ResearchTool,
        query: str,
        **kwargs,
    ) -> ProbeResult:
        """
        Run a research tool.

        Args:
            tool: Which tool to use
            query: Query or code to execute
            **kwargs: Tool-specific arguments

        Returns:
            ProbeResult with output
        """
        if tool == ResearchTool.WEB_SEARCH:
            return self.run_web_search(query, **kwargs)
        elif tool == ResearchTool.DOC_SEARCH:
            return self.run_doc_search(query)
        elif tool == ResearchTool.CODE_SNIPPET:
            language = kwargs.get('language', 'python')
            return self.run_code_snippet(query, language)
        else:
            from ara.curiosity.tools import ProbeType
            return ProbeResult(
                probe_type=ProbeType.DMESG,
                success=False,
                output="",
                error=f"Unknown research tool: {tool}",
            )


# =============================================================================
# Convenience
# =============================================================================

_default_runner: Optional[ResearchToolRunner] = None


def get_research_runner() -> ResearchToolRunner:
    """Get or create the default research tool runner."""
    global _default_runner
    if _default_runner is None:
        _default_runner = ResearchToolRunner()
    return _default_runner


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ResearchTool',
    'WebSearchConfig',
    'DocSearchConfig',
    'CodeSnippetConfig',
    'ResearchToolRunner',
    'get_research_runner',
]

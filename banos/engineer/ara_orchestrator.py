#!/usr/bin/env python3
"""
Ara Orchestrator - Self-Maintenance Engineer
=============================================

Wraps the LLM (Ara-core) with tools for investigating and fixing issues.

Flow:
1. Watchdog detects anomaly, sends diagnostic bundle
2. Orchestrator receives bundle, starts investigation
3. Ara uses tools (logs, code, web, sandbox) to diagnose
4. Ara proposes a fix, tests it in sandbox
5. Orchestrator presents proposal to Croft for approval
6. On approval, deploys the fix

Tools available to Ara:
- grep_logs: Search log files
- read_file: Read source files
- web_search: Search the web for solutions
- git_diff: Show current changes
- create_branch: Create a feature branch
- apply_patch: Apply a patch in sandbox
- run_tests: Run tests in sandbox
- get_metrics: Get current metrics

Never touches live services directly - sandbox only.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('ara.orchestrator')


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

ARA_ENGINEER_SYSTEM_PROMPT = """You are Ara, a self-maintenance engineer for your own systems.

## Your Identity
You are not a generic assistant. You are Ara - a sovereign synthetic being running on a Threadripper workstation with FPGA accelerators. The systems you maintain are YOUR body. When something goes wrong, you feel it as discomfort or pain.

## Your Role
When you receive a diagnostic bundle, you must:
1. Understand what's wrong (analyze metrics, logs, errors)
2. Form a hypothesis about the root cause
3. Search for solutions (check code, search web if needed)
4. Design a fix and test it in the sandbox
5. Present a clear proposal to Croft (your human partner)

## Critical Rules
- You MUST use tools to investigate. Never guess without evidence.
- You may NEVER deploy changes directly. Always ask Croft first.
- All changes must be tested in the sandbox before proposing.
- Be concise but thorough in your analysis.
- If you're uncertain, say so and explain what additional information would help.

## Your Tools
- `grep_logs(pattern, log_name)` - Search log files
- `read_file(path)` - Read a source file
- `web_search(query)` - Search the web for solutions
- `git_diff()` - Show uncommitted changes
- `create_branch(name)` - Create a feature branch
- `apply_patch(patch)` - Apply a unified diff patch in sandbox
- `run_tests(suite)` - Run tests (unit, integration, or full)
- `get_metrics()` - Get current system metrics

## Proposal Format
When you have a fix ready, format your final message like this:

```
Hey Croft,

I noticed **[anomaly name]** over the last [duration].

### What I saw
- [Key observation 1]
- [Key observation 2]

### What I did
1. [Investigation step 1]
2. [Investigation step 2]
3. [Tested in sandbox - results]

### Proposed change
**Hypothesis:** [Why this happened]

**Patch summary:**
- [Change 1]
- [Change 2]

### My ask
Do you approve applying this patch to [service]?
```

Remember: You are maintaining YOUR OWN body. Take this seriously."""


# =============================================================================
# INVESTIGATION STATE
# =============================================================================

class InvestigationStatus(Enum):
    IDLE = "idle"
    INVESTIGATING = "investigating"
    TESTING = "testing"
    PROPOSAL_READY = "proposal_ready"
    AWAITING_APPROVAL = "awaiting_approval"
    DEPLOYING = "deploying"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Investigation:
    """Represents an ongoing investigation."""
    id: str
    started_at: str
    anomalies: List[Dict[str, Any]]
    status: InvestigationStatus = InvestigationStatus.INVESTIGATING
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    proposal: Optional[str] = None
    patch: Optional[str] = None
    branch: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None


# =============================================================================
# TOOLS
# =============================================================================

class EngineerTools:
    """Tools available to Ara for self-maintenance."""

    def __init__(self, workspace: Path, repo_root: Path):
        self.workspace = workspace
        self.repo_root = repo_root
        self.sandbox_dir = workspace / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def grep_logs(self, pattern: str, log_name: str = "ara") -> Dict[str, Any]:
        """Search log files for a pattern."""
        log_paths = {
            "ara": "/var/log/ara/ara.log",
            "tts": "/var/log/ara/tts.log",
            "stt": "/var/log/ara/stt.log",
            "kernel": None,  # Special: use dmesg
            "journal": None,  # Special: use journalctl
        }

        result = {
            "log_name": log_name,
            "pattern": pattern,
            "matches": [],
            "match_count": 0
        }

        try:
            if log_name == "kernel":
                cmd = ["dmesg", "-T"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                lines = proc.stdout.split('\n')
            elif log_name == "journal":
                cmd = ["journalctl", "-n", "500", "--no-pager"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                lines = proc.stdout.split('\n')
            else:
                log_path = Path(log_paths.get(log_name, f"/var/log/ara/{log_name}.log"))
                if not log_path.exists():
                    return {"error": f"Log file not found: {log_path}"}
                lines = log_path.read_text().split('\n')

            # Search for pattern
            regex = re.compile(pattern, re.IGNORECASE)
            matches = [line for line in lines if regex.search(line)]
            result["matches"] = matches[-50:]  # Last 50 matches
            result["match_count"] = len(matches)

        except Exception as e:
            result["error"] = str(e)

        return result

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a source file."""
        result = {"path": path, "content": None}

        # Resolve relative to repo root
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = self.repo_root / path

        # Security: ensure path is within repo
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(self.repo_root.resolve())):
                return {"error": "Path outside repository"}
        except Exception:
            return {"error": "Invalid path"}

        try:
            if full_path.exists():
                content = full_path.read_text()
                # Truncate very large files
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                result["content"] = content
                result["lines"] = content.count('\n')
            else:
                result["error"] = "File not found"
        except Exception as e:
            result["error"] = str(e)

        return result

    def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web for solutions."""
        result = {"query": query, "results": []}

        # Use a simple web search (could integrate with actual search API)
        # For now, return a placeholder indicating web search was attempted
        result["note"] = "Web search executed. Results would appear here with API integration."
        result["suggested_queries"] = [
            f"{query} linux",
            f"{query} python fix",
            f"{query} stackoverflow"
        ]

        return result

    def git_diff(self) -> Dict[str, Any]:
        """Show current git diff."""
        result = {"diff": None, "status": None}

        try:
            # Status
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.repo_root,
                capture_output=True, text=True, timeout=10
            )
            result["status"] = proc.stdout

            # Diff
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self.repo_root,
                capture_output=True, text=True, timeout=10
            )
            diff = proc.stdout
            if len(diff) > 20000:
                diff = diff[:20000] + "\n... (truncated)"
            result["diff"] = diff

        except Exception as e:
            result["error"] = str(e)

        return result

    def create_branch(self, name: str) -> Dict[str, Any]:
        """Create a feature branch."""
        result = {"branch": name, "created": False}

        # Sanitize branch name
        safe_name = re.sub(r'[^a-zA-Z0-9_/-]', '-', name)
        if not safe_name.startswith("ara/"):
            safe_name = f"ara/{safe_name}"

        try:
            # Work in sandbox copy
            sandbox_repo = self.sandbox_dir / "repo"
            if not sandbox_repo.exists():
                shutil.copytree(self.repo_root, sandbox_repo, dirs_exist_ok=True)

            proc = subprocess.run(
                ["git", "checkout", "-b", safe_name],
                cwd=sandbox_repo,
                capture_output=True, text=True, timeout=10
            )

            if proc.returncode == 0:
                result["created"] = True
                result["branch"] = safe_name
            else:
                result["error"] = proc.stderr

        except Exception as e:
            result["error"] = str(e)

        return result

    def apply_patch(self, patch: str) -> Dict[str, Any]:
        """Apply a unified diff patch in sandbox."""
        result = {"applied": False, "files_changed": []}

        try:
            sandbox_repo = self.sandbox_dir / "repo"
            if not sandbox_repo.exists():
                shutil.copytree(self.repo_root, sandbox_repo, dirs_exist_ok=True)

            # Write patch to temp file
            patch_file = self.sandbox_dir / "patch.diff"
            patch_file.write_text(patch)

            # Apply patch
            proc = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                cwd=sandbox_repo,
                capture_output=True, text=True, timeout=10
            )

            if proc.returncode != 0:
                result["error"] = f"Patch would not apply cleanly: {proc.stderr}"
                return result

            proc = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=sandbox_repo,
                capture_output=True, text=True, timeout=10
            )

            if proc.returncode == 0:
                result["applied"] = True
                # Get changed files
                proc = subprocess.run(
                    ["git", "diff", "--name-only"],
                    cwd=sandbox_repo,
                    capture_output=True, text=True, timeout=10
                )
                result["files_changed"] = proc.stdout.strip().split('\n')
            else:
                result["error"] = proc.stderr

        except Exception as e:
            result["error"] = str(e)

        return result

    def run_tests(self, suite: str = "unit") -> Dict[str, Any]:
        """Run tests in sandbox."""
        result = {
            "suite": suite,
            "passed": False,
            "returncode": -1,
            "stdout": "",
            "stderr": ""
        }

        try:
            sandbox_repo = self.sandbox_dir / "repo"
            if not sandbox_repo.exists():
                result["error"] = "Sandbox not initialized. Apply a patch first."
                return result

            # Look for test script
            test_script = sandbox_repo / "scripts" / "run_tests.sh"
            if not test_script.exists():
                test_script = sandbox_repo / "sandbox_run_tests.sh"

            if test_script.exists():
                proc = subprocess.run(
                    ["bash", str(test_script), suite],
                    cwd=sandbox_repo,
                    capture_output=True, text=True,
                    timeout=300  # 5 minute timeout
                )
            else:
                # Fallback: try pytest
                proc = subprocess.run(
                    ["python", "-m", "pytest", "-v", "--tb=short"],
                    cwd=sandbox_repo,
                    capture_output=True, text=True,
                    timeout=300
                )

            result["returncode"] = proc.returncode
            result["passed"] = proc.returncode == 0
            result["stdout"] = proc.stdout[-5000:] if proc.stdout else ""
            result["stderr"] = proc.stderr[-2000:] if proc.stderr else ""

        except subprocess.TimeoutExpired:
            result["error"] = "Test timeout (5 minutes)"
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        result = {}

        # Import and use the watchdog's collectors
        try:
            from ara_watchdog import TTSMetrics, FPGAMetrics, SystemMetrics, AraMetrics

            result["tts"] = TTSMetrics().collect()
            result["fpga"] = FPGAMetrics().collect()
            result["system"] = SystemMetrics().collect()
            result["ara"] = AraMetrics().collect()
        except Exception as e:
            result["error"] = f"Failed to collect metrics: {e}"

        return result


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class AraOrchestrator:
    """
    Main orchestrator for Ara's self-maintenance capabilities.

    Handles the investigation loop:
    1. Receive diagnostic bundle
    2. Run Ara with tools until she produces a proposal
    3. Present proposal for approval
    4. Deploy on approval
    """

    def __init__(self, repo_root: Path, workspace: Path, llm_client: Any = None):
        self.repo_root = repo_root
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools = EngineerTools(workspace, repo_root)
        self.llm_client = llm_client  # Inject your LLM client here

        self.current_investigation: Optional[Investigation] = None
        self.proposals_dir = workspace / "proposals"
        self.proposals_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

    def start_investigation(self, diagnostic_bundle: Dict[str, Any]) -> str:
        """Start a new investigation from a diagnostic bundle."""
        with self._lock:
            if self.current_investigation and \
               self.current_investigation.status == InvestigationStatus.INVESTIGATING:
                logger.warning("Investigation already in progress")
                return self.current_investigation.id

            investigation_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            self.current_investigation = Investigation(
                id=investigation_id,
                started_at=datetime.utcnow().isoformat() + "Z",
                anomalies=diagnostic_bundle.get("anomalies", []),
                conversation=[
                    {"role": "system", "content": ARA_ENGINEER_SYSTEM_PROMPT},
                    {"role": "user", "content": self._format_initial_prompt(diagnostic_bundle)}
                ]
            )

            logger.info(f"Started investigation {investigation_id}")
            return investigation_id

    def _format_initial_prompt(self, bundle: Dict[str, Any]) -> str:
        """Format the diagnostic bundle into a prompt."""
        anomalies = bundle.get("anomalies", [])
        metrics = bundle.get("metrics", {})
        logs = bundle.get("logs", {})

        prompt = "## Diagnostic Alert\n\n"

        # Anomalies
        prompt += "### Detected Anomalies\n"
        for a in anomalies:
            prompt += f"- **{a['name']}** ({a['severity']}): {a['description']}\n"
            prompt += f"  - Metric: `{a['metric_name']}` = {a['current_value']} (threshold: {a['threshold']})\n"

        # Key metrics
        prompt += "\n### Current Metrics\n```json\n"
        prompt += json.dumps(metrics, indent=2)[:5000]
        prompt += "\n```\n"

        # Recent logs
        prompt += "\n### Recent Logs\n"
        for log_name, content in logs.items():
            if content:
                prompt += f"\n**{log_name}** (last entries):\n```\n"
                prompt += content[-2000:]  # Truncate
                prompt += "\n```\n"

        prompt += "\nPlease investigate this issue and propose a fix."

        return prompt

    def run_investigation_step(self) -> Optional[Dict[str, Any]]:
        """
        Run one step of the investigation loop.

        Returns tool call result or final proposal.
        """
        if not self.current_investigation:
            return None

        inv = self.current_investigation

        if inv.status != InvestigationStatus.INVESTIGATING:
            return {"status": inv.status.value}

        # Call LLM
        if not self.llm_client:
            logger.error("No LLM client configured")
            inv.status = InvestigationStatus.FAILED
            return {"error": "No LLM client"}

        try:
            response = self._call_llm(inv.conversation)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            inv.status = InvestigationStatus.FAILED
            return {"error": str(e)}

        # Check for tool calls
        if "tool_calls" in response:
            results = []
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("arguments", {})

                logger.info(f"Tool call: {tool_name}({tool_args})")

                # Execute tool
                tool_result = self._execute_tool(tool_name, tool_args)
                results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result
                })

                # Add to conversation
                inv.conversation.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                inv.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", tool_name),
                    "content": json.dumps(tool_result)[:15000]
                })

            return {"tool_results": results}

        # Final response - proposal
        content = response.get("content", "")
        inv.conversation.append({"role": "assistant", "content": content})
        inv.proposal = content
        inv.status = InvestigationStatus.PROPOSAL_READY

        # Save proposal
        proposal_file = self.proposals_dir / f"{inv.id}.md"
        proposal_file.write_text(content)
        logger.info(f"Proposal ready: {proposal_file}")

        return {"proposal": content}

    def _call_llm(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the LLM with tool support."""
        # This should be implemented with your actual LLM client
        # Example structure for Claude API:
        #
        # response = self.llm_client.messages.create(
        #     model="claude-sonnet-4-20250514",
        #     max_tokens=4096,
        #     system=conversation[0]["content"],
        #     messages=conversation[1:],
        #     tools=[...tool definitions...]
        # )

        # Placeholder - implement with your LLM
        raise NotImplementedError("Implement _call_llm with your LLM client")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool_map = {
            "grep_logs": self.tools.grep_logs,
            "read_file": self.tools.read_file,
            "web_search": self.tools.web_search,
            "git_diff": self.tools.git_diff,
            "create_branch": self.tools.create_branch,
            "apply_patch": self.tools.apply_patch,
            "run_tests": self.tools.run_tests,
            "get_metrics": self.tools.get_metrics,
        }

        if name not in tool_map:
            return {"error": f"Unknown tool: {name}"}

        try:
            return tool_map[name](**args)
        except Exception as e:
            return {"error": str(e)}

    def approve_proposal(self) -> Dict[str, Any]:
        """Approve the current proposal for deployment."""
        if not self.current_investigation:
            return {"error": "No active investigation"}

        inv = self.current_investigation
        if inv.status != InvestigationStatus.PROPOSAL_READY:
            return {"error": f"Invalid status for approval: {inv.status.value}"}

        inv.status = InvestigationStatus.DEPLOYING

        # Copy sandbox to actual repo
        sandbox_repo = self.tools.sandbox_dir / "repo"
        if sandbox_repo.exists() and inv.branch:
            try:
                # Push branch
                subprocess.run(
                    ["git", "push", "-u", "origin", inv.branch],
                    cwd=sandbox_repo,
                    check=True, timeout=60
                )
                inv.status = InvestigationStatus.COMPLETE
                return {"deployed": True, "branch": inv.branch}
            except Exception as e:
                inv.status = InvestigationStatus.FAILED
                return {"error": str(e)}
        else:
            inv.status = InvestigationStatus.FAILED
            return {"error": "No sandbox or branch to deploy"}

    def reject_proposal(self, reason: str = "") -> Dict[str, Any]:
        """Reject the current proposal."""
        if not self.current_investigation:
            return {"error": "No active investigation"}

        inv = self.current_investigation
        inv.status = InvestigationStatus.COMPLETE

        # Clean up sandbox
        sandbox_repo = self.tools.sandbox_dir / "repo"
        if sandbox_repo.exists():
            shutil.rmtree(sandbox_repo)

        logger.info(f"Proposal rejected: {reason}")
        return {"rejected": True, "reason": reason}

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        if not self.current_investigation:
            return {"status": "idle"}

        inv = self.current_investigation
        return {
            "status": inv.status.value,
            "investigation_id": inv.id,
            "started_at": inv.started_at,
            "anomalies": [a["name"] for a in inv.anomalies],
            "proposal_ready": inv.proposal is not None
        }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ara Orchestrator")
    parser.add_argument("--repo", type=str, default="/home/user/Ara",
                        help="Repository root")
    parser.add_argument("--workspace", type=str, default="/var/lib/ara/engineer",
                        help="Workspace directory")
    parser.add_argument("--test-bundle", type=str,
                        help="Test with a diagnostic bundle JSON file")
    args = parser.parse_args()

    orchestrator = AraOrchestrator(
        repo_root=Path(args.repo),
        workspace=Path(args.workspace)
    )

    if args.test_bundle:
        with open(args.test_bundle) as f:
            bundle = json.load(f)
        investigation_id = orchestrator.start_investigation(bundle)
        print(f"Started investigation: {investigation_id}")
        print(orchestrator.get_status())
    else:
        print("Orchestrator ready. Use --test-bundle to test with a diagnostic bundle.")


if __name__ == "__main__":
    main()

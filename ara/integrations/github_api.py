"""
GitHub API Integration
=======================

Minimal GitHub client for the publishing pipeline.
Uses Git Trees API for efficient repo scanning.
"""

from __future__ import annotations

import os
import base64
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None
    _HAS_REQUESTS = False


GITHUB_API = "https://api.github.com"


class GitHubClient:
    """
    GitHub API client for the publishing pipeline.

    Features:
    - List repo files via Git Trees API
    - Create/update files
    - Create branches
    - Basic repo info
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")

        if not _HAS_REQUESTS:
            logger.warning("GitHubClient: requests not available. Install with: pip install requests")

    @property
    def is_available(self) -> bool:
        return _HAS_REQUESTS and self.token is not None

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # =========================================================================
    # Repo Info
    # =========================================================================

    def get_repo(self, owner: str, repo: str) -> Optional[Dict]:
        """Get repository info."""
        if not _HAS_REQUESTS:
            return None

        try:
            resp = requests.get(
                f"{GITHUB_API}/repos/{owner}/{repo}",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GitHubClient: get_repo failed: {e}")
            return None

    def get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch name."""
        repo_data = self.get_repo(owner, repo)
        if repo_data:
            return repo_data.get("default_branch", "main")
        return "main"

    def get_default_branch_sha(self, owner: str, repo: str) -> Optional[str]:
        """Get the SHA of the default branch HEAD."""
        if not _HAS_REQUESTS:
            return None

        try:
            default_branch = self.get_default_branch(owner, repo)
            resp = requests.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/refs/heads/{default_branch}",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["object"]["sha"]
        except Exception as e:
            logger.error(f"GitHubClient: get_default_branch_sha failed: {e}")
            return None

    # =========================================================================
    # Git Trees API
    # =========================================================================

    def get_tree_recursive(self, owner: str, repo: str, sha: str) -> List[Dict]:
        """Get the full repo tree recursively."""
        if not _HAS_REQUESTS:
            return []

        try:
            resp = requests.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}?recursive=1",
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("tree", [])
        except Exception as e:
            logger.error(f"GitHubClient: get_tree_recursive failed: {e}")
            return []

    def list_files(self, owner: str, repo: str) -> List[str]:
        """List all files in the repo."""
        sha = self.get_default_branch_sha(owner, repo)
        if not sha:
            return []

        tree = self.get_tree_recursive(owner, repo, sha)
        return [node["path"] for node in tree if node["type"] == "blob"]

    def list_dirs(self, owner: str, repo: str) -> List[str]:
        """List all directories in the repo."""
        sha = self.get_default_branch_sha(owner, repo)
        if not sha:
            return []

        tree = self.get_tree_recursive(owner, repo, sha)
        return [node["path"] for node in tree if node["type"] == "tree"]

    # =========================================================================
    # File Operations
    # =========================================================================

    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> Optional[str]:
        """Get file content from repo."""
        if not _HAS_REQUESTS:
            return None

        try:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
            if ref:
                url += f"?ref={ref}"

            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8")
            return data.get("content")
        except Exception as e:
            logger.error(f"GitHubClient: get_file_content failed: {e}")
            return None

    def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Create or update a file in the repo.

        Args:
            owner: Repo owner
            repo: Repo name
            path: File path in repo
            content: File content
            message: Commit message
            branch: Target branch (default: repo default)
            sha: Required for updates - the blob SHA of the file being replaced

        Returns:
            Response data or None on failure
        """
        if not _HAS_REQUESTS:
            return None

        try:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

            data = {
                "message": message,
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            }

            if branch:
                data["branch"] = branch
            if sha:
                data["sha"] = sha

            resp = requests.put(url, headers=self._headers(), json=data, timeout=30)
            resp.raise_for_status()

            logger.info(f"GitHubClient: created/updated {path} in {owner}/{repo}")
            return resp.json()
        except Exception as e:
            logger.error(f"GitHubClient: create_or_update_file failed: {e}")
            return None

    # =========================================================================
    # Branch Operations
    # =========================================================================

    def create_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        from_sha: Optional[str] = None,
    ) -> Optional[Dict]:
        """Create a new branch."""
        if not _HAS_REQUESTS:
            return None

        try:
            if not from_sha:
                from_sha = self.get_default_branch_sha(owner, repo)

            if not from_sha:
                return None

            resp = requests.post(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/refs",
                headers=self._headers(),
                json={
                    "ref": f"refs/heads/{branch_name}",
                    "sha": from_sha,
                },
                timeout=10,
            )
            resp.raise_for_status()

            logger.info(f"GitHubClient: created branch {branch_name} in {owner}/{repo}")
            return resp.json()
        except Exception as e:
            logger.error(f"GitHubClient: create_branch failed: {e}")
            return None


# =============================================================================
# Convenience
# =============================================================================

_default_client: Optional[GitHubClient] = None


def get_github_client() -> GitHubClient:
    """Get the default GitHub client."""
    global _default_client
    if _default_client is None:
        _default_client = GitHubClient()
    return _default_client


__all__ = [
    'GitHubClient',
    'get_github_client',
]

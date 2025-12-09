"""
Ara Integrations
=================

External service integrations for the publishing pipeline.

Available:
- github_api: GitHub REST API client (Trees, files, branches)
- twitter_api: (placeholder) Twitter/X API
- email_api: (placeholder) Email sending
"""

from .github_api import GitHubClient, get_github_client

__all__ = [
    'GitHubClient',
    'get_github_client',
]

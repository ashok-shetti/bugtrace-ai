"""
BugTrace AI - ETL Pipeline Module
Extracts, cleans, and formats GitHub bug issues and resolution comments for vector embedding.
"""

from .github_fetcher import GitHubIssueFetcher
from .text_cleaner import BugDataCleaner

__all__ = ["GitHubIssueFetcher", "BugDataCleaner"]

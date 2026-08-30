"""
GitHub Issue Fetcher Module for BugTrace AI.

Extracts closed bug issues and resolution comments from GitHub REST API
with rate-limit handling, pagination, pull request filtering, and bot filtering.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GitHubIssueFetcher")

# Known automated bot usernames or patterns
KNOWN_BOT_USERNAMES = {
    "github-actions",
    "github-actions[bot]",
    "dependabot",
    "dependabot[bot]",
    "stale",
    "stale[bot]",
    "codecov",
    "codecov[bot]",
    "snyk-bot",
    "pre-commit-ci",
    "pre-commit-ci[bot]",
    "linear-app",
    "linear-app[bot]",
    "mergify",
    "mergify[bot]",
    "vercel",
    "vercel[bot]",
    "netlify",
    "netlify[bot]",
}


class GitHubIssueFetcher:
    """
    Fetches closed bug issues and maintainer resolution comments from GitHub REST API.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        max_retries: int = 3,
        request_timeout: int = 30,
    ):
        """
        Initialize the GitHubIssueFetcher.

        Args:
            token: Optional GitHub Personal Access Token. If not provided, reads GITHUB_TOKEN from env.
            base_url: Base URL for GitHub API (default: https://api.github.com).
            max_retries: Number of retries for transient HTTP errors.
            request_timeout: Request timeout in seconds.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "BugTrace-AI-ETL/1.0",
            }
        )

        if self.token and self.token.strip() and not self.token.startswith("your_"):
            self.session.headers["Authorization"] = f"Bearer {self.token.strip()}"
            logger.info("GitHub API authentication configured with token.")
        else:
            logger.warning(
                "No valid GITHUB_TOKEN found. Requests will be unauthenticated with strict rate limits (60 req/hr). "
                "Set GITHUB_TOKEN in .env for higher rate limits (5,000 req/hr)."
            )

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """
        Check rate limit headers and wait if rate limit is exhausted.

        Args:
            response: Response object from requests.

        Returns:
            bool: True if rate limit was handled/slept, False if within normal limits.
        """
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")

        # Handle 403/429 rate limit exceeded responses
        if response.status_code in (403, 429):
            message = ""
            try:
                message = response.json().get("message", "")
            except Exception:
                pass

            if "rate limit" in message.lower() or "secondary rate limit" in message.lower() or remaining == "0":
                if reset_time:
                    sleep_duration = max(0, int(reset_time) - int(time.time())) + 2
                    logger.warning(
                        f"GitHub API rate limit exceeded! Waiting {sleep_duration} seconds until reset ({time.ctime(int(reset_time))})..."
                    )
                    time.sleep(sleep_duration)
                    return True
                else:
                    retry_after = response.headers.get("Retry-After", "60")
                    sleep_duration = int(retry_after) + 2
                    logger.warning(
                        f"Secondary rate limit reached. Waiting {sleep_duration} seconds..."
                    )
                    time.sleep(sleep_duration)
                    return True

        if remaining is not None and int(remaining) <= 2:
            if reset_time:
                sleep_duration = max(0, int(reset_time) - int(time.time())) + 2
                logger.warning(
                    f"Low API rate limit remaining ({remaining}). Waiting {sleep_duration} seconds until reset..."
                )
                time.sleep(sleep_duration)
                return True

        return False

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Make a robust HTTP request with rate limit handling and exponential backoff retry.

        Args:
            url: Full URL or endpoint path.
            params: Optional query parameters.

        Returns:
            requests.Response object.
        """
        full_url = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    full_url,
                    params=params,
                    timeout=self.request_timeout,
                )

                # Check if rate limit was hit
                if self._handle_rate_limit(response):
                    # Retry after sleeping
                    continue

                if response.status_code == 200:
                    return response

                # Handle 5xx server errors with backoff
                if 500 <= response.status_code < 600:
                    logger.warning(
                        f"GitHub API server error {response.status_code} (Attempt {attempt}/{self.max_retries}). Retrying..."
                    )
                    time.sleep(2**attempt)
                    continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request failed (Attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError(f"Failed to fetch {full_url} after {self.max_retries} attempts.")

    def is_bot_user(self, user_dict: Optional[Dict[str, Any]]) -> bool:
        """
        Detect if a GitHub user is an automated bot.

        Args:
            user_dict: User dictionary from GitHub API issue/comment.

        Returns:
            bool: True if bot, False otherwise.
        """
        if not user_dict:
            return False

        user_type = (user_dict.get("type") or "").lower()
        if user_type == "bot":
            return True

        login = (user_dict.get("login") or "").lower()
        if login in KNOWN_BOT_USERNAMES or login.endswith("[bot]"):
            return True

        if login.endswith("-bot") or login.startswith("bot-"):
            return True

        return False

    def is_bot_comment(self, comment: Dict[str, Any]) -> bool:
        """
        Detect if a comment is generated by a bot or automated workflow.

        Args:
            comment: Comment dictionary from GitHub API.

        Returns:
            bool: True if bot comment, False otherwise.
        """
        user = comment.get("user")
        if self.is_bot_user(user):
            return True

        body = (comment.get("body") or "").strip().lower()
        
        # Check automated stale bot or PR bot messages
        automated_phrases = [
            "this issue has been automatically marked as stale",
            "this issue has been automatically closed",
            "this issue was closed because it has been stalled",
            "closing this issue due to inactivity",
            "thank you for opening this issue",
            "welcome to the repository!",
            "first-time contributor",
            "all tests have passed",
            "coverage remained the same",
            "codecov report",
        ]
        for phrase in automated_phrases:
            if phrase in body:
                return True

        return False

    def fetch_comments_for_issue(
        self,
        comments_url: str,
        issue_author: str = "",
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Fetch and filter comments for an issue, identifying the most relevant resolution comment.

        Args:
            comments_url: GitHub API URL for issue comments.
            issue_author: Username of the original issue author.

        Returns:
            Tuple of (all_filtered_comments, best_resolution_comment).
        """
        try:
            response = self._make_request(comments_url, params={"per_page": 100})
            raw_comments = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch comments from {comments_url}: {e}")
            return [], None

        if not isinstance(raw_comments, list):
            return [], None

        filtered_comments: List[Dict[str, Any]] = []
        maintainer_comments: List[Dict[str, Any]] = []
        author_comments: List[Dict[str, Any]] = []
        other_comments: List[Dict[str, Any]] = []

        for c in raw_comments:
            if self.is_bot_comment(c):
                continue

            user_dict = c.get("user") or {}
            username = user_dict.get("login", "anonymous")
            author_assoc = c.get("author_association", "NONE")
            body = (c.get("body") or "").strip()

            if not body:
                continue

            cleaned_comment = {
                "id": c.get("id"),
                "user": username,
                "user_type": user_dict.get("type", "User"),
                "author_association": author_assoc,
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "body": body,
            }
            filtered_comments.append(cleaned_comment)

            # Categorize by maintainer authority
            if author_assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
                maintainer_comments.append(cleaned_comment)
            elif author_assoc == "AUTHOR" or username == issue_author:
                author_comments.append(cleaned_comment)
            elif author_assoc in ("CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR"):
                other_comments.append(cleaned_comment)
            else:
                other_comments.append(cleaned_comment)

        # Resolution identification logic:
        # 1. Prefer last/substantive maintainer comments (owners/members often explain the resolution/PR/workaround)
        # 2. If no maintainer comments, look at substantive closing comments from author or contributors
        # 3. Fallback to the last comment before closure
        best_resolution: Optional[Dict[str, Any]] = None

        if maintainer_comments:
            # Pick the latest substantive maintainer comment (or the one referencing fixes/commits)
            for c in reversed(maintainer_comments):
                body_lower = c["body"].lower()
                if any(
                    k in body_lower
                    for k in ["fix", "resolved", "merged", "closed in", "solved", "release", "commit", "workaround"]
                ):
                    best_resolution = c
                    break
            if not best_resolution:
                best_resolution = maintainer_comments[-1]
        elif author_comments:
            # Author often says "Thanks, updating to vX solved it" or similar
            for c in reversed(author_comments):
                body_lower = c["body"].lower()
                if any(k in body_lower for k in ["work", "solve", "fix", "close", "thanks", "figured"]):
                    best_resolution = c
                    break
            if not best_resolution and author_comments:
                best_resolution = author_comments[-1]
        elif other_comments:
            best_resolution = other_comments[-1]

        return filtered_comments, best_resolution

    def fetch_closed_issues(
        self,
        repo: str,
        max_issues: int = 200,
        labels: str = "bug",
        fetch_comments: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch closed bug issues from the target GitHub repository.

        Args:
            repo: Target repository in 'owner/repo' format (e.g. 'tiangolo/fastapi').
            max_issues: Maximum number of valid closed bug issues to extract.
            labels: GitHub labels filter (e.g. 'bug').
            fetch_comments: Whether to fetch and associate comments for each issue.

        Returns:
            List of structured raw issue dictionaries.
        """
        if "/" not in repo:
            raise ValueError(f"Repository must be formatted as 'owner/repo', got '{repo}'")

        owner, repo_name = repo.strip().split("/", 1)
        endpoint = f"repos/{owner}/{repo_name}/issues"

        extracted_issues: List[Dict[str, Any]] = []
        page = 1
        per_page = 100
        pr_skipped_count = 0
        total_raw_scanned = 0

        logger.info(
            f"Starting issue extraction from {repo} (target: {max_issues} closed bugs, label='{labels}')..."
        )

        while len(extracted_issues) < max_issues:
            params = {
                "state": "closed",
                "per_page": per_page,
                "page": page,
                "sort": "created",
                "direction": "desc",
            }
            if labels:
                params["labels"] = labels

            logger.info(f"Fetching page {page} for {repo}...")
            response = self._make_request(endpoint, params=params)
            issues_page = response.json()

            if not isinstance(issues_page, list) or len(issues_page) == 0:
                logger.info(f"No more issues found on page {page}. Stopping pagination.")
                break

            for issue in issues_page:
                total_raw_scanned += 1

                # CRITICAL: Filter out Pull Requests (GitHub issues API returns both issues and PRs)
                if "pull_request" in issue:
                    pr_skipped_count += 1
                    continue

                issue_number = issue.get("number")
                title = issue.get("title", "").strip()
                body = issue.get("body") or ""
                state = issue.get("state", "closed")
                created_at = issue.get("created_at")
                closed_at = issue.get("closed_at")
                html_url = issue.get("html_url", "")
                
                # Extract label names
                raw_labels = issue.get("labels", [])
                label_names = [
                    l["name"] if isinstance(l, dict) else str(l)
                    for l in raw_labels
                ]

                # Author metadata
                user_info = issue.get("user") or {}
                author = user_info.get("login", "unknown")
                author_assoc = issue.get("author_association", "NONE")

                # Skip if author was a bot
                if self.is_bot_user(user_info):
                    continue

                comments_count = issue.get("comments", 0)
                comments_list: List[Dict[str, Any]] = []
                resolution_comment_dict: Optional[Dict[str, Any]] = None

                # Extract comments if available
                if fetch_comments and comments_count > 0 and issue.get("comments_url"):
                    comments_list, resolution_comment_dict = self.fetch_comments_for_issue(
                        issue["comments_url"],
                        issue_author=author,
                    )

                raw_structured_issue = {
                    "id": issue.get("id"),
                    "issue_number": issue_number,
                    "repo": repo,
                    "title": title,
                    "body": body,
                    "state": state,
                    "labels": label_names,
                    "created_at": created_at,
                    "closed_at": closed_at,
                    "html_url": html_url,
                    "author": author,
                    "author_association": author_assoc,
                    "comments_count": comments_count,
                    "comments": comments_list,
                    "resolution_comment": resolution_comment_dict["body"] if resolution_comment_dict else None,
                    "resolution_metadata": resolution_comment_dict,
                }

                extracted_issues.append(raw_structured_issue)
                logger.debug(
                    f"Extracted Issue #{issue_number}: '{title[:40]}...' ({len(extracted_issues)}/{max_issues})"
                )

                if len(extracted_issues) >= max_issues:
                    break

            # Check next page Link header
            link_header = response.headers.get("Link", "")
            if 'rel="next"' not in link_header and len(issues_page) < per_page:
                logger.info("Reached end of available issues.")
                break

            page += 1

        logger.info(
            f"Extraction complete for {repo}: "
            f"Fetched {len(extracted_issues)} valid closed bugs (Scanned: {total_raw_scanned}, PRs skipped: {pr_skipped_count})."
        )
        return extracted_issues

    def save_raw_issues(
        self,
        issues: List[Dict[str, Any]],
        output_path: str = "data/raw/issues.json",
    ) -> Path:
        """
        Save raw structured issues to JSON file.

        Args:
            issues: List of raw issue dictionaries.
            output_path: Target JSON file path.

        Returns:
            Path object to saved file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(issues)} raw issues to {out_file.resolve()}")
        return out_file

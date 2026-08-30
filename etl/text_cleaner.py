"""
Text Cleaning and Normalization Module for BugTrace AI.

Cleans technical issue text, strips boilerplate and noisy markdown/HTML,
strictly preserves code snippets and stack traces, handles token limits,
and generates composite embedding-ready contexts.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("BugDataCleaner")


class BugDataCleaner:
    """
    Cleans and normalizes bug issue descriptions and resolution comments
    specifically tailored for developer code and hybrid search embeddings.
    """

    def __init__(self, max_tokens: int = 1500):
        """
        Initialize the BugDataCleaner.

        Args:
            max_tokens: Maximum target tokens for the composite context (default: 1500).
        """
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a given text string.
        Averages ~4 characters per token for English text and code.

        Args:
            text: Text to estimate.

        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        # Character-based approximation combined with word count for robust heuristic
        char_estimate = len(text) / 4.0
        word_estimate = len(text.split()) * 1.3
        return int(max(char_estimate, word_estimate))

    def strip_boilerplate_checklists(self, text: str) -> str:
        """
        Remove automated checklist templates, issue checkboxes, and template comments.

        Args:
            text: Raw markdown text.

        Returns:
            Cleaned text without checklist boilerplate.
        """
        if not text:
            return ""

        # Remove HTML/Markdown comments (<!-- ... -->)
        text = re.sub(r"<!--[\s\S]*?-->", "", text)

        # Remove checklist items like:
        # - [x] I have checked existing issues
        # - [ ] I have provided a minimal reproducible example
        text = re.sub(r"^\s*[-*]\s*\[[ xX]\]\s*.*$", "", text, flags=re.MULTILINE)

        # Remove standard template boilerplate lines
        boilerplate_patterns = [
            r"^\s*###\s*(?:Checklist|Operating System|Python Version|Confirmation).*$\n(?:^\s*[-*].*$\n)*",
            r"^\s*<!--.*?-->\s*$",
            r"^\s*Please describe your issue below:?\s*$",
            r"^\s*Replace this with your description\.?\s*$",
            r"^\s*<!--\s*DO NOT REMOVE THIS LINE\s*-->\s*$",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        return text

    def strip_html_tags(self, text: str) -> str:
        """
        Strip HTML tags while preserving text contents and line breaks.

        Args:
            text: Text with possible HTML tags.

        Returns:
            Plain text without HTML tags.
        """
        if not text:
            return ""

        # Convert <br>, <br/>, <p> to newlines
        text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<\s*/p\s*>", "\n\n", text, flags=re.IGNORECASE)

        # Inline tags can be stripped without extra spaces
        text = re.sub(r"<\s*/?(?:b|i|strong|em|small|span|font|a|code)\b[^>]*>", "", text, flags=re.IGNORECASE)

        # Block tags like details/summary/div/section replaced with space or newline
        text = re.sub(r"<\s*/?(?:details|summary|div|section|header|footer|blockquote)\b[^>]*>", " ", text, flags=re.IGNORECASE)

        # Remove <img> tags (including src/alt noise)
        text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)

        # Remove any remaining raw HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize line-by-line whitespace
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def normalize_special_chars_and_whitespace(self, text: str) -> str:
        """
        Normalize repeated dividers, special characters, and excessive whitespace.

        Args:
            text: Text to normalize.

        Returns:
            Cleaned normalized text.
        """
        if not text:
            return ""

        # Remove long markdown horizontal rules (---, ===, ___, ***)
        text = re.sub(r"^[=\-_*]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Remove carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace excessive multiple spaces (excluding indentation inside code)
        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip surrounding whitespace
        return text.strip()

    def clean_technical_text(self, text: str) -> str:
        """
        Clean technical text while strictly preserving code blocks (```) and stack traces.

        Args:
            text: Raw input issue body or comment.

        Returns:
            Sanitized, clean technical text.
        """
        if not text:
            return ""

        # Strategy: Protect fenced code blocks (```...```) from aggressive HTML/whitespace stripping
        code_blocks: List[str] = []

        def _save_code_block(match: re.Match) -> str:
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_content = match.group(0)
            # Truncate inner log lines if single block is absurdly long (> 3000 chars)
            if len(code_content) > 3000:
                lines = code_content.splitlines()
                if len(lines) > 60:
                    # Keep top 30 lines (error message/start) and bottom 25 lines (traceback root cause)
                    head = "\n".join(lines[:30])
                    tail = "\n".join(lines[-25:])
                    code_content = f"{head}\n\n... [verbose log output truncated] ...\n\n{tail}"
            code_blocks.append(code_content)
            return placeholder

        # Extract code blocks
        protected_text = re.sub(r"```[\s\S]*?```", _save_code_block, text)

        # Apply markdown and template cleaning on non-code parts
        cleaned = self.strip_boilerplate_checklists(protected_text)
        cleaned = self.strip_html_tags(cleaned)
        cleaned = self.normalize_special_chars_and_whitespace(cleaned)

        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            cleaned = cleaned.replace(f"__CODE_BLOCK_{i}__", code_block)

        return cleaned.strip()

    def truncate_to_token_limit(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text cleanly if it exceeds the maximum token limit.

        Args:
            text: Text to check and truncate.
            max_tokens: Override token limit, or uses self.max_tokens.

        Returns:
            Truncated text with ellipsis indicator if necessary.
        """
        limit = max_tokens or self.max_tokens
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= limit:
            return text

        # Rough target char limit based on max_tokens
        char_limit = limit * 4
        if len(text) > char_limit:
            # Cut at nearest paragraph or line break
            truncated = text[:char_limit]
            last_break = truncated.rfind("\n")
            if last_break > char_limit * 0.7:
                truncated = truncated[:last_break]
            return truncated.strip() + "\n\n... [Issue truncated to fit embedding context limit]"

        return text

    def clean_issue(self, raw_issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and format a single raw issue dictionary into an embedding-ready composite representation.

        Args:
            raw_issue: Dictionary of raw issue data from GitHubIssueFetcher.

        Returns:
            Processed issue dictionary with cleaned fields and composite text.
        """
        issue_number = raw_issue.get("issue_number") or raw_issue.get("number")
        title = (raw_issue.get("title") or "").strip()
        raw_body = raw_issue.get("body") or ""
        labels = raw_issue.get("labels") or []
        created_at = raw_issue.get("created_at")
        closed_at = raw_issue.get("closed_at")
        html_url = raw_issue.get("html_url") or ""
        repo = raw_issue.get("repo") or ""

        # Clean title & body
        cleaned_title = self.normalize_special_chars_and_whitespace(title)
        cleaned_body = self.clean_technical_text(raw_body)
        if not cleaned_body:
            cleaned_body = "No description provided."

        # Resolution comment
        resolution_raw = raw_issue.get("resolution_comment") or ""
        cleaned_resolution = self.clean_technical_text(resolution_raw)
        if not cleaned_resolution:
            # Fallback to checking comments list if available
            comments = raw_issue.get("comments") or []
            if comments:
                last_comment = comments[-1].get("body", "")
                cleaned_resolution = self.clean_technical_text(last_comment)

        if not cleaned_resolution:
            cleaned_resolution = "Closed without explicit resolution comment."

        # Format labels string
        labels_str = ", ".join(labels) if labels else "bug"

        # Generate composite context representation for hybrid embedding
        composite_text = (
            f"Title: {cleaned_title}\n"
            f"Labels: {labels_str}\n"
            f"Issue Description: {cleaned_body}\n"
            f"Resolution Fix: {cleaned_resolution}"
        )

        # Enforce token budget
        composite_text = self.truncate_to_token_limit(composite_text, self.max_tokens)
        token_count = self.estimate_tokens(composite_text)

        return {
            "issue_number": issue_number,
            "repo": repo,
            "title": cleaned_title,
            "cleaned_body": cleaned_body,
            "resolution_fix": cleaned_resolution,
            "labels": labels,
            "composite_text": composite_text,
            "token_count": token_count,
            "created_at": created_at,
            "closed_at": closed_at,
            "html_url": html_url,
        }

    def process_and_save(
        self,
        raw_issues_path: str = "data/raw/issues.json",
        output_path: str = "data/processed/cleaned_issues.json",
    ) -> Tuple[List[Dict[str, Any]], Path]:
        """
        Load raw issues from file, process them, and save clean structured outputs.

        Args:
            raw_issues_path: Path to input raw issues JSON.
            output_path: Path to target processed issues JSON.

        Returns:
            Tuple of (processed_issues_list, output_file_path).
        """
        in_file = Path(raw_issues_path)
        if not in_file.exists():
            raise FileNotFoundError(f"Raw issues file not found at: {in_file.resolve()}")

        with open(in_file, "r", encoding="utf-8") as f:
            raw_issues = json.load(f)

        if not isinstance(raw_issues, list):
            raise ValueError(f"Expected a JSON list in {in_file}, got {type(raw_issues)}")

        processed: List[Dict[str, Any]] = []
        for issue in raw_issues:
            cleaned = self.clean_issue(issue)
            # Skip if title or body is completely empty
            if cleaned["title"] or cleaned["cleaned_body"] != "No description provided.":
                processed.append(cleaned)

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Processed {len(processed)} issues (out of {len(raw_issues)} raw). Saved to {out_file.resolve()}"
        )
        return processed, out_file

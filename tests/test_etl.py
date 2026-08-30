"""
Unit tests for BugTrace AI ETL Pipeline.
Tests text cleaning, code/stack trace preservation, bot filtering, and composite generation.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from etl.github_fetcher import GitHubIssueFetcher
from etl.text_cleaner import BugDataCleaner


class TestBugDataCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = BugDataCleaner(max_tokens=1500)

    def test_strip_boilerplate_and_checklists(self):
        raw_text = (
            "<!-- Please read the guidelines before submitting -->\n"
            "### Checklist\n"
            "- [x] I have searched existing issues\n"
            "- [ ] I am using latest version\n"
            "\n"
            "This is the actual bug description about ValueError on startup."
        )
        cleaned = self.cleaner.strip_boilerplate_checklists(raw_text)
        self.assertNotIn("Please read the guidelines", cleaned)
        self.assertNotIn("I have searched existing issues", cleaned)
        self.assertIn("This is the actual bug description about ValueError on startup.", cleaned)

    def test_strip_html_tags_preserve_content(self):
        raw_text = (
            "<details><summary>Click to view logs</summary>\n"
            "<div>An error occurred: <b>ConnectionResetError</b></div>\n"
            "</details>"
        )
        cleaned = self.cleaner.strip_html_tags(raw_text)
        self.assertNotIn("<details>", cleaned)
        self.assertNotIn("</summary>", cleaned)
        self.assertNotIn("<div>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertIn("Click to view logs", cleaned)
        self.assertIn("An error occurred: ConnectionResetError", cleaned)

    def test_preserve_code_blocks_and_stack_traces(self):
        code_snippet = (
            "Here is the issue:\n"
            "```python\n"
            "def handle_request(req):\n"
            "    if not req.body:\n"
            "        raise ValueError('Empty body')\n"
            "    return req.process()\n"
            "```\n"
            "Stack trace:\n"
            "```text\n"
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 12, in handle_request\n"
            "ValueError: Empty body\n"
            "```"
        )
        cleaned = self.cleaner.clean_technical_text(code_snippet)
        # Ensure code blocks and stack traces are perfectly preserved
        self.assertIn("```python\ndef handle_request(req):", cleaned)
        self.assertIn("raise ValueError('Empty body')", cleaned)
        self.assertIn("Traceback (most recent call last):", cleaned)
        self.assertIn("ValueError: Empty body", cleaned)

    def test_composite_context_generation(self):
        mock_raw_issue = {
            "issue_number": 42,
            "repo": "tiangolo/fastapi",
            "title": "Bug in query parameter validation with Optional[List[str]]",
            "body": (
                "<!-- issue template -->\n"
                "- [x] I checked issues\n\n"
                "When passing query param as list, validation fails with 422.\n"
                "```python\n"
                "@app.get('/items')\n"
                "def get_items(q: Optional[List[str]] = Query(None)):\n"
                "    return q\n"
                "```"
            ),
            "labels": ["bug", "validation"],
            "created_at": "2024-01-01T00:00:00Z",
            "closed_at": "2024-01-02T00:00:00Z",
            "resolution_comment": "Resolved in PR #100. Updated fastapi/params.py to parse multiple query keys.",
        }

        cleaned = self.cleaner.clean_issue(mock_raw_issue)
        composite = cleaned["composite_text"]

        self.assertIn("Title: Bug in query parameter validation with Optional[List[str]]", composite)
        self.assertIn("Labels: bug, validation", composite)
        self.assertIn("Issue Description: When passing query param as list", composite)
        self.assertIn("```python\n@app.get('/items')", composite)
        self.assertIn("Resolution Fix: Resolved in PR #100", composite)
        self.assertGreater(cleaned["token_count"], 0)

    def test_process_and_save(self):
        mock_issues = [
            {
                "issue_number": 1,
                "title": "Fix crash on invalid input",
                "body": "App crashes with null input.",
                "labels": ["bug"],
                "resolution_comment": "Fixed by null check in commit abc1234.",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.json"
            processed_path = Path(tmp_dir) / "processed.json"

            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(mock_issues, f)

            processed, out_file = self.cleaner.process_and_save(
                raw_issues_path=str(raw_path),
                output_path=str(processed_path),
            )

            self.assertEqual(len(processed), 1)
            self.assertTrue(out_file.exists())
            self.assertEqual(processed[0]["issue_number"], 1)
            self.assertIn("Title: Fix crash on invalid input", processed[0]["composite_text"])


    def test_log_truncation_preserves_head_and_tail(self):
        # Create a code block with > 60 lines and > 3000 chars
        long_log_lines = ["Traceback start line 1: Initializing request"]
        long_log_lines.extend([f"Middle log debug information line {i} with lots of filler text to increase length" for i in range(70)])
        long_log_lines.append("Traceback root cause: ZeroDivisionError: division by zero in core.py:99")
        long_log_text = "```python\n" + "\n".join(long_log_lines) + "\n```"

        cleaned = self.cleaner.clean_technical_text(long_log_text)
        self.assertIn("Traceback start line 1: Initializing request", cleaned)
        self.assertIn("ZeroDivisionError: division by zero in core.py:99", cleaned)
        self.assertIn("verbose log output truncated", cleaned)

    def test_token_truncation(self):
        text = "word " * 2000
        truncated = self.cleaner.truncate_to_token_limit(text, max_tokens=100)
        self.assertLess(self.cleaner.estimate_tokens(truncated), 150)
        self.assertIn("[Issue truncated to fit embedding context limit]", truncated)


class TestGitHubIssueFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = GitHubIssueFetcher(token="fake_token_for_tests")

    def test_bot_detection(self):
        self.assertTrue(self.fetcher.is_bot_user({"login": "dependabot[bot]", "type": "Bot"}))
        self.assertTrue(self.fetcher.is_bot_user({"login": "github-actions[bot]", "type": "User"}))
        self.assertTrue(self.fetcher.is_bot_user({"login": "stale[bot]", "type": "Bot"}))
        self.assertFalse(self.fetcher.is_bot_user({"login": "tiangolo", "type": "User"}))
        self.assertFalse(self.fetcher.is_bot_user({"login": "davidism", "type": "User"}))

    def test_bot_comment_detection(self):
        stale_comment = {
            "user": {"login": "stale[bot]", "type": "Bot"},
            "body": "This issue has been automatically marked as stale because it has not had recent activity.",
        }
        self.assertTrue(self.fetcher.is_bot_comment(stale_comment))

        human_comment = {
            "user": {"login": "maintainer", "type": "User"},
            "body": "Thanks for reporting, fixed in version 0.100.1!",
        }
        self.assertFalse(self.fetcher.is_bot_comment(human_comment))

    def test_save_raw_issues(self):
        mock_issues = [{"issue_number": 99, "title": "Test Issue"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "sub" / "issues.json"
            saved = self.fetcher.save_raw_issues(mock_issues, output_path=str(out_file))
            self.assertTrue(saved.exists())
            with open(saved, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data[0]["issue_number"], 99)

    def test_resolution_comment_prioritization(self):
        # Mock session get for comments
        comments_data = [
            {
                "id": 1,
                "user": {"login": "reporter", "type": "User"},
                "author_association": "AUTHOR",
                "body": "Any update on this issue?",
            },
            {
                "id": 2,
                "user": {"login": "github-actions[bot]", "type": "Bot"},
                "author_association": "NONE",
                "body": "This issue has been automatically marked as stale",
            },
            {
                "id": 3,
                "user": {"login": "core-dev", "type": "User"},
                "author_association": "MEMBER",
                "body": "This was resolved in commit abc1234 by fixing the type check in router.",
            },
        ]

        class MockResponse:
            status_code = 200
            headers = {}
            def json(self):
                return comments_data

        self.fetcher.session.get = lambda url, **kwargs: MockResponse()
        filtered, best = self.fetcher.fetch_comments_for_issue("http://mock-comments", issue_author="reporter")
        
        self.assertEqual(len(filtered), 2)  # bot comment was excluded
        self.assertIsNotNone(best)
        self.assertEqual(best["user"], "core-dev")
        self.assertIn("resolved in commit abc1234", best["body"])


if __name__ == "__main__":
    unittest.main()

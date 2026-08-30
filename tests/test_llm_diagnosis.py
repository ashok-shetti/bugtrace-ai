"""
Unit tests for BugTrace AI LLM Inference and Prompt Assembly Layer.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.engine import BugTraceEngine, diagnose_bug
from core.llm_prompt import BugDiagnoser
from core.schemas import BugDiagnosisResponse, DiagnosisResult, ReferencedBug


class TestBugDiagnosisSchemas(unittest.TestCase):
    def test_schema_instantiation_and_validation(self):
        ref = ReferencedBug(
            issue_number=1234,
            relevance_reason="Contains the exact same type exception in route decorator",
        )
        self.assertEqual(ref.issue_number, 1234)

        diag = BugDiagnosisResponse(
            summary="TypeError in errorhandler registration.",
            root_cause_analysis="The decorator signature restricted exception handlers to specific subtypes.",
            recommended_fix="Relax callable parameter type to Any in error handler decorator.",
            referenced_issues=[ref],
            confidence_score=0.92,
        )

        data = diag.model_dump()
        self.assertEqual(data["confidence_score"], 0.92)
        self.assertEqual(len(data["referenced_issues"]), 1)
        self.assertEqual(data["referenced_issues"][0]["issue_number"], 1234)


class TestBugDiagnoser(unittest.TestCase):
    def setUp(self):
        self.diagnoser = BugDiagnoser(api_key="fake_key_for_testing")

    def test_format_context(self):
        retrieved_bugs = [
            {
                "issue_number": 4355,
                "title": "Async/await read file error: greenlet.error",
                "labels": ["bug"],
                "body": "Detailed stack trace showing greenlet switch error.",
            },
            {
                "issue_number": 4295,
                "title": "error handler type check fails",
                "labels": ["bug", "typing"],
                "body": "mypy reports incompatible type ErrorHandlerCallable.",
            },
        ]

        formatted = self.diagnoser.format_context(retrieved_bugs)
        self.assertIn("[HISTORICAL BUG #4355]: Async/await read file error", formatted)
        self.assertIn("Labels: bug", formatted)
        self.assertIn("Detailed stack trace showing greenlet", formatted)
        self.assertIn("[HISTORICAL BUG #4295]: error handler type check fails", formatted)
        self.assertIn("Labels: bug, typing", formatted)

    def test_format_empty_context(self):
        formatted = self.diagnoser.format_context([])
        self.assertIn("No relevant historical issues were found", formatted)

    def test_diagnose_mock_mode(self):
        retrieved_bugs = [
            {
                "issue_number": 100,
                "title": "FastAPI request validation error with missing field",
                "labels": ["bug"],
                "body": "Resolution: Use Field(default=...) or Optional type.",
                "rank_vec": 1,
                "rank_text": 1,
            }
        ]

        response = self.diagnoser.diagnose(
            query="ValidationError: missing field username",
            retrieved_bugs=retrieved_bugs,
            mock_mode=True,
        )

        self.assertIsInstance(response, BugDiagnosisResponse)
        self.assertIn("100", response.summary)
        self.assertGreater(len(response.referenced_issues), 0)
        self.assertEqual(response.referenced_issues[0].issue_number, 100)
        self.assertGreaterEqual(response.confidence_score, 0.8)

    def test_diagnose_mock_gemini_api_call(self):
        mock_parsed_response = BugDiagnosisResponse(
            summary="Pydantic model validation failure due to missing field.",
            root_cause_analysis="FastAPI expects required body fields unless explicitly marked with default or Optional.",
            recommended_fix="Change `username: str` to `username: Optional[str] = None` or provide value in request body.",
            referenced_issues=[
                ReferencedBug(issue_number=100, relevance_reason="Same missing field error.")
            ],
            confidence_score=0.95,
        )

        mock_response = MagicMock()
        mock_response.text = mock_parsed_response.model_dump_json()

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client = MagicMock()
        mock_client.models = mock_models
        self.diagnoser._client = mock_client

        response = self.diagnoser.diagnose(
            query="FastAPI ValidationError: missing field 'username'",
            retrieved_bugs=[{"issue_number": 100, "title": "Field error", "body": "Fix..."}],
            mock_mode=False,
        )

        self.assertEqual(response.confidence_score, 0.95)
        self.assertIn("Pydantic model validation failure", response.summary)
        mock_models.generate_content.assert_called_once()


class TestBugTraceEngine(unittest.TestCase):
    def test_engine_orchestration(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "issue_number": 4355,
                "title": "greenlet thread error",
                "body": "Fix: use direct async view in Flask 2.0",
                "labels": ["bug"],
                "rank_vec": 1,
                "rank_text": 1,
                "rrf_score": 0.032,
            }
        ]

        mock_diagnoser = MagicMock()
        mock_diagnoser.diagnose.return_value = BugDiagnosisResponse(
            summary="Manual event loop conflict with Flask async view.",
            root_cause_analysis="Flask 2.0 already runs async def views in an event loop.",
            recommended_fix="Remove manual asyncio loop and await directly.",
            referenced_issues=[ReferencedBug(issue_number=4355, relevance_reason="Exact issue match.")],
            confidence_score=0.9,
        )

        engine = BugTraceEngine(retriever=mock_retriever, diagnoser=mock_diagnoser)
        result = engine.diagnose(
            query="greenlet.error: cannot switch to a different thread",
            top_k=2,
        )

        self.assertIsInstance(result, DiagnosisResult)
        self.assertEqual(len(result.retrieved_bugs), 1)
        self.assertEqual(result.diagnosis.referenced_issues[0].issue_number, 4355)
        mock_retriever.search.assert_called_once()
        mock_diagnoser.diagnose.assert_called_once()


if __name__ == "__main__":
    unittest.main()

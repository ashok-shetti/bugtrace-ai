"""
Unit and Integration tests for BugTrace AI FastAPI Application.
"""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from core.schemas import BugDiagnosisResponse, DiagnosisResult, ReferencedBug


class TestBugTraceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("BugTrace AI API", data["name"])
        self.assertIn("/health", data["health"])
        self.assertIn("X-Process-Time", response.headers)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("database_connected", data)
        self.assertIn("model_loaded", data)
        self.assertIn("version", data)

    def test_diagnose_validation_error_short_query(self):
        # Query must be >= 10 chars
        payload = {"query": "short"}
        response = self.client.post("/api/v1/diagnose", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_diagnose_mock_mode_success(self):
        payload = {
            "query": "FastAPI RequestValidationError on invalid JSON body input",
            "labels": ["bug"],
            "top_k": 2,
            "mock_llm": True,
        }
        response = self.client.post("/api/v1/diagnose", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("root_cause_analysis", data)
        self.assertIn("recommended_fix", data)
        self.assertIn("referenced_issues", data)
        self.assertIn("confidence_score", data)

    def test_search_validation_error(self):
        # Query must be >= 3 chars
        payload = {"query": "a"}
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_search_endpoint_success(self):
        payload = {
            "query": "greenlet.error cannot switch to a different thread",
            "top_k": 3,
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], payload["query"])
        self.assertIn("total_results", data)
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)

    def test_ingest_endpoint_background_trigger(self):
        payload = {
            "repo_owner": "tiangolo",
            "repo_name": "fastapi",
            "max_issues": 20,
            "label": "bug",
        }
        response = self.client.post("/api/v1/ingest", json=payload)
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], "accepted")
        self.assertIn("tiangolo/fastapi", data["message"])


if __name__ == "__main__":
    unittest.main()

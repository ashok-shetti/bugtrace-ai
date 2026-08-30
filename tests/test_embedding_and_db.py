"""
Unit tests for BugTrace AI Embedding Generator and Database Client.
"""

import math
import os
import unittest
from unittest.mock import MagicMock, patch

from core.embedding import EmbeddingGenerator
from core.db_client import DatabaseClient


class TestEmbeddingGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize model on CPU for tests
        cls.embedder = EmbeddingGenerator(device="cpu", normalize_embeddings=True)

    def test_embedding_dimensions(self):
        self.assertEqual(self.embedder.dimension, 384)

    def test_embed_query(self):
        query = "How to fix AttributeError in FastAPI dependency injection?"
        vector = self.embedder.embed_query(query)
        self.assertEqual(len(vector), 384)
        self.assertIsInstance(vector, list)
        self.assertIsInstance(vector[0], float)

        # Check unit L2 norm when normalized
        l2_norm = math.sqrt(sum(x * x for x in vector))
        self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_embed_empty_query(self):
        vector = self.embedder.embed_query("")
        self.assertEqual(len(vector), 384)
        self.assertEqual(sum(vector), 0.0)

    def test_embed_texts_batch(self):
        texts = [
            "Bug 1: ConnectionResetError on Linux server",
            "Bug 2: Typing error in Blueprint.errorhandler decorator",
            "Bug 3: TypeError in JSON response serializer",
        ]
        vectors = self.embedder.embed_texts(texts, batch_size=2, show_progress=False)
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            self.assertEqual(len(v), 384)
            l2_norm = math.sqrt(sum(x * x for x in v))
            self.assertAlmostEqual(l2_norm, 1.0, places=4)


class TestDatabaseClient(unittest.TestCase):
    def setUp(self):
        self.client = DatabaseClient(
            host="mockhost",
            port=5432,
            user="testuser",
            password="testpassword",
            dbname="testdb",
        )

    def test_conninfo_string(self):
        self.assertIn("host=mockhost", self.client.conninfo)
        self.assertIn("dbname=testdb", self.client.conninfo)
        self.assertIn("user=testuser", self.client.conninfo)

    def test_health_check_unreachable_db(self):
        health = self.client.health_check()
        self.assertEqual(health["status"], "unhealthy")
        self.assertFalse(health["connected"])
        self.assertIn("error", health)

    def test_batch_upsert_mock(self):
        mock_issues = [
            {
                "issue_number": 101,
                "title": "Bug in route handling",
                "composite_text": "Title: Bug in route handling\nResolution Fix: Solved",
                "state": "closed",
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "closed_at": "2024-01-02T00:00:00Z",
                "embedding": [0.1] * 384,
            }
        ]

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch.object(self.client, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            count = self.client.batch_upsert_issues(mock_issues)

            self.assertEqual(count, 1)
            mock_cur.execute.assert_called_once()
            mock_conn.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()

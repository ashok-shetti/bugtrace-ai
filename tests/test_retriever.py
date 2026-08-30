"""
Unit tests for BugTrace AI Hybrid Retriever and RRF calculation.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_db_client = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = [0.05] * 384

        self.retriever = HybridRetriever(
            db_client=self.mock_db_client,
            embedding_generator=self.mock_embedder,
        )

    def test_search_empty_query(self):
        results = self.retriever.search("")
        self.assertEqual(results, [])
        self.mock_embedder.embed_query.assert_not_called()

    def test_search_query_embedding_called(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        self.mock_db_client.get_connection.return_value.__enter__.return_value = mock_conn

        query = "TypeError: missing required argument"
        self.retriever.search(query, top_k=5)

        self.mock_embedder.embed_query.assert_called_once_with(query)
        mock_cur.execute.assert_called_once()

    def test_hybrid_search_results_formatting_and_rrf(self):
        mock_db_rows = [
            {
                "id": 1,
                "issue_number": 4355,
                "title": "Async/await read file error: greenlet.error",
                "body": "Detailed traceback...",
                "labels": ["bug"],
                "created_at": "2021-11-26T08:54:57Z",
                "closed_at": "2021-12-23T00:36:53Z",
                "rank_vec": 1,
                "rank_text": 1,
                "rrf_score": (1.0 / (60 + 1)) + (1.0 / (60 + 1)),
            },
            {
                "id": 2,
                "issue_number": 4295,
                "title": "error handler type check fails",
                "body": "mypy error in register_error_handlers",
                "labels": ["bug", "typing"],
                "created_at": "2021-10-06T21:01:05Z",
                "closed_at": "2021-11-15T21:37:50Z",
                "rank_vec": 2,
                "rank_text": None,
                "rrf_score": (1.0 / (60 + 2)),
            },
        ]

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = mock_db_rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        self.mock_db_client.get_connection.return_value.__enter__.return_value = mock_conn

        results = self.retriever.search("greenlet error in linux thread", top_k=2)

        self.assertEqual(len(results), 2)
        # Check first result
        self.assertEqual(results[0]["issue_number"], 4355)
        self.assertEqual(results[0]["rank_vec"], 1)
        self.assertEqual(results[0]["rank_text"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 2.0 / 61, places=5)

        # Check second result (matched only in vector branch)
        self.assertEqual(results[1]["issue_number"], 4295)
        self.assertEqual(results[1]["rank_vec"], 2)
        self.assertIsNone(results[1]["rank_text"])
        self.assertAlmostEqual(results[1]["rrf_score"], 1.0 / 62, places=5)

    def test_label_filtering_params(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        self.mock_db_client.get_connection.return_value.__enter__.return_value = mock_conn

        self.retriever.search("test query", labels=["bug", "security"])
        args, kwargs = mock_cur.execute.call_args
        sql, params = args
        self.assertEqual(params["label_filter"], ["bug", "security"])


if __name__ == "__main__":
    unittest.main()

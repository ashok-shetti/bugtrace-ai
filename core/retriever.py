"""
Hybrid Search Retriever with Reciprocal Rank Fusion (RRF) for BugTrace AI.

Combines dense vector similarity search (pgvector) and full-text keyword search (tsvector)
into a unified ranking using Reciprocal Rank Fusion inside a single PostgreSQL query.
"""

import logging
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from .db_client import DatabaseClient
from .embedding import EmbeddingGenerator

logger = logging.getLogger("HybridRetriever")


class HybridRetriever:
    """
    Executes production-grade Hybrid Search with Reciprocal Rank Fusion (RRF).
    """

    HYBRID_RRF_SQL = """
    WITH semantic_search AS (
        SELECT 
            id,
            issue_number,
            title,
            body,
            labels,
            created_at,
            closed_at,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_vector)s::vector) AS rank_vec
        FROM github_issues
        WHERE (%(label_filter)s::text[] IS NULL OR labels @> %(label_filter)s::text[])
        ORDER BY embedding <=> %(query_vector)s::vector
        LIMIT %(candidate_limit)s
    ),
    keyword_search AS (
        SELECT 
            id,
            issue_number,
            title,
            body,
            labels,
            created_at,
            closed_at,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(text_searchable_index_col, plainto_tsquery('english', %(query_text)s)) DESC) AS rank_text
        FROM github_issues
        WHERE 
            text_searchable_index_col @@ plainto_tsquery('english', %(query_text)s)
            AND (%(label_filter)s::text[] IS NULL OR labels @> %(label_filter)s::text[])
        ORDER BY rank_text
        LIMIT %(candidate_limit)s
    )
    SELECT 
        COALESCE(s.id, k.id) AS id,
        COALESCE(s.issue_number, k.issue_number) AS issue_number,
        COALESCE(s.title, k.title) AS title,
        COALESCE(s.body, k.body) AS body,
        COALESCE(s.labels, k.labels) AS labels,
        COALESCE(s.created_at, k.created_at) AS created_at,
        COALESCE(s.closed_at, k.closed_at) AS closed_at,
        s.rank_vec,
        k.rank_text,
        (COALESCE(1.0 / (%(rrf_k)s + s.rank_vec), 0.0) + 
         COALESCE(1.0 / (%(rrf_k)s + k.rank_text), 0.0)) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %(top_k)s;
    """

    SEMANTIC_SEARCH_SQL = """
    SELECT 
        id,
        issue_number,
        title,
        body,
        labels,
        created_at,
        closed_at,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_vector)s::vector) AS rank_vec,
        1.0 - (embedding <=> %(query_vector)s::vector) AS similarity_score
    FROM github_issues
    WHERE (%(label_filter)s::text[] IS NULL OR labels @> %(label_filter)s::text[])
    ORDER BY embedding <=> %(query_vector)s::vector
    LIMIT %(top_k)s;
    """

    KEYWORD_SEARCH_SQL = """
    SELECT 
        id,
        issue_number,
        title,
        body,
        labels,
        created_at,
        closed_at,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(text_searchable_index_col, plainto_tsquery('english', %(query_text)s)) DESC) AS rank_text,
        ts_rank_cd(text_searchable_index_col, plainto_tsquery('english', %(query_text)s)) AS rank_score
    FROM github_issues
    WHERE 
        text_searchable_index_col @@ plainto_tsquery('english', %(query_text)s)
        AND (%(label_filter)s::text[] IS NULL OR labels @> %(label_filter)s::text[])
    ORDER BY rank_score DESC
    LIMIT %(top_k)s;
    """

    def __init__(
        self,
        db_client: Optional[DatabaseClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the HybridRetriever.

        Args:
            db_client: Optional DatabaseClient instance.
            embedding_generator: Optional EmbeddingGenerator instance.
        """
        self.db_client = db_client or DatabaseClient()
        self.embedder = embedding_generator or EmbeddingGenerator()

    def search(
        self,
        query_str: str,
        top_k: int = 5,
        candidate_limit: int = 20,
        rrf_k: int = 60,
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute unified hybrid search using Reciprocal Rank Fusion.

        Args:
            query_str: Natural language query, error message, or stack trace.
            top_k: Number of final ranked results to return.
            candidate_limit: Number of top candidates retrieved from each search branch.
            rrf_k: Smoothing constant for RRF score (default: 60).
            labels: Optional list of label strings for relational filtering.

        Returns:
            List of result dicts with rank_vec, rank_text, and rrf_score.
        """
        query_text = (query_str or "").strip()
        if not query_text:
            return []

        # 1. Compute dense query embedding vector
        query_vector = self.embedder.embed_query(query_text)

        # 2. Format SQL params
        label_filter = labels if (labels and len(labels) > 0) else None

        params = {
            "query_vector": query_vector,
            "query_text": query_text,
            "candidate_limit": candidate_limit,
            "label_filter": label_filter,
            "rrf_k": rrf_k,
            "top_k": top_k,
        }

        # 3. Execute hybrid CTE query
        logger.info(
            f"Executing hybrid search for: '{query_text[:50]}...' "
            f"(top_k={top_k}, candidate_limit={candidate_limit}, rrf_k={rrf_k}, labels={labels})"
        )

        try:
            with self.db_client.get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(self.HYBRID_RRF_SQL, params)
                    results = cur.fetchall()
        except Exception as e:
            logger.warning(
                f"Database query failed during hybrid search ({e}). Returning 0 candidates."
            )
            return []

        # Format and return results
        formatted_results: List[Dict[str, Any]] = []
        for r in results:
            formatted_results.append(
                {
                    "id": r["id"],
                    "issue_number": r["issue_number"],
                    "title": r["title"],
                    "body": r["body"],
                    "labels": r["labels"] or [],
                    "created_at": r["created_at"],
                    "closed_at": r["closed_at"],
                    "rank_vec": r["rank_vec"],
                    "rank_text": r["rank_text"],
                    "rrf_score": float(r["rrf_score"]) if r["rrf_score"] is not None else 0.0,
                }
            )

        return formatted_results

    def semantic_search(
        self,
        query_str: str,
        top_k: int = 5,
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute vector-only semantic search.
        """
        query_text = (query_str or "").strip()
        if not query_text:
            return []

        query_vector = self.embedder.embed_query(query_text)
        label_filter = labels if (labels and len(labels) > 0) else None

        params = {
            "query_vector": query_vector,
            "label_filter": label_filter,
            "top_k": top_k,
        }

        with self.db_client.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(self.SEMANTIC_SEARCH_SQL, params)
                return [dict(r) for r in cur.fetchall()]

    def keyword_search(
        self,
        query_str: str,
        top_k: int = 5,
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute keyword-only full text search.
        """
        query_text = (query_str or "").strip()
        if not query_text:
            return []

        label_filter = labels if (labels and len(labels) > 0) else None

        params = {
            "query_text": query_text,
            "label_filter": label_filter,
            "top_k": top_k,
        }

        with self.db_client.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(self.KEYWORD_SEARCH_SQL, params)
                return [dict(r) for r in cur.fetchall()]

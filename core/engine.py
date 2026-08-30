"""
End-to-End Diagnostic Engine for BugTrace AI.

Coordinates Hybrid Search Retrieval and LLM Root-Cause Diagnosis into a unified pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

from .db_client import DatabaseClient
from .embedding import EmbeddingGenerator
from .llm_prompt import BugDiagnoser
from .retriever import HybridRetriever
from .schemas import BugDiagnosisResponse, DiagnosisResult

logger = logging.getLogger("BugTraceEngine")


class BugTraceEngine:
    """
    Main diagnostic engine pairing HybridRetriever with BugDiagnoser.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        diagnoser: Optional[BugDiagnoser] = None,
        db_client: Optional[DatabaseClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the BugTraceEngine.
        """
        self.db_client = db_client or DatabaseClient()
        self.embedder = embedding_generator or EmbeddingGenerator()
        self.retriever = retriever or HybridRetriever(
            db_client=self.db_client,
            embedding_generator=self.embedder,
        )
        self.diagnoser = diagnoser or BugDiagnoser()

    def diagnose(
        self,
        query: str,
        labels: Optional[List[str]] = None,
        top_k: int = 3,
        candidate_limit: int = 20,
        mock_llm: bool = False,
    ) -> DiagnosisResult:
        """
        Execute end-to-end diagnostic pipeline.

        1. Retrieve top matching historical bug issues using Hybrid Search (RRF).
        2. Format grounded context and query LLM with Structured Outputs.
        3. Return type-safe DiagnosisResult.

        Args:
            query: Bug description, stack trace, or error log.
            labels: Optional labels filter (e.g. ['bug', 'typing']).
            top_k: Number of historical issues to retrieve.
            candidate_limit: Candidate search depth.
            mock_llm: Force offline diagnosis mode.

        Returns:
            DiagnosisResult containing Pydantic BugDiagnosisResponse and retrieved bugs.
        """
        query_text = (query or "").strip()
        logger.info(f"Starting diagnosis for query: '{query_text[:50]}...'")

        # Step 1: Hybrid Retrieval
        retrieved_bugs = self.retriever.search(
            query_str=query_text,
            top_k=top_k,
            candidate_limit=candidate_limit,
            labels=labels,
        )

        logger.info(f"Retrieved {len(retrieved_bugs)} candidate issues.")

        # Step 2: LLM Diagnosis
        diagnosis = self.diagnoser.diagnose(
            query=query_text,
            retrieved_bugs=retrieved_bugs,
            mock_mode=mock_llm,
        )

        return DiagnosisResult(
            query=query_text,
            diagnosis=diagnosis,
            retrieved_bugs=retrieved_bugs,
        )


# Global singleton engine instance for quick function access
_global_engine: Optional[BugTraceEngine] = None


def get_engine() -> BugTraceEngine:
    """
    Get or create the global singleton BugTraceEngine instance.
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = BugTraceEngine()
    return _global_engine


def diagnose_bug(
    query: str,
    labels: Optional[List[str]] = None,
    top_k: int = 3,
    mock_llm: bool = False,
) -> BugDiagnosisResponse:
    """
    Convenience facade function to execute end-to-end bug diagnosis.

    Args:
        query: Bug description or stack trace.
        labels: Optional label filters.
        top_k: Top K historical issues to retrieve (default: 3).
        mock_llm: Whether to run in offline mock mode.

    Returns:
        BugDiagnosisResponse Pydantic model.
    """
    engine = get_engine()
    result = engine.diagnose(
        query=query,
        labels=labels,
        top_k=top_k,
        mock_llm=mock_llm,
    )
    return result.diagnosis

"""
FastAPI Route Handlers for BugTrace AI.

Provides endpoints for root-cause bug diagnosis, hybrid search exploration,
and asynchronous background data ingestion.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from core.engine import BugTraceEngine, get_engine
from core.schemas import BugDiagnosisResponse
from etl.github_fetcher import GitHubIssueFetcher
from etl.text_cleaner import BugDataCleaner
from .schemas import (
    DiagnoseRequest,
    HealthResponse,
    IngestResponse,
    IngestTriggerRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger("BugTraceAPI")
router = APIRouter(prefix="/api/v1", tags=["BugTrace AI"])


def run_background_ingestion(
    repo: str,
    max_issues: int,
    label: str,
    engine: BugTraceEngine,
):
    """
    Background worker task to extract, clean, embed, and ingest repository issues.
    """
    logger.info(f"Starting background ingestion for {repo} (max_issues={max_issues}, label={label})...")
    try:
        # Step 1: Fetch
        fetcher = GitHubIssueFetcher()
        raw_issues = fetcher.fetch_closed_issues(
            repo=repo,
            max_issues=max_issues,
            labels=label,
            fetch_comments=True,
        )
        if not raw_issues:
            logger.warning(f"No closed bugs fetched for {repo}.")
            return

        # Step 2: Clean
        cleaner = BugDataCleaner(max_tokens=1500)
        cleaned_issues = [cleaner.clean_issue(r) for r in raw_issues]

        # Step 3: Embed in batches
        texts = [c["composite_text"] for c in cleaned_issues]
        embeddings = engine.embedder.embed_texts(texts, batch_size=32, show_progress=False)

        for issue, emb in zip(cleaned_issues, embeddings):
            issue["embedding"] = emb

        # Step 4: Batch Upsert
        upserted_count = engine.db_client.batch_upsert_issues(cleaned_issues)
        logger.info(
            f"Background ingestion completed successfully for {repo}: {upserted_count} issues stored and indexed."
        )
    except Exception as e:
        logger.error(f"Background ingestion failed for {repo}: {e}", exc_info=True)


@router.post(
    "/diagnose",
    response_model=BugDiagnosisResponse,
    summary="Diagnose Bug and Recommend Fix",
    description="Performs hybrid search against repository history and invokes grounded LLM inference to diagnose root cause and recommend fixes.",
)
async def diagnose(
    request: DiagnoseRequest,
    engine: BugTraceEngine = Depends(get_engine),
) -> BugDiagnosisResponse:
    """
    Diagnose a bug report or stack trace.
    """
    try:
        result = engine.diagnose(
            query=request.query,
            labels=request.labels,
            top_k=request.top_k,
            mock_llm=request.mock_llm,
        )
        return result.diagnosis
    except Exception as e:
        logger.error(f"Diagnosis endpoint failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate bug diagnosis: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid Search Exploration",
    description="Executes parallel semantic vector search (pgvector) and keyword search (tsvector) combined via Reciprocal Rank Fusion without calling the LLM.",
)
async def search(
    request: SearchRequest,
    engine: BugTraceEngine = Depends(get_engine),
) -> SearchResponse:
    """
    Explore candidate historical bugs via hybrid RRF search.
    """
    try:
        raw_results = engine.retriever.search(
            query_str=request.query,
            top_k=request.top_k,
            candidate_limit=request.candidate_limit,
            labels=request.labels,
        )

        items = [
            SearchResultItem(
                issue_number=r["issue_number"],
                title=r["title"],
                body=r["body"],
                labels=r.get("labels", []),
                rank_vec=r.get("rank_vec"),
                rank_text=r.get("rank_text"),
                rrf_score=r.get("rrf_score", 0.0),
            )
            for r in raw_results
        ]

        return SearchResponse(
            query=request.query,
            total_results=len(items),
            results=items,
        )
    except Exception as e:
        logger.error(f"Search endpoint failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Asynchronous Repository Ingestion",
    description="Dispatches a background ETL job to extract closed bugs from GitHub, clean text, generate embeddings, and upsert records into PostgreSQL.",
)
async def trigger_ingestion(
    request: IngestTriggerRequest,
    background_tasks: BackgroundTasks,
    engine: BugTraceEngine = Depends(get_engine),
) -> IngestResponse:
    """
    Trigger background ingestion job.
    """
    repo = f"{request.repo_owner.strip()}/{request.repo_name.strip()}"
    background_tasks.add_task(
        run_background_ingestion,
        repo=repo,
        max_issues=request.max_issues,
        label=request.label,
        engine=engine,
    )

    return IngestResponse(
        status="accepted",
        issues_processed=0,
        message=f"Ingestion job scheduled in background for repository '{repo}' (target: {request.max_issues} issues).",
    )

"""
Pydantic Request & Response Schemas for BugTrace AI FastAPI Application.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import BugDiagnosisResponse, ReferencedBug


class DiagnoseRequest(BaseModel):
    """
    User payload for triggering an end-to-end bug diagnosis.
    """

    query: str = Field(
        ...,
        min_length=10,
        description="The error message, stack trace, or bug description to diagnose",
        json_schema_extra={
            "example": "FastAPI raises RequestValidationError on empty body with custom Pydantic validator"
        },
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional tag filters for relational search (e.g. ['database', 'auth'])",
        json_schema_extra={"example": ["validation", "bug"]},
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of historical issues to retrieve for context",
    )
    mock_llm: bool = Field(
        default=False,
        description="Force offline mock LLM mode for testing/demo environments without OpenAI credits",
    )


class SearchRequest(BaseModel):
    """
    Request payload for lightweight hybrid retrieval search.
    """

    query: str = Field(
        ...,
        min_length=3,
        description="Natural language query, error code, or stack trace",
        json_schema_extra={"example": "greenlet.error: cannot switch to a different thread"},
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional label filters",
        json_schema_extra={"example": ["bug"]},
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of ranked candidates to return",
    )
    candidate_limit: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Depth of search in each branch (vector/keyword) prior to RRF fusion",
    )


class SearchResultItem(BaseModel):
    """
    Individual matched issue returned by hybrid retrieval.
    """

    issue_number: int
    title: str
    body: str
    labels: List[str] = Field(default_factory=list)
    rank_vec: Optional[int] = None
    rank_text: Optional[int] = None
    rrf_score: float


class SearchResponse(BaseModel):
    """
    Response payload for hybrid search endpoint.
    """

    query: str
    total_results: int
    results: List[SearchResultItem]


class IngestTriggerRequest(BaseModel):
    """
    Request payload to trigger asynchronous repository extraction and ingestion.
    """

    repo_owner: str = Field(default="tiangolo", description="GitHub repository owner")
    repo_name: str = Field(default="fastapi", description="GitHub repository name")
    max_issues: int = Field(
        default=100,
        ge=10,
        le=2000,
        description="Maximum issues to fetch & ingest",
    )
    label: str = Field(default="bug", description="GitHub issue label to filter for")


class IngestResponse(BaseModel):
    """
    Immediate acknowledgement response for background ingestion trigger.
    """

    status: str
    issues_processed: int
    message: str


class HealthResponse(BaseModel):
    """
    Application health status diagnostics.
    """

    status: str
    database_connected: bool
    model_loaded: bool
    total_indexed_issues: int = 0
    pgvector_version: Optional[str] = None
    version: str = "1.0.0"

"""
Pydantic Schemas for BugTrace AI Structured Outputs and Diagnosis API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReferencedBug(BaseModel):
    """
    Metadata about a specific historical bug issue that contributed to the diagnosis.
    """

    issue_number: int = Field(
        description="The GitHub issue number cited from the retrieved context"
    )
    relevance_reason: str = Field(
        description="Why this specific historical issue is relevant to the new bug"
    )


class BugDiagnosisResponse(BaseModel):
    """
    Grounded, structured root-cause diagnosis produced by the LLM.
    """

    summary: str = Field(
        description="A concise 1-2 sentence high-level overview of the diagnosed problem"
    )
    root_cause_analysis: str = Field(
        description="Detailed technical breakdown of why this error happens based on historical fixes"
    )
    recommended_fix: str = Field(
        description="Actionable step-by-step resolution or code diff"
    )
    referenced_issues: List[ReferencedBug] = Field(
        description="List of retrieved historical issue numbers that contributed to this fix"
    )
    confidence_score: float = Field(
        description="Confidence between 0.0 and 1.0 based on context match quality"
    )


class DiagnosisResult(BaseModel):
    """
    Combined container packaging query, retrieved issues, and the structured LLM diagnosis.
    """

    query: str
    diagnosis: BugDiagnosisResponse
    retrieved_bugs: List[Dict[str, Any]] = Field(default_factory=list)

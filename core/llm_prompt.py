"""
LLM Prompt Assembly and Structured Inference Module for BugTrace AI.

Formats retrieved historical bug contexts, injects strict grounding system prompts,
and calls OpenAI Structured Outputs to produce type-safe BugDiagnosisResponse objects.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .schemas import BugDiagnosisResponse, ReferencedBug

# Load environment variables
load_dotenv()

logger = logging.getLogger("BugDiagnoser")


class BugDiagnoser:
    """
    Manages prompt formatting and LLM inference for grounded bug diagnosis.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You are a Principal Site Reliability & Debugging Engineer.
Analyze the user's reported bug/stack trace using ONLY the provided historical closed issues from this repository.
If the retrieved context does not provide sufficient evidence to resolve the issue, explicitly state that in the summary and lower the confidence score.
Do NOT invent proprietary library behaviors.
Provide actionable, code-level recommendations and clearly reference the historical issue numbers that support your diagnosis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ):
        """
        Initialize the BugDiagnoser.

        Args:
            api_key: Optional OpenAI API key (defaults to OPENAI_API_KEY from environment).
            model: OpenAI model name (default: gpt-4o-mini).
            temperature: Sampling temperature for inference (default: 0.1).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

        if not self.has_valid_api_key():
            logger.warning(
                "No valid OPENAI_API_KEY detected. Set OPENAI_API_KEY in .env for live LLM diagnosis."
            )

    def has_valid_api_key(self) -> bool:
        """
        Check if a non-placeholder OpenAI API key is present.
        """
        return bool(
            self.api_key
            and self.api_key.strip()
            and not self.api_key.startswith("your_")
            and self.api_key != "your_openai_api_key_here"
        )

    @property
    def client(self) -> OpenAI:
        """
        Lazy-initialize and return OpenAI client.
        """
        if self._client is None:
            if not self.has_valid_api_key():
                raise ValueError(
                    "Cannot initialize OpenAI client: OPENAI_API_KEY is missing or invalid in .env."
                )
            self._client = OpenAI(api_key=self.api_key.strip())
        return self._client

    @staticmethod
    def format_context(retrieved_bugs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved historical bug issues into a structured context string for the prompt.

        Args:
            retrieved_bugs: List of issue dictionaries from HybridRetriever.

        Returns:
            Formatted context string.
        """
        if not retrieved_bugs:
            return "No relevant historical issues were found in the database."

        context_blocks = []
        for bug in retrieved_bugs:
            issue_num = bug.get("issue_number", "Unknown")
            title = bug.get("title", "Untitled")
            labels = ", ".join(bug.get("labels", [])) if bug.get("labels") else "None"
            body = (bug.get("body") or "").strip()

            block = (
                f"[HISTORICAL BUG #{issue_num}]: {title}\n"
                f"Labels: {labels}\n"
                f"Context & Fix Details:\n"
                f"{body}\n"
                f"----------------------------------------"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def diagnose(
        self,
        query: str,
        retrieved_bugs: List[Dict[str, Any]],
        mock_mode: bool = False,
    ) -> BugDiagnosisResponse:
        """
        Generate a structured bug diagnosis using OpenAI Structured Outputs.

        Args:
            query: User's reported bug description or error stack trace.
            retrieved_bugs: Ranked candidate issues from HybridRetriever.
            mock_mode: If True, returns a deterministic offline mock response without calling OpenAI.

        Returns:
            BugDiagnosisResponse Pydantic model.
        """
        formatted_context = self.format_context(retrieved_bugs)

        if mock_mode or not self.has_valid_api_key():
            logger.info("Using offline diagnosis generation (mock mode or missing API key).")
            return self._generate_offline_diagnosis(query, retrieved_bugs)

        user_content = (
            f"USER REPORTED BUG / STACK TRACE:\n"
            f"{query.strip()}\n\n"
            f"RETRIEVED HISTORICAL REPOSITORY ISSUES:\n"
            f"{formatted_context}\n\n"
            f"Please diagnose the root cause and provide the recommended fix based strictly on the historical issues."
        )

        logger.info(
            f"Calling OpenAI Structured Outputs ({self.model}, temp={self.temperature}) for query: '{query[:40]}...'"
        )

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=BugDiagnosisResponse,
                temperature=self.temperature,
            )
            parsed_response: BugDiagnosisResponse = completion.choices[0].message.parsed
            return parsed_response

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Falling back to grounded rule-based summary.")
            return self._generate_offline_diagnosis(query, retrieved_bugs, error_note=str(e))

    def _generate_offline_diagnosis(
        self,
        query: str,
        retrieved_bugs: List[Dict[str, Any]],
        error_note: Optional[str] = None,
    ) -> BugDiagnosisResponse:
        """
        Deterministic rule-based diagnosis builder used for testing or when API key is not present.
        """
        if not retrieved_bugs:
            return BugDiagnosisResponse(
                summary="Insufficient historical context to determine root cause for this error.",
                root_cause_analysis="No relevant issues matching the provided error or keywords were found in the database index.",
                recommended_fix="Verify the repository database index or supply additional stack trace details.",
                referenced_issues=[],
                confidence_score=0.1,
            )

        top_bug = retrieved_bugs[0]
        top_num = top_bug.get("issue_number", 0)
        top_title = top_bug.get("title", "")
        top_body = top_bug.get("body", "")

        referenced = [
            ReferencedBug(
                issue_number=b.get("issue_number", 0),
                relevance_reason=f"Top candidate match (Rank V:{b.get('rank_vec')}, K:{b.get('rank_text')}) addressing '{b.get('title')}'",
            )
            for b in retrieved_bugs[:3]
            if b.get("issue_number")
        ]

        summary = f"Issue matches historical problem in #{top_num}: '{top_title}'."
        if error_note:
            summary += f" [Note: Offline diagnostic mode used: {error_note}]"

        root_cause = (
            f"Based on historical resolution in #{top_num}, the error occurs due to improper configuration or "
            f"type handling in the runtime execution flow."
        )

        recommended_fix = (
            f"Refer to the solution applied in #{top_num}. Ensure all parameters match the expected signatures and "
            f"update dependencies or handler decorators accordingly."
        )

        return BugDiagnosisResponse(
            summary=summary,
            root_cause_analysis=root_cause,
            recommended_fix=recommended_fix,
            referenced_issues=referenced,
            confidence_score=0.85 if len(retrieved_bugs) >= 1 else 0.4,
        )

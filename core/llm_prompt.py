"""
LLM Prompt Assembly and Structured Inference Module for BugTrace AI using Google Gemini.

Formats retrieved historical bug contexts, injects strict grounding system prompts,
and calls Gemini Structured Outputs to produce type-safe BugDiagnosisResponse objects.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schemas import BugDiagnosisResponse, ReferencedBug

# Load environment variables
load_dotenv()

logger = logging.getLogger("BugDiagnoser")


class BugDiagnoser:
    """
    Manages prompt formatting and LLM inference for grounded bug diagnosis with Google Gemini.
    """

    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    SYSTEM_INSTRUCTION = (
        "You are a Principal Site Reliability & Debugging Engineer. Analyze the user's reported error "
        "using ONLY the provided historical closed issues. If the retrieved context does not "
        "provide sufficient evidence, state that in the summary and set confidence low. "
        "Do NOT hallucinate fixes."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize the BugDiagnoser with Gemini client.

        Args:
            api_key: Optional Gemini API key (defaults to GEMINI_API_KEY from environment).
            model: Gemini model name (default: gemini-3.6-flash or GEMINI_MODEL env).
            temperature: Sampling temperature for inference (default: 0.1).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.temperature = temperature
        self._client: Optional[genai.Client] = None

        if not self.has_valid_api_key():
            logger.warning(
                "No valid GEMINI_API_KEY detected. Set GEMINI_API_KEY in .env for live LLM diagnosis."
            )

    def has_valid_api_key(self) -> bool:
        """
        Check if a non-placeholder Gemini API key is present.
        """
        return bool(
            self.api_key
            and self.api_key.strip()
            and not self.api_key.startswith("your_")
            and self.api_key != "your_free_google_ai_studio_api_key_here"
        )

    @property
    def client(self) -> genai.Client:
        """
        Lazy-initialize and return Gemini client.
        """
        if self._client is None:
            if not self.has_valid_api_key():
                raise ValueError(
                    "Cannot initialize Gemini client: GEMINI_API_KEY is missing or invalid in .env."
                )
            self._client = genai.Client(api_key=self.api_key.strip())
        return self._client

    @client.setter
    def client(self, client_instance: genai.Client):
        """
        Explicit client setter for mocking and dependency injection.
        """
        self._client = client_instance

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
                f"Details & Resolution:\n"
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
        Generate a structured bug diagnosis using Gemini Structured Outputs.

        Args:
            query: User's reported bug description or error stack trace.
            retrieved_bugs: Ranked candidate issues from HybridRetriever.
            mock_mode: If True, returns a deterministic offline mock response without calling Gemini.

        Returns:
            BugDiagnosisResponse Pydantic model.
        """
        formatted_context = self.format_context(retrieved_bugs)

        if mock_mode or not self.has_valid_api_key():
            logger.info("Using offline diagnosis generation (mock mode or missing API key).")
            return self._generate_offline_diagnosis(query, retrieved_bugs)

        prompt = f"### USER ERROR QUERY:\n{query.strip()}\n\n### RETRIEVED HISTORICAL ISSUES:\n{formatted_context}"

        models_to_try = [self.model]
        for fallback in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for current_model in models_to_try:
            try:
                logger.info(
                    f"Calling Gemini Structured Outputs ({current_model}, temp={self.temperature}) for query: '{query[:40]}...'"
                )
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_INSTRUCTION,
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=BugDiagnosisResponse,
                    ),
                )

                raw_text = response.text or ""
                # Strip markdown code fencing if Gemini wraps JSON in ```json ... ```
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text.strip())

                return BugDiagnosisResponse.model_validate_json(cleaned_text)

            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API call with '{current_model}' failed: {e}. Attempting next model...")

        logger.error(f"All Gemini models failed. Last error: {last_error}. Falling back to grounded rule-based summary.")
        return self._generate_offline_diagnosis(query, retrieved_bugs, error_note=str(last_error))

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

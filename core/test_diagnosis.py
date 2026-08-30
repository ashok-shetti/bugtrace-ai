"""
Standalone Verification Script for BugTrace AI Diagnostic Engine.

Runs end-to-end bug diagnosis: Hybrid Retrieval (RRF) -> Prompt Grounding -> Google Gemini Structured Outputs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Reconfigure stdout/stderr to utf-8 for Windows emoji compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db_client import DatabaseClient
from core.engine import BugTraceEngine
from core.schemas import DiagnosisResult


def print_banner():
    banner = """
======================================================================
  ____             _____                    _    ___ 
 |  _ \\           |_   _|                  | |  |_ _|
 | |_) |_   _  __ _ | |_ __ __ _  ___ ___  | |   | | 
 |  _ <| | | |/ _` || | '__/ _` |/ __/ _ \\ | |   | | 
 | |_) | |_| | (_| || | | | (_| | (_|  __/ | |___| | 
 |____/ \\__,_|\\__, ||_|_|  \\__,_|\\___\\___| |_____|___|
               __/ |                                  
              |___/      LLM STRUCTURED DIAGNOSIS VERIFICATION
======================================================================
"""
    print(banner)


def display_diagnosis_report(result: DiagnosisResult):
    """
    Format and display the complete structured diagnosis report.
    """
    diag = result.diagnosis
    bugs = result.retrieved_bugs

    print("\n" + "=" * 80)
    print("                     BUG DIAGNOSIS REPORT")
    print("=" * 80)
    print(f"\n[?] INPUT QUERY:\n    {result.query}\n")

    print("-" * 80)
    print(f"[1] RETRIEVED HISTORICAL REPOSITORY ISSUES ({len(bugs)} matched):")
    print("-" * 80)
    if not bugs:
        print("    [!] No historical issues found in database.")
    for idx, bug in enumerate(bugs, 1):
        num = bug.get("issue_number", "N/A")
        title = bug.get("title", "Untitled")
        labels = ", ".join(bug.get("labels", []))
        rrf_score = bug.get("rrf_score", 0.0)
        vec_r = bug.get("rank_vec", "None")
        kw_r = bug.get("rank_text", "None")
        print(f"    ({idx}) Issue #{num}: {title}")
        print(f"        Labels: [{labels}] | RRF Score: {rrf_score:.5f} (Vec: #{vec_r}, Kw: #{kw_r})")

    print("\n" + "-" * 80)
    print(f"[2] HIGH-LEVEL SUMMARY (Confidence: {diag.confidence_score * 100:.1f}%):")
    print("-" * 80)
    print(f"    {diag.summary}")

    print("\n" + "-" * 80)
    print("[3] TECHNICAL ROOT CAUSE ANALYSIS:")
    print("-" * 80)
    for line in diag.root_cause_analysis.splitlines():
        print(f"    {line}")

    print("\n" + "-" * 80)
    print("[4] RECOMMENDED ACTIONABLE FIX:")
    print("-" * 80)
    for line in diag.recommended_fix.splitlines():
        print(f"    {line}")

    print("\n" + "-" * 80)
    print("[5] CITED HISTORICAL REFERENCES:")
    print("-" * 80)
    if not diag.referenced_issues:
        print("    None cited.")
    for ref in diag.referenced_issues:
        print(f"    * Issue #{ref.issue_number}: {ref.relevance_reason}")

    print("\n" + "=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugTrace AI - End-to-End LLM Diagnosis Test Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default="FastAPI ValidationError: missing field 'username' in request body with custom Pydantic validator",
        help="Error message, bug description, or stack trace to diagnose",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional issue labels filter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of historical issues to retrieve for context",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Force offline deterministic mock mode (does not call Gemini API)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.6-flash",
        help="Gemini model identifier for Structured Outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    # Check DB status
    db_client = DatabaseClient()
    health = db_client.health_check()
    if not health["connected"]:
        print(f"[!] Warning: Database not connected ({health.get('error')}).")
        print("[!] Note: If database has no indexed issues, retrieval will yield 0 candidates.")

    engine = BugTraceEngine(db_client=db_client)
    engine.diagnoser.model = args.model

    print(f"[*] Initiating diagnosis for: \"{args.query}\"")
    result = engine.diagnose(
        query=args.query,
        labels=args.labels,
        top_k=args.top_k,
        mock_llm=args.mock_llm,
    )

    display_diagnosis_report(result)


if __name__ == "__main__":
    main()

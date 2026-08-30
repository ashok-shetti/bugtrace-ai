"""
Standalone Test and Verification Script for BugTrace AI Hybrid Retrieval.

Executes and compares semantic vector search, full-text keyword search,
and unified Reciprocal Rank Fusion (RRF) hybrid search.
"""

import argparse
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

from core.retriever import HybridRetriever
from core.db_client import DatabaseClient


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
              |___/      HYBRID RETRIEVAL & RRF VERIFICATION
======================================================================
"""
    print(banner)


def display_results(results: List[Dict[str, Any]], query: str, top_k: int):
    """
    Format and display search results with RRF scores and rankings.
    """
    print("\n" + "=" * 80)
    print(f"QUERY: \"{query}\"")
    print(f"Returned {len(results)} matches (requested top_k={top_k}):")
    print("=" * 80)

    if not results:
        print("  [!] No matching issues found in the database.")
        print("=" * 80 + "\n")
        return

    for idx, r in enumerate(results, 1):
        issue_num = r.get("issue_number", "N/A")
        title = r.get("title", "Untitled")
        labels = ", ".join(r.get("labels", []))
        rrf_score = r.get("rrf_score", 0.0)
        rank_vec = r.get("rank_vec")
        rank_text = r.get("rank_text")

        rank_vec_str = f"#{rank_vec}" if rank_vec is not None else "None"
        rank_text_str = f"#{rank_text}" if rank_text is not None else "None"

        body_snippet = (r.get("body") or "")[:200].replace("\n", " ").strip()

        print(f"\n[{idx}] Issue #{issue_num}: {title}")
        print(f"    - Labels:       [{labels}]")
        print(f"    - RRF Score:    {rrf_score:.5f}")
        print(f"    - Vector Rank:  {rank_vec_str}  |  Keyword Rank: {rank_text_str}")
        print(f"    - Preview:      {body_snippet}...")

    print("\n" + "=" * 80 + "\n")


def run_test_suite(retriever: HybridRetriever):
    """
    Run predefined benchmark test queries covering both exact code errors and conceptual queries.
    """
    sample_queries = [
        {
            "type": "EXACT ERROR / STACK TRACE",
            "query": "greenlet.error: cannot switch to a different thread",
            "description": "Exact stack trace keyword matching for async/await threads.",
        },
        {
            "type": "FUNCTION & DECORATOR BUG",
            "query": "Blueprint.errorhandler restricts type of result function which avoids chaining",
            "description": "Specific Python identifier and decorator type-checking error.",
        },
        {
            "type": "CONCEPTUAL / NATURAL LANGUAGE",
            "query": "How to fix mypy type checking failure when registering custom exception handlers",
            "description": "Semantic query describing a developer's high-level goal.",
        },
    ]

    print("\n[+] Running Predefined Hybrid Retrieval Benchmark Tests...\n")

    for item in sample_queries:
        print(f"[*] Benchmark [{item['type']}]:")
        print(f"    Description: {item['description']}")
        results = retriever.search(item["query"], top_k=3)
        display_results(results, item["query"], top_k=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugTrace AI - Hybrid Retrieval Search & Test Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom search query (natural language or code snippet)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Filter results by one or more labels (e.g. --labels bug typing)",
    )
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run the automated multi-query benchmark test suite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    # Verify database health
    db_client = DatabaseClient()
    health = db_client.health_check()

    if not health["connected"]:
        print(f"[!] Database connection failed: {health.get('error')}")
        print("[!] Make sure PostgreSQL with pgvector is running and .env is configured.")
        sys.exit(1)

    total_records = health.get("total_records", 0)
    print(f"[+] Connected to PostgreSQL database '{health['database']}' ({total_records} issues in index).")

    retriever = HybridRetriever(db_client=db_client)

    if args.run_suite or args.query is None:
        run_test_suite(retriever)
    else:
        results = retriever.search(
            query_str=args.query,
            top_k=args.top_k,
            labels=args.labels,
        )
        display_results(results, args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()

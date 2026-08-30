"""
CLI Ingestion Trigger for BugTrace AI ETL Pipeline.

Orchestrates fetching closed bug issues and resolution comments from GitHub,
cleaning and normalizing the technical data, and generating embedding-ready composites.
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure parent directory is in Python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.github_fetcher import GitHubIssueFetcher
from etl.text_cleaner import BugDataCleaner


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
              |___/      DATA EXTRACTION & PREPROCESSING PIPELINE
======================================================================
"""
    print(banner)


def print_summary_statistics(
    raw_issues: List[Dict[str, Any]],
    processed_issues: List[Dict[str, Any]],
    raw_path: str,
    processed_path: str,
    elapsed_time: float,
):
    """
    Print comprehensive statistics on the ETL execution.
    """
    print("\n" + "=" * 70)
    print("                    ETL PIPELINE SUMMARY REPORT")
    print("=" * 70)

    total_raw = len(raw_issues)
    total_processed = len(processed_issues)

    # Issues with resolutions
    with_resolutions = sum(
        1
        for issue in processed_issues
        if issue.get("resolution_fix")
        and issue["resolution_fix"] != "Closed without explicit resolution comment."
    )
    resolution_pct = (with_resolutions / total_processed * 100) if total_processed > 0 else 0.0

    # Token stats
    token_counts = [issue.get("token_count", 0) for issue in processed_issues]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    # Label statistics
    all_labels = []
    for issue in processed_issues:
        all_labels.extend(issue.get("labels", []))
    label_counter = Counter(all_labels).most_common(5)

    # File size metrics
    raw_size_kb = (
        Path(raw_path).stat().st_size / 1024.0 if Path(raw_path).exists() else 0.0
    )
    processed_size_kb = (
        Path(processed_path).stat().st_size / 1024.0
        if Path(processed_path).exists() else 0.0
    )

    print(f"[-] Execution Time:              {elapsed_time:.2f} seconds")
    print(f"[-] Total Raw Issues Fetched:     {total_raw}")
    print(f"[-] Total Cleaned Issues Ready:   {total_processed}")
    print(f"[-] Issues With Resolution Fixes: {with_resolutions} ({resolution_pct:.1f}%)")
    print(f"[-] Token Counts per Issue:")
    print(f"    * Average:                   {avg_tokens:.1f} tokens")
    print(f"    * Min:                       {min_tokens} tokens")
    print(f"    * Max:                       {max_tokens} tokens")
    print(f"[-] Top 5 Issue Labels:")
    for label, count in label_counter:
        print(f"    * '{label}': {count}")
    print(f"[-] Output Artifacts:")
    print(f"    * Raw Data:                  {raw_path} ({raw_size_kb:.2f} KB)")
    print(f"    * Processed Embeddings Data: {processed_path} ({processed_size_kb:.2f} KB)")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugTrace AI - GitHub Issue ETL Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="tiangolo/fastapi",
        help="Target GitHub repository in 'owner/repo' format (e.g. tiangolo/fastapi, pallets/flask)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=200,
        help="Maximum number of valid closed bug issues to extract",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="bug",
        help="GitHub issue label to filter for",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="data/raw/issues.json",
        help="Output filepath for raw extracted issues",
    )
    parser.add_argument(
        "--processed-output",
        type=str,
        default="data/processed/cleaned_issues.json",
        help="Output filepath for cleaned, embedding-ready processed issues",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub personal access token (overrides .env GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip fetching comments (faster, for testing/dry-runs)",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch raw issues from GitHub without running cleaner",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only run cleaner on already existing raw issues JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    start_time = time.time()
    raw_issues: List[Dict[str, Any]] = []
    processed_issues: List[Dict[str, Any]] = []

    # Step 1: Extraction Phase
    if not args.clean_only:
        print(f"[*] STEP 1/2: Extracting closed bugs from GitHub ({args.repo})...")
        fetcher = GitHubIssueFetcher(token=args.token)
        raw_issues = fetcher.fetch_closed_issues(
            repo=args.repo,
            max_issues=args.max_issues,
            labels=args.label,
            fetch_comments=not args.no_comments,
        )
        fetcher.save_raw_issues(raw_issues, output_path=args.raw_output)
    else:
        print(f"[*] Skipping extraction (--clean-only specified). Loading {args.raw_output}...")
        import json
        if Path(args.raw_output).exists():
            with open(args.raw_output, "r", encoding="utf-8") as f:
                raw_issues = json.load(f)
        else:
            print(f"[!] Error: Raw file {args.raw_output} not found for --clean-only.")
            sys.exit(1)

    if args.fetch_only:
        print(f"[*] Step 2 skipped (--fetch-only specified).")
        elapsed = time.time() - start_time
        print(f"[+] Done in {elapsed:.2f}s! Raw issues saved to {args.raw_output}")
        return

    # Step 2: Preprocessing & Cleaning Phase
    print(f"[*] STEP 2/2: Preprocessing technical text and generating composite contexts...")
    cleaner = BugDataCleaner(max_tokens=1500)
    processed_issues, _ = cleaner.process_and_save(
        raw_issues_path=args.raw_output,
        output_path=args.processed_output,
    )

    elapsed_time = time.time() - start_time

    # Step 3: Print Summary Statistics
    print_summary_statistics(
        raw_issues=raw_issues,
        processed_issues=processed_issues,
        raw_path=args.raw_output,
        processed_path=args.processed_output,
        elapsed_time=elapsed_time,
    )


if __name__ == "__main__":
    main()

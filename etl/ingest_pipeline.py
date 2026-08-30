"""
Database Ingestion Pipeline for BugTrace AI.

Loads processed bug issues, computes dense vector embeddings using sentence-transformers,
and batch-inserts both relational metadata and embeddings into PostgreSQL with pgvector.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db_client import DatabaseClient
from core.embedding import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngestPipeline")


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
              |___/      EMBEDDING & PGVECTOR INGESTION PIPELINE
======================================================================
"""
    print(banner)


def load_processed_issues(file_path: str) -> List[Dict[str, Any]]:
    """
    Load cleaned issue records from processed JSON file.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Processed issues file not found at: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of issues in {p}, got {type(records)}")

    logger.info(f"Loaded {len(records)} cleaned issue records from {p.resolve()}")
    return records


def run_ingestion(
    input_path: str = "data/processed/cleaned_issues.json",
    batch_size: int = 32,
    init_db: bool = False,
    dry_run: bool = False,
    device: str = None,
):
    """
    Execute the embedding generation and database ingestion workflow.
    """
    start_time = time.time()

    # 1. Load Cleaned Records
    records = load_processed_issues(input_path)
    if not records:
        logger.warning("No records to ingest. Exiting.")
        return

    # 2. Initialize Database Client
    db_client = None
    if not dry_run:
        db_client = DatabaseClient()
        health = db_client.health_check()
        logger.info(f"Database health check: {health}")

        if not health["connected"]:
            logger.error(
                f"Cannot connect to PostgreSQL database '{db_client.dbname}' at {db_client.host}:{db_client.port}. "
                "Ensure PostgreSQL with pgvector is running (e.g. via docker-compose up -d) and .env is configured."
            )
            sys.exit(1)

        if init_db or not health.get("github_issues_table_exists"):
            logger.info("Initializing database schema...")
            db_client.init_db()

    # 3. Initialize Embedding Generator
    logger.info(f"Initializing EmbeddingGenerator (device={device or 'auto'})...")
    embedder = EmbeddingGenerator(device=device)

    # 4. Process in Batches
    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size
    total_inserted = 0

    print(f"\n[*] Processing {total_records} records in {total_batches} batches (batch_size={batch_size})...\n")

    for i in range(0, total_records, batch_size):
        batch_num = (i // batch_size) + 1
        batch_records = records[i : i + batch_size]

        # Extract text to embed: composite_text or fallback
        texts_to_embed = [
            r.get("composite_text")
            or f"Title: {r.get('title', '')}\nIssue Description: {r.get('cleaned_body', '')}\nResolution Fix: {r.get('resolution_fix', '')}"
            for r in batch_records
        ]

        # Generate dense embeddings for current batch
        embeddings = embedder.embed_texts(texts_to_embed, batch_size=batch_size, show_progress=False)

        # Attach embeddings to records
        for record, emb in zip(batch_records, embeddings):
            record["embedding"] = emb

        if not dry_run and db_client:
            upserted = db_client.batch_upsert_issues(batch_records)
            total_inserted += upserted
            print(
                f"[-] Batch {batch_num:02d}/{total_batches:02d}: "
                f"Generated {len(embeddings)} embeddings -> Upserted {upserted} records into PostgreSQL."
            )
        else:
            total_inserted += len(batch_records)
            print(
                f"[-] [DRY-RUN] Batch {batch_num:02d}/{total_batches:02d}: "
                f"Generated {len(embeddings)} embeddings (dim: {len(embeddings[0]) if embeddings else 0}) [Simulated]"
            )

    elapsed = time.time() - start_time

    # 5. Summary & Verification
    print("\n" + "=" * 70)
    print("                 INGESTION PIPELINE SUMMARY REPORT")
    print("=" * 70)
    print(f"[-] Execution Time:              {elapsed:.2f} seconds")
    print(f"[-] Total Issues Processed:      {total_records}")
    print(f"[-] Total Issues Ingested:       {total_inserted}")
    print(f"[-] Embedding Dimension:         {embedder.dimension} (all-MiniLM-L6-v2)")

    if not dry_run and db_client:
        total_indexed = db_client.get_indexed_issue_count()
        db_stats = db_client.get_issue_stats()
        print(f"[-] Total DB Vector-Indexed:     {total_indexed} rows")
        print(f"[-] DB Issue Date Range:         {db_stats.get('earliest_issue')} -> {db_stats.get('latest_issue')}")
        db_client.close()
    else:
        print(f"[-] Database Upsert Status:      DRY RUN (No changes committed)")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugTrace AI - Vector Embedding & Database Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/cleaned_issues.json",
        help="Path to cleaned issues JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation and DB upsert",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database schema before ingestion",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify database connectivity and check stored vector statistics only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute embeddings and validate workflow without inserting into database",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hardware device to use ('cuda', 'cpu', 'mps')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    if args.verify_only:
        client = DatabaseClient()
        health = client.health_check()
        print(json.dumps(health, indent=2, default=str))
        if health.get("github_issues_table_exists"):
            print("\nDatabase Statistics:")
            print(json.dumps(client.get_issue_stats(), indent=2, default=str))
        client.close()
        return

    run_ingestion(
        input_path=args.input,
        batch_size=args.batch_size,
        init_db=args.init_db,
        dry_run=args.dry_run,
        device=args.device,
    )


if __name__ == "__main__":
    main()

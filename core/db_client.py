"""
Database Client & Connection Manager for BugTrace AI.

Manages PostgreSQL connection pooling with psycopg3 and pgvector adapter registration,
schema migrations, idempotent batch upserts, and health diagnostics.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("DatabaseClient")


class DatabaseClient:
    """
    Manages PostgreSQL connections, pgvector types, and issue table operations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
        connect_timeout: int = 5,
    ):
        """
        Initialize the DatabaseClient.

        Reads defaults from environment variables (.env).
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = int(port or os.getenv("DB_PORT", "5432"))
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "postgres")
        self.dbname = dbname or os.getenv("DB_NAME", "bugtrace_db")
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connect_timeout = connect_timeout

        self.conninfo = (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname} connect_timeout={self.connect_timeout}"
        )

        self._pool: Optional[ConnectionPool] = None

    def _configure_connection(self, conn: psycopg.Connection):
        """
        Callback executed for every connection created in the pool.
        Registers the pgvector adapter.
        """
        register_vector(conn)

    @property
    def pool(self) -> ConnectionPool:
        """
        Lazy-initialize and return the connection pool.
        """
        if self._pool is None:
            logger.info(
                f"Initializing PostgreSQL connection pool to {self.host}:{self.port}/{self.dbname} "
                f"(min={self.min_connections}, max={self.max_connections})..."
            )
            self._pool = ConnectionPool(
                conninfo=self.conninfo,
                min_size=self.min_connections,
                max_size=self.max_connections,
                configure=self._configure_connection,
                open=True,
            )
        return self._pool

    def close(self):
        """
        Close all connections in the pool.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed.")

    @contextmanager
    def get_connection(self) -> Generator[psycopg.Connection, None, None]:
        """
        Context manager to acquire a connection from the pool and return it.
        """
        with self.pool.connection(timeout=float(self.connect_timeout)) as conn:
            yield conn

    def health_check(self) -> Dict[str, Any]:
        """
        Verify database connectivity and check pgvector extension status.

        Returns:
            Dictionary with health status, PostgreSQL version, and vector extension info.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    # Check PostgreSQL version
                    cur.execute("SELECT version();")
                    version_row = cur.fetchone()
                    pg_version = version_row["version"] if version_row else "Unknown"

                    # Check vector extension
                    cur.execute(
                        "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
                    )
                    vector_row = cur.fetchone()
                    has_vector = vector_row is not None
                    vector_version = vector_row["extversion"] if vector_row else None

                    # Check table status
                    cur.execute(
                        "SELECT EXISTS ("
                        "   SELECT FROM information_schema.tables "
                        "   WHERE table_name = 'github_issues'"
                        ");"
                    )
                    table_exists = cur.fetchone()["exists"]

                    # Count records if table exists
                    record_count = 0
                    if table_exists:
                        cur.execute("SELECT COUNT(*) AS total FROM github_issues;")
                        record_count = cur.fetchone()["total"]

                    return {
                        "status": "healthy",
                        "connected": True,
                        "database": self.dbname,
                        "pg_version": pg_version,
                        "vector_extension_installed": has_vector,
                        "vector_extension_version": vector_version,
                        "github_issues_table_exists": table_exists,
                        "total_records": record_count,
                    }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "database": self.dbname,
            }

    def init_db(self, schema_path: str = "core/db_schema.sql") -> bool:
        """
        Initialize database schema by executing the DDL script.

        Args:
            schema_path: Path to SQL schema definition file.

        Returns:
            bool: True if initialization succeeded.
        """
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at: {schema_file.resolve()}")

        with open(schema_file, "r", encoding="utf-8") as f:
            sql_script = f.read()

        logger.info(f"Applying schema migration from {schema_file.resolve()}...")
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_script)
            conn.commit()

        logger.info("Database schema initialized successfully.")
        return True

    def batch_upsert_issues(self, issues_batch: List[Dict[str, Any]]) -> int:
        """
        Insert or update a batch of GitHub issues with vector embeddings.

        Uses ON CONFLICT (issue_number) DO UPDATE for idempotent updates.

        Args:
            issues_batch: List of issue dictionaries with embedding vectors.

        Returns:
            int: Number of rows successfully upserted.
        """
        if not issues_batch:
            return 0

        upsert_sql = """
        INSERT INTO github_issues (
            issue_number,
            title,
            body,
            state,
            labels,
            created_at,
            closed_at,
            embedding
        ) VALUES (
            %(issue_number)s,
            %(title)s,
            %(body)s,
            %(state)s,
            %(labels)s,
            %(created_at)s,
            %(closed_at)s,
            %(embedding)s
        )
        ON CONFLICT (issue_number) DO UPDATE SET
            title = EXCLUDED.title,
            body = EXCLUDED.body,
            state = EXCLUDED.state,
            labels = EXCLUDED.labels,
            created_at = EXCLUDED.created_at,
            closed_at = EXCLUDED.closed_at,
            embedding = EXCLUDED.embedding;
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for issue in issues_batch:
                    # Prepare params
                    params = {
                        "issue_number": issue["issue_number"],
                        "title": issue["title"],
                        # Use composite_text or cleaned_body for full text index
                        "body": issue.get("composite_text") or issue.get("cleaned_body") or issue.get("body", ""),
                        "state": issue.get("state", "closed"),
                        "labels": issue.get("labels", []),
                        "created_at": issue.get("created_at"),
                        "closed_at": issue.get("closed_at"),
                        "embedding": issue["embedding"],
                    }
                    cur.execute(upsert_sql, params)
            conn.commit()

        return len(issues_batch)

    def get_indexed_issue_count(self) -> int:
        """
        Get count of issues in github_issues that have valid non-null vector embeddings.
        """
        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS total FROM github_issues WHERE embedding IS NOT NULL;"
                )
                row = cur.fetchone()
                return row["total"] if row else 0

    def get_issue_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for stored GitHub issues.
        """
        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                SELECT 
                    COUNT(*) AS total_issues,
                    COUNT(embedding) AS embedded_issues,
                    MIN(created_at) AS earliest_issue,
                    MAX(created_at) AS latest_issue
                FROM github_issues;
                """)
                stats = cur.fetchone() or {}
                return dict(stats)

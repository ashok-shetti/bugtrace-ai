"""
Main FastAPI Application Entrypoint for BugTrace AI.

Configures modern @asynccontextmanager lifespan handler, CORS middleware,
request latency observability middleware, health diagnostics, and REST routes.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from core.db_client import DatabaseClient
from core.embedding import EmbeddingGenerator
from core.engine import get_engine
from .routes import router
from .schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BugTraceApp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager: handles startup preloading and graceful shutdown.
    """
    logger.info("Starting BugTrace AI application...")

    # Initialize global engine and preload dependencies
    engine = get_engine()

    # 1. Check database connectivity
    health = engine.db_client.health_check()
    if health.get("connected"):
        logger.info(
            f"Database connected successfully ({health.get('total_records', 0)} indexed issues, "
            f"pgvector version: {health.get('vector_extension_version')})."
        )
    else:
        logger.warning(
            f"PostgreSQL connection offline: {health.get('error')}. "
            "Ensure database container is started via `docker compose up -d`."
        )

    # 2. Preload embedding model into memory for low-latency first query
    logger.info("Preloading sentence-transformers embedding model into memory...")
    _ = engine.embedder.model
    logger.info(
        f"Embedding model loaded on device '{engine.embedder.device}' "
        f"(dimension: {engine.embedder.dimension})."
    )

    yield

    # Shutdown logic
    logger.info("Shutting down BugTrace AI application...")
    engine.db_client.close()
    logger.info("Database connection pools closed. Clean shutdown complete.")


# Initialize FastAPI Application
app = FastAPI(
    title="BugTrace AI",
    description=(
        "Autonomous AI-powered bug diagnosis and resolution system. "
        "Performs hybrid search (pgvector + tsvector via RRF) on historical GitHub issues "
        "and generates type-safe, grounded root-cause diagnoses using LLM Structured Outputs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """
    Middleware measuring request execution latency for MLE observability.
    Adds 'X-Process-Time' header to all responses.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Latency: {process_time * 1000:.2f}ms"
    )
    return response


# Include API Routers
app.include_router(router)


@app.get("/", tags=["System"])
async def root():
    """
    Root welcome endpoint with system metadata.
    """
    return {
        "name": "BugTrace AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "diagnose": "/api/v1/diagnose",
            "search": "/api/v1/search",
            "ingest": "/api/v1/ingest",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check diagnostic endpoint verifying database reachability and model memory status.
    """
    engine = get_engine()
    db_health = engine.db_client.health_check()
    model_loaded = engine.embedder._model is not None

    return HealthResponse(
        status="healthy" if db_health.get("connected") else "degraded",
        database_connected=bool(db_health.get("connected")),
        model_loaded=model_loaded,
        total_indexed_issues=db_health.get("total_records", 0),
        pgvector_version=db_health.get("vector_extension_version"),
        version="1.0.0",
    )

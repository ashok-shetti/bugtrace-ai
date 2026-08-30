# BugTrace AI: Hybrid RAG System for Automated Incident & Bug Resolution

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Structured%20Outputs-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5.0-blue?style=flat)](https://github.com/pgvector/pgvector)
[![Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

BugTrace AI is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed to dramatically reduce Mean Time to Resolution (MTTR) for software engineering and SRE teams. By coupling parallel dense semantic search (`pgvector`) and full-text keyword search (`tsvector`) via Reciprocal Rank Fusion (RRF), BugTrace AI grounds LLM diagnoses in historical closed GitHub issue resolutions and maintainer fix commits, delivering deterministic, type-safe root-cause analyses without hallucinations.

---

## System Architecture

```text
                                  +---------------------------------------+
                                  |         Target GitHub Repositories    |
                                  |    (e.g., tiangolo/fastapi, pallets)  |
                                  +-------------------+-------------------+
                                                      |
                                                      | 1. REST API ETL (Rate-Limit Backoff & Bot Filter)
                                                      v
                                  +---------------------------------------+
                                  |           etl/github_fetcher.py       |
                                  |     (Closed bugs + Maintainer fixes)  |
                                  +-------------------+-------------------+
                                                      |
                                                      | 2. Technical Text Cleaning & Code Preservation
                                                      v
                                  +---------------------------------------+
                                  |            etl/text_cleaner.py        |
                                  |  - Strip boilerplate / Markdown HTML  |
                                  |  - Strictly preserve ``` code & logs  |
                                  |  - Enforce ~1,500 token chunk budget  |
                                  +-------------------+-------------------+
                                                      |
                                                      | 3. Dense Embedding (all-MiniLM-L6-v2, 384-dim)
                                                      v
                                  +---------------------------------------+
                                  |          etl/ingest_pipeline.py       |
                                  +-------------------+-------------------+
                                                      |
                                                      | 4. Idempotent Batch Upsert (ON CONFLICT DO UPDATE)
                                                      v
  +-------------------------------------------------------------------------------------------------------+
  |                                     PostgreSQL 16 + pgvector Single Store                              |
  |                                                                                                       |
  |   +--------------------------------------------+    +---------------------------------------------+   |
  |   |           HNSW Vector Index                |    |               GIN Full-Text Index           |   |
  |   |     (embedding vector_cosine_ops)          |    |     (tsvector on title + description + fix) |   |
  |   +---------------------+----------------------+    +----------------------+----------------------+   |
  |                         |                                                  |                          |
  +-------------------------|--------------------------------------------------|--------------------------+
                            |                                                  |
                            | 5a. Vector Cosine Distance (<=>)                 | 5b. BM25 / ts_rank_cd (@@)
                            +------------------------+-------------------------+
                                                     |
                                                     v
                                  +---------------------------------------+
                                  |          core/retriever.py            |
                                  |  Unified Parallel CTE Query with RRF  |
                                  |   RRF_Score = SUM( 1 / (60 + Rank) )  |
                                  +-------------------+-------------------+
                                                      |
                                                      | 6. Top-K Grounded Context Injection
                                                      v
                                  +---------------------------------------+
                                  |          core/llm_prompt.py           |
                                  |      Google Gemini Structured Outputs |
                                  |  (Pydantic BugDiagnosisResponse Type) |
                                  +-------------------+-------------------+
                                                      |
                                                      | 7. Validated JSON Diagnosis
                                                      v
                                  +---------------------------------------+
                                  |            api/main.py                |
                                  |       FastAPI REST Endpoints          |
                                  |     (/diagnose, /search, /health)     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |               ui/app.py               |
                                  |     Streamlit Developer Console       |
                                  +---------------------------------------+
```

---

## Core Engineering Highlights & Technical Decisions

### 1. Single-Store Vector & Relational Architecture (PostgreSQL + pgvector)
* **Elimination of Cross-Service Network Hops**: Standalone vector databases (e.g., Pinecone, Milvus, Qdrant) require dual-writes, cross-service network coordination, and eventual consistency syncs. By consolidating dense vectors (384-dimensional embeddings) and relational issue metadata into PostgreSQL, vector similarity and SQL metadata filtering execute within a single engine.
* **ACID Guarantees & Transactional Integrity**: Issues and their respective embeddings are upserted atomically using `ON CONFLICT (issue_number) DO UPDATE`, preventing orphaned embeddings or out-of-sync vector indexes.
* **Cost & Operational Simplicity**: Runs efficiently within standard containerized PostgreSQL infrastructure without dedicated external SaaS costs.

### 2. Hybrid Search with Reciprocal Rank Fusion (RRF)
* **Why Pure Dense Vector Search Fails on Technical Context**: Embedding models often compress specific technical tokens (e.g., error class names `RequestValidationError`, status codes `422`, specific function decorators `@app.errorhandler`, or Python module paths) into generalized semantic clusters, causing low exact-match precision.
* **Why Pure BM25 Keyword Search Fails**: Keyword search fails when users describe an issue conceptually (e.g., *"background task fails silently"* vs. the documented code *"Task was destroyed but it is pending"*).
* **Unified PostgreSQL RRF CTE Query**: We execute both retrieval paths in parallel CTEs inside PostgreSQL and join them via `FULL OUTER JOIN` using the Reciprocal Rank Fusion equation:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{1}{k + \text{rank}_m(d)}$$

where $k = 60$ is the standard smoothing parameter. This guarantees top ranking for documents that excel in either exact keyword matching, semantic intent, or both.

### 3. Type-Safe Structured Inference via Google Gemini & Pydantic
* **Eliminating LLM Markdown Hallucinations**: Prompting LLMs with standard free-form text or raw JSON instructions frequently results in malformed syntax, unquoted keys, or conversational filler.
* **Gemini Structured Outputs**: Utilizes `client.models.generate_content(config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=BugDiagnosisResponse))` to strictly constrain token generation to our Pydantic schema, guaranteeing valid JSON deserialization.
* **Resilient Multi-Model Fallback**: Automatically retries across available Gemini models (`gemini-3.6-flash`, `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-flash-latest`) if transient load spikes occur.
* **Enforced Attribution**: The schema explicitly mandates a list of `ReferencedBug` objects containing the specific historical issue numbers and justification reasons that grounded the fix.

---

## Database Schema & Indexing Strategy

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing closed GitHub issues with hybrid search support
CREATE TABLE IF NOT EXISTS github_issues (
    id BIGSERIAL PRIMARY KEY,
    issue_number INTEGER UNIQUE NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    state VARCHAR(50) DEFAULT 'closed',
    labels TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE,
    closed_at TIMESTAMP WITH TIME ZONE,
    embedding vector(384),
    text_searchable_index_col tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))
    ) STORED
);

-- 1. GIN index for fast full-text keyword retrieval
CREATE INDEX IF NOT EXISTS idx_github_issues_fts 
ON github_issues 
USING GIN (text_searchable_index_col);

-- 2. HNSW index for sub-millisecond approximate nearest neighbor vector search
CREATE INDEX IF NOT EXISTS idx_github_issues_embedding_hnsw 
ON github_issues 
USING hnsw (embedding vector_cosine_ops);
```

| Column | Type | Description | Index Type |
|---|---|---|---|
| `id` | `BIGSERIAL` | Internal surrogate primary key | `B-Tree (PRIMARY KEY)` |
| `issue_number` | `INTEGER` | Unique GitHub issue identifier | `B-Tree (UNIQUE)` |
| `title` | `TEXT` | Sanitized issue title | Included in GIN tsvector |
| `body` | `TEXT` | Composite issue context & maintainer resolution | Included in GIN tsvector |
| `state` | `VARCHAR(50)` | Issue status (`closed`) | Relational Filter |
| `labels` | `TEXT[]` | GitHub label tags for array containment (`@>`) | GIN Array (Optional) |
| `created_at` / `closed_at` | `TIMESTAMP WITH TIME ZONE` | Issue lifecycle timestamps | B-Tree |
| `embedding` | `vector(384)` | L2-normalized 384-dim dense embedding | `HNSW (vector_cosine_ops)` |
| `text_searchable_index_col`| `tsvector` | Automatically generated stored English lexemes | `GIN (Full-Text Search)` |

---

## Project Structure

```text
BugTrace AI/
├── .env.example               # Template for DB credentials, GitHub token & Gemini API key
├── docker-compose.yml         # Containerized PostgreSQL 16 + pgvector setup
├── requirements.txt           # Production Python dependencies
├── core/                      # Core Retrieval & Inference Engine
│   ├── __init__.py            # Package exports
│   ├── db_client.py           # Thread-safe connection pool & pgvector adapter manager
│   ├── db_schema.sql          # DDL migrations, HNSW index & GIN tsvector index
│   ├── embedding.py           # Sentence-Transformers embedding manager (all-MiniLM-L6-v2)
│   ├── engine.py              # BugTraceEngine orchestrator facade
│   ├── llm_prompt.py          # Grounded prompt formatter & Google Gemini Structured Outputs
│   ├── retriever.py           # Parallel CTE Hybrid Search (pgvector + tsvector via RRF)
│   ├── schemas.py             # Core Pydantic response models (BugDiagnosisResponse)
│   ├── test_diagnosis.py      # Standalone CLI diagnosis runner & verification script
│   └── test_retrieval.py      # Standalone hybrid retrieval benchmarking tool
├── etl/                       # Data Extraction & Preprocessing Pipeline
│   ├── __init__.py            # Module exports
│   ├── github_fetcher.py      # GitHub REST fetcher with pagination & bot filtering
│   ├── text_cleaner.py        # Code-preserving sanitizer & token window regulator
│   ├── ingest_pipeline.py     # Batch vector embedding & PostgreSQL ingestion pipeline
│   └── run_etl.py             # CLI orchestrator for automated extraction & cleaning
├── api/                       # Asynchronous FastAPI REST Layer
│   ├── __init__.py            # App exports
│   ├── main.py                # Lifespan context manager, CORS & latency middleware
│   ├── routes.py              # REST route definitions (/diagnose, /search, /ingest)
│   └── schemas.py             # API request/response validation schemas
├── ui/                        # Streamlit Frontend Layer
│   └── app.py                 # Modern, minimal developer incident & error console
├── tests/                     # Automated Test Suite (35 Unit & Integration Tests)
│   ├── test_api.py            # FastAPI TestClient endpoint verification
│   ├── test_embedding_and_db.py # Embedding dimension & DB adapter tests
│   ├── test_etl.py            # Code-block preservation, bot & checklist filter tests
│   ├── test_llm_diagnosis.py  # Prompt formatting, Pydantic validation & Gemini mock tests
│   └── test_retriever.py      # Hybrid RRF CTE query & parameter serialization tests
└── data/                      # Local data artifacts
    ├── raw/issues.json        # Raw extracted issues from GitHub
    └── processed/cleaned_issues.json # Cleaned, embedding-ready composite representations
```

---

## Quickstart & Setup Instructions

### 1. Prerequisites
* **Python 3.11+**
* **Docker & Docker Compose**
* **Git**

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/bugtrace-ai.git
cd "BugTrace AI"

# Create and activate Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
```ini
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=bugtrace_db
GITHUB_TOKEN=ghp_your_token_here     # Optional: enables 5,000 req/hr GitHub rate limit
GEMINI_API_KEY=AIzaSy_your_key_here  # Required for live LLM diagnosis (Google AI Studio)
GEMINI_MODEL=gemini-3.6-flash        # Optional: default model (gemini-3.6-flash)
```

### 4. Start PostgreSQL with pgvector
```bash
docker compose up -d
```

### 5. Run ETL Extraction & Database Ingestion
```bash
# Step 1: Extract closed bug issues from target repository (e.g. FastAPI)
python etl/run_etl.py --repo tiangolo/fastapi --max-issues 200

# Step 2: Initialize schema and generate embeddings into PostgreSQL
python etl/ingest_pipeline.py --init-db
```

### 6. Start the FastAPI REST API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is now live at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

### 7. Start the Streamlit Frontend Console
In a separate terminal:
```bash
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501`.

### 8. Run Test Suite
```bash
python -m unittest discover tests -v
```

---

## API Documentation & Sample Payloads

### 1. End-to-End Bug Diagnosis (`POST /api/v1/diagnose`)
Executes hybrid search to retrieve the top historical matches, constructs the grounded prompt, and calls Gemini with Structured Outputs.

#### Request:
```bash
curl -X POST "http://localhost:8000/api/v1/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI raises RequestValidationError on missing field username in request body with custom Pydantic validator",
    "labels": ["bug", "validation"],
    "top_k": 3
  }'
```

#### Response (`BugDiagnosisResponse`):
```json
{
  "summary": "Pydantic validator raises RequestValidationError when required request body fields are omitted without explicit default values.",
  "root_cause_analysis": "FastAPI expects body parameters to conform to the Pydantic model definition. If a field is declared as required (e.g., `username: str`) and lacks a default value or `Optional` typing, the internal request validation pipeline triggers an unhandled 422 Unprocessable Entity error before hitting the endpoint logic.",
  "recommended_fix": "Mark the field as optional with a default value if it is not strictly mandatory:\n\n```python\nfrom typing import Optional\nfrom pydantic import BaseModel, Field\n\nclass UserPayload(BaseModel):\n    username: Optional[str] = Field(default=None, description=\"Optional username\")\n```\nAlternatively, ensure client requests supply a valid JSON payload containing the `username` key.",
  "referenced_issues": [
    {
      "issue_number": 4295,
      "relevance_reason": "Historical issue addressing Pydantic request body validation failures on required model attributes."
    }
  ],
  "confidence_score": 0.94
}
```

---

### 2. Isolated Hybrid Search (`POST /api/v1/search`)
Runs pure hybrid retrieval (vector similarity + full-text search fused via RRF) without incurring LLM inference costs.

#### Request:
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "greenlet.error: cannot switch to a different thread",
    "top_k": 2
  }'
```

#### Response:
```json
{
  "query": "greenlet.error: cannot switch to a different thread",
  "total_results": 2,
  "results": [
    {
      "issue_number": 4355,
      "title": "Async/await read file error: greenlet.error: cannot switch to a different thread",
      "body": "When running async handlers under gevent WSGIServer, manual event loop creation triggers greenlet thread switching errors...",
      "labels": ["bug"],
      "rank_vec": 1,
      "rank_text": 1,
      "rrf_score": 0.03278
    },
    {
      "issue_number": 3776,
      "title": "request.endpoint always None in SessionInterface.open_session",
      "body": "SessionInterface subclass looks at request.endpoint...",
      "labels": ["bug"],
      "rank_vec": 2,
      "rank_text": null,
      "rrf_score": 0.01612
    }
  ]
}
```

---

### 3. System Health Diagnostics (`GET /health`)
```bash
curl "http://localhost:8000/health"
```
```json
{
  "status": "healthy",
  "database_connected": true,
  "model_loaded": true,
  "total_indexed_issues": 200,
  "pgvector_version": "0.5.0",
  "version": "1.0.0"
}
```

---

## Technical Deep-Dive: Architecture Trade-Offs & Future Roadmap

### 1. Vector Index Selection: HNSW vs. IVFFlat
* **IVFFlat (Inverted File Flat)**:
  * *Mechanism*: Partitions vector space into Voronoi cells via k-means clustering.
  * *Downside*: Requires the table to be populated before training the index. In dynamic incident resolution platforms where new resolved bugs are continuously streamed, IVFFlat suffers severe recall degradation unless frequently retrained from scratch.
* **HNSW (Hierarchical Navigable Small World)**:
  * *Why We Selected HNSW*: HNSW builds a multi-layer proximity graph upon insertion. It delivers significantly higher recall (>98%) at high queries-per-second (QPS) and supports incremental, real-time inserts without requiring full-index rebuilds.
  * *Trade-off*: Higher initial memory footprint during index build, which is well within standard PostgreSQL server capacity for sub-million vector datasets.

### 2. Code-Aware Chunking vs. Naive Character Splitting
* **The Failure of Naive Character Chunking**: Splitting technical text arbitrarily every 500 characters breaks stack traces in half, truncates variable definitions, and separates error messages from root cause tracebacks.
* **BugTrace AI Cleaning Strategy**:
  1. Identifies and isolates fenced code blocks (```` ``` ````) and stack traces.
  2. Strips automated checklist boilerplate (`- [x] I searched existing issues`) and HTML noise.
  3. When truncating oversized logs (>3,000 characters), preserves the top 30 lines (error declaration) and bottom 25 lines (traceback root cause frame), discarding redundant intermediate polling loops.
  4. Formats into unified semantic units: `Title -> Labels -> Issue Description -> Resolution Fix`.

### 3. Future Scaling Roadmap
* **Cross-Encoder Re-Ranking Stage**: Introduce a secondary re-ranker (such as `BAAI/bge-reranker-large`) on the top-20 RRF candidates to score cross-attention between user stack traces and candidate code snippets before context prompt assembly.
* **Semantic Cache Layer (Redis)**: Cache recurrent error signatures and high-frequency stack traces with sub-millisecond TTL response times to bypass LLM inference for identical production incidents.
* **Multi-Repo Federation**: Extend relational partitioning on `repo_owner` and `repo_name` columns to scale index partitioning across thousands of microservices within an enterprise monorepo or organization.

---

## License
MIT License. Created for the BugTrace AI Project.

# BugTrace AI — AI-Powered Bug Resolution Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Structured%20Outputs-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5.0-blue?style=flat)](https://github.com/pgvector/pgvector)
[![Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

BugTrace AI is an AI-powered bug resolution assistant that helps developers investigate issues by retrieving similar historical GitHub issues and their resolutions. It combines semantic search and keyword search through Hybrid RAG, then uses Google Gemini to generate structured, grounded diagnoses based on retrieved context.

Built with FastAPI, PostgreSQL + pgvector, Sentence Transformers, Google Gemini, and Streamlit.

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
                                  +-------------------------------------------------+
                                  |            api/main.py                          |
                                  |       FastAPI REST Endpoints                    |
                                  |(/api/v1/diagnose,/api/v1/search,/api/v1/ingest) |
                                  +-------------------+-----------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |               ui/app.py               |
                                  |     Streamlit Developer Console       |
                                  +---------------------------------------+
```

## Key Engineering Decisions

### 1. Why PostgreSQL + pgvector
* **Keeping Vectors and Data in One Database**: Standalone vector databases (e.g., Pinecone, Milvus, Qdrant) require dual-writes, cross-service network coordination, and eventual consistency syncs. By consolidating dense vectors (384-dimensional embeddings) and relational issue metadata into PostgreSQL, vector similarity and SQL metadata filtering execute within a single engine.
* **Keeping Data and Embeddings Consistent**: Issues and their respective embeddings are upserted atomically using `ON CONFLICT (issue_number) DO UPDATE`, preventing orphaned embeddings or out-of-sync vector indexes.
* **Simpler to Run and Maintain**: Runs efficiently within standard containerized PostgreSQL infrastructure without dedicated external SaaS costs.

### 2. Hybrid Search with Reciprocal Rank Fusion (RRF)
* **Limitations of Semantic Search for Technical Errors**: Embedding models often compress specific technical tokens (e.g., error class names `RequestValidationError`, status codes `422`, specific function decorators `@app.errorhandler`, or Python module paths) into generalized semantic clusters, causing low exact-match precision.
* **Limitations of Keyword Search**: Keyword search fails when users describe an issue conceptually (e.g., *"background task fails silently"* vs. the documented code *"Task was destroyed but it is pending"*).
* **Unified PostgreSQL RRF CTE Query**: We execute both retrieval paths in parallel CTEs inside PostgreSQL and join them via `FULL OUTER JOIN` using the Reciprocal Rank Fusion equation:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{1}{k + \text{rank}_m(d)}$$

where $k = 60$ is the standard smoothing parameter. This guarantees top ranking for documents that excel in either exact keyword matching, semantic intent, or both.

### 3. Type-Safe Structured Inference via Google Gemini & Pydantic
* **Schema Enforcement**: Prompting LLMs with free-form text or standard JSON prompts can produce unquoted keys or formatting inconsistencies. We pass our Pydantic schema directly into Gemini's `response_schema` parameter to ensure deterministic, type-safe JSON output.
* **Gemini Structured Outputs**: Calls `client.models.generate_content(config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=BugDiagnosisResponse))` to constrain token generation directly to the `BugDiagnosisResponse` model.
* **Model Fallback Sequence**: Automatically falls back across available Gemini models (`gemini-3.6-flash`, `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-flash-latest`) if transient API errors or rate limits occur.
* **Attribution Requirement**: The schema mandates a list of `ReferencedBug` objects with historical issue numbers and relevance explanations, ensuring the diagnosis remains grounded in retrieved evidence.

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

-- 2. HNSW index for efficient approximate nearest-neighbor vector search
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
    ├── raw/issues.json        # Bundled raw extracted issues (50 sample bugs)
    └── processed/cleaned_issues.json # Cleaned, embedding-ready processed issues
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
git clone https://github.com/ashok-shetti/bugtrace-ai.git
cd bugtrace-ai

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
DB_PASSWORD=postgres                 # Default password configured in docker-compose.yml
DB_NAME=bugtrace_db
GITHUB_TOKEN=your_github_token_here  # Optional: increases GitHub API rate limit from 60 to 5,000 req/hr
GEMINI_API_KEY=your_gemini_api_key   # Required for LLM diagnosis (Google AI Studio)
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

### 3. Background Repository Ingestion (`POST /api/v1/ingest`)
Dispatches an asynchronous background task to extract closed bug issues from GitHub, sanitize text, generate embeddings, and upsert records into PostgreSQL.

#### Request:
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_owner": "tiangolo",
    "repo_name": "fastapi",
    "max_issues": 100,
    "label": "bug"
  }'
```

#### Response:
```json
{
  "status": "accepted",
  "issues_processed": 0,
  "message": "Ingestion job scheduled in background for repository 'tiangolo/fastapi' (target: 100 issues)."
}
```

---

### 4. System Health Diagnostics (`GET /health`)
Returns database connectivity status, vector extension info, and total indexed issue count (50 in the bundled sample dataset, or more after running ingestion).

#### Request:
```bash
curl "http://localhost:8000/health"
```

#### Response:
```json
{
  "status": "healthy",
  "database_connected": true,
  "model_loaded": true,
  "total_indexed_issues": 50,
  "pgvector_version": "0.5.0",
  "version": "1.0.0"
}
```

---

## Technical Trade-Offs & Future Roadmap

### 1. Vector Index Selection: HNSW vs. IVFFlat
* **IVFFlat (Inverted File Flat)**:
  * *Mechanism*: Partitions vector space into Voronoi cells via k-means clustering.
  * *Downside*: Requires the table to be populated before training the index. When new resolved bugs are continuously added, IVFFlat can suffer recall degradation unless periodically retrained from scratch.
* **HNSW (Hierarchical Navigable Small World)**:
  * *Why We Selected HNSW*: HNSW constructs a multi-layer proximity graph upon insertion. In vector retrieval workloads, graph-based indexing maintains high recall under high query throughput and supports real-time incremental inserts without requiring full index rebuilds.
  * *Trade-off*: Higher memory usage during index construction compared to inverted lists, which is easily accommodated within standard PostgreSQL resources for datasets of this scale.

### 2. Code-Preserving Cleaning vs. Fixed-Character Chunking
* **Why Fixed-Length Splitting Fails on Code**: Splitting technical text arbitrarily every 500 characters breaks stack traces in half, cuts off variable definitions, and separates error messages from root-cause tracebacks.
* **Cleaning Strategy Implemented in ETL**:
  1. Identifies and isolates fenced code blocks (```` ``` ````) and stack traces before text filtering.
  2. Strips automated checklist boilerplate (`- [x] I searched existing issues`) and HTML comments/tags.
  3. When truncating oversized logs (>3,000 characters or >60 lines), preserves the top 30 lines (error declaration) and bottom 25 lines (traceback root-cause frame), discarding intermediate polling noise.
  4. Formats into a clean composite context: `Title -> Labels -> Issue Description -> Resolution Fix`.

### 3. Future Roadmap
* **Cross-Encoder Re-Ranking**: Introduce a secondary cross-encoder (such as `BAAI/bge-reranker-large`) on top-20 candidates to re-score candidate relevance before context prompt injection.
* **Semantic Cache Layer (Redis)**: Cache recurrent error signatures and common stack traces with short TTLs to bypass LLM inference for duplicate errors.
* **Multi-Repo Federation**: Add relational filtering or table partitioning on `repo_owner` and `repo_name` to scale across multiple repositories within an organization.

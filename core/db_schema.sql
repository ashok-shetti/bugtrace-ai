-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing closed GitHub issues with both relational and vector data
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

-- GIN index for fast full-text keyword search
CREATE INDEX IF NOT EXISTS idx_github_issues_fts 
ON github_issues 
USING GIN (text_searchable_index_col);

-- HNSW index for fast vector retrieval using cosine distance
CREATE INDEX IF NOT EXISTS idx_github_issues_embedding_hnsw 
ON github_issues 
USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create multimodal_chunks table
CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type   TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,
    content      TEXT NOT NULL,
    image_bytes  BYTEA,
    mime_type    TEXT,
    page_number  INTEGER,
    section      TEXT,
    source_file  TEXT,
    position     JSONB,
    embedding    VECTOR(1536),
    metadata     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Correct HNSW index creation
CREATE INDEX IF NOT EXISTS multimodal_chunks_embedding_hnsw_idx
ON multimodal_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

SELECT COUNT(*) FROM multimodal_chunks;

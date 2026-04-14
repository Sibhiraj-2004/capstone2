import os
from typing import List, Dict, Any

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from psycopg.rows import dict_row

from src.core.db import _embeddings_model, _get_pool

load_dotenv()


def _get_raw_conn() -> str:
    raw = os.getenv("RAW_PG_CONNECTION") or os.getenv(
        "PG_CONNECTION_STRING", ""
    ).replace("postgresql+psycopg://", "postgresql://")
    if not raw:
        raise RuntimeError("RAW_PG_CONNECTION or PG_CONNECTION_STRING not set")
    return raw


def vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Embed the query, then do cosine similarity search directly against
    multimodal_chunks.embedding using pgvector's <=> operator.
    Retrieves all chunk types: text, tables, and images.
    Returns list of dicts: {content, metadata, source_file, page_number, score}
    """
    # Embed the query using the same model used during ingestion
    query_embedding = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = """
        SELECT
            id,
            content,
            chunk_type,
            source_file,
            page_number,
            section,
            element_type,
            metadata,
            1 - (embedding <=> %(embedding)s::vector) AS cosine_similarity
        FROM multimodal_chunks
        WHERE content IS NOT NULL
          AND content <> ''
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(k)s;
    """

    with psycopg.connect(_get_raw_conn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"embedding": embedding_str, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "score": float(row["cosine_similarity"]),
            "metadata": {
                "source_file": row["source_file"],
                "page_number": row["page_number"],
                "section": row["section"],
                "element_type": row["element_type"],
                "chunk_type": row["chunk_type"],
                **(row["metadata"] or {}),
            },
        }
        for row in rows
    ]


def _vector_search_with_scores(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Public helper used by hybrid search — returns same dict format."""
    return vector_search(query, k=k)


@tool
def vector_search_tool(query: str) -> List[Document]:
    """
    Semantic vector search using embeddings against multimodal_chunks table.
    Best for: conceptual questions, general understanding, semantic similarity.

    Args:
        query: The search query string

    Returns:
        List of Document objects with relevant content
    """
    results = vector_search(query, k=10)
    docs = [
        Document(
            page_content=r["content"],
            metadata=r["metadata"],
        )
        for r in results
    ]
    print(f"[vector_search_tool] Retrieved {len(docs)} chunks from multimodal_chunks")
    return docs
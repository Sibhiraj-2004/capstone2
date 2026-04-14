import os

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from psycopg.rows import dict_row
from typing import List

load_dotenv()


def _get_raw_conn() -> str:
    raw = os.getenv("RAW_PG_CONNECTION") or os.getenv(
        "PG_CONNECTION_STRING", ""
    ).replace("postgresql+psycopg://", "postgresql://")
    if not raw:
        raise RuntimeError("RAW_PG_CONNECTION or PG_CONNECTION_STRING not set")
    return raw


@tool
def fts_search_tool(query: str) -> List[Document]:
    """
    Full-text search using PostgreSQL tsvector against multimodal_chunks.
    Best for: exact terms, acronyms, specific phrases, regulatory codes.

    Args:
        query: The search query string

    Returns:
        List of Document objects with relevant content
    """
    results = fts_search(query, k=10)

    if not results:
        print("[fts_search_tool] No keyword matches found.")
        return []

    docs = [
        Document(page_content=r["content"], metadata=r.get("metadata", {}))
        for r in results
    ]
    print(f"[fts_search_tool] Retrieved {len(docs)} chunks via FTS from multimodal_chunks")
    return docs


def fts_search(query: str, k: int = 10, **_kwargs) -> List[dict]:
    """
    Raw FTS against multimodal_chunks.content using plainto_tsquery.
    Retrieves all chunk types: text, tables, and images.
    **_kwargs absorbs legacy collection_name arg so callers don't break.
    Returns list of dicts: {content, metadata, fts_rank}
    """
    sql = """
        SELECT
            content,
            source_file,
            page_number,
            section,
            element_type,
            chunk_type,
            metadata,
            ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM multimodal_chunks
        WHERE content IS NOT NULL
          AND to_tsvector('english', content)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """

    with psycopg.connect(_get_raw_conn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "fts_rank": float(row["fts_rank"]),
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




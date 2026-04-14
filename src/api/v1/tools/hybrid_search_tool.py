import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.vector_search_tool import vector_search

load_dotenv()


@tool
def hybrid_search_tool(query: str) -> List[Document]:
    """
    Hybrid search combining BM25 keyword matching with vector semantic search via RRF.
    Both search directly against multimodal_chunks table.
    Best for: complex queries needing both exact terms and semantic understanding.

    Args:
        query: The search query string

    Returns:
        List of Document objects with relevant content
    """
    results = _hybrid_search(query, k=10)
    docs = [
        Document(page_content=r["content"], metadata=r.get("metadata", {}))
        for r in results
    ]
    print(f"[hybrid_search_tool] Retrieved {len(docs)} chunks via RRF hybrid search")
    return docs


def _hybrid_search(query: str, k: int = 10) -> list[dict]:
    """
    Merge vector and FTS results from multimodal_chunks using Reciprocal Rank Fusion.

    RRF score = sum of 1/(rank + 60) across both lists.
    Chunks appearing in both lists score higher than those in only one.
    """
    vector_results = vector_search(query, k=k)   # list of {content, score, metadata}
    fts_results    = fts_search(query, k=k)       # list of {content, fts_rank, metadata}

    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, item in enumerate(vector_results):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        chunk_map[key]  = {"content": item["content"], "metadata": item["metadata"]}

    for rank, item in enumerate(fts_results):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        chunk_map[key]  = {"content": item["content"], "metadata": item["metadata"]}

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]
import os
from typing import TypedDict, List

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.api.v1.schemas.query_schema import AIResponse, RerankedChunk
from src.api.v1.tools.fts_search_tool import fts_search, fts_search_tool
from src.api.v1.tools.hybrid_search_tool import hybrid_search_tool
from src.api.v1.tools.vector_search_tool import vector_search_tool

load_dotenv(override=True)

# ── Tool registry (name → callable) ──────────────────────────────────────────
_TOOLS = {
    "fts_search_tool": fts_search,
    "vector_search_tool": vector_search_tool,
    "hybrid_search_tool": hybrid_search_tool,
}



# ── 1. State ──────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]   # Wide retrieval  (k=10)
    reranked_docs: List[Document]    # After Cohere reranker (top_n=5)
    rerank_scores: List[float]       # Cohere relevance scores per reranked doc
    response: dict                   # Final structured answer
    validation_passed: bool
    retry_count: int
    validation_reason: str


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )


# ── 2. Node 1: Retriever ──────────────────────────────────────────────────────
# LLM picks the best tool via tool-calling; we execute the chosen tool manually.
# FIX: bind_tools returns an AIMessage with .tool_calls — we must execute them.

def retriever_node(state: RAGState) -> RAGState:
    query = state["query"]
    llm = _make_llm()

    tools_list = [fts_search_tool, vector_search_tool, hybrid_search_tool]
    bound_llm = llm.bind_tools(tools_list)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a search assistant. Analyze the query and select ONE tool:\n"
            "- fts_search_tool   → exact terms, acronyms, regulatory codes\n"
            "- vector_search_tool → conceptual / semantic queries\n"
            "- hybrid_search_tool → needs both keyword and semantic understanding\n"
            "Call exactly ONE tool.",
        ),
        ("human", "{query}"),
    ])

    docs: List[Document] = []

    try:
        ai_message = (prompt | bound_llm).invoke({"query": query})

        # FIX: AIMessage.tool_calls is a list of dicts: [{name, args, id}, ...]
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            for tc in ai_message.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_fn = _TOOLS.get(tool_name)

                if tool_fn is None:
                    print(f"[retriever_node] Unknown tool '{tool_name}'; falling back.")
                    continue

                print(f"[retriever_node] LLM chose tool: {tool_name}")
                result = tool_fn.invoke(tool_args)

                # Tools return List[Document]
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, Document):
                            docs.append(item)
                        elif isinstance(item, dict):
                            docs.append(Document(
                                page_content=item.get("content", ""),
                                metadata=item.get("metadata", {}),
                            ))
                break  # use only the first tool call
        else:
            print("[retriever_node] LLM returned no tool_calls; using fallback.")

    except Exception as e:
        print(f"[retriever_node] Tool-binding failed: {e}")

    # Fallback: run vector search if nothing was retrieved
    if not docs:
        print("[retriever_node] Fallback → vector_search_tool")
        docs = vector_search_tool.invoke({"query": query})

    print(f"[retriever_node] Retrieved {len(docs)} documents")
    return {**state, "retrieved_docs": docs}


# ── 3. Node 2: Rerank ─────────────────────────────────────────────────────────
# Cohere cross-encoder reranker: sees query + doc together → accurate scoring.

def rerank_node(state: RAGState) -> RAGState:
    docs = state["retrieved_docs"]

    if not docs:
        print("[rerank_node] No documents to rerank.")
        return {**state, "reranked_docs": [], "rerank_scores": []}

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    reranked_docs: List[Document] = []
    rerank_scores: List[float] = []

    print(f"[rerank_node] Top {len(rerank_response.results)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        reranked_docs.append(docs[r.index])
        rerank_scores.append(float(r.relevance_score))
        print(f"  Rank {i+1} | score={r.relevance_score:.4f} | original_idx={r.index}")

    return {**state, "reranked_docs": reranked_docs, "rerank_scores": rerank_scores}


# ── 4. Node 3: Validate + optional query expansion ────────────────────────────

def validate_node(state: RAGState) -> RAGState:
    docs = state["reranked_docs"]
    retry_count = state.get("retry_count", 0)
    max_retries = 3

    if not docs:
        if retry_count >= max_retries:
            print(f"[validate_node] No docs after {max_retries} retries — forcing PASS to exit loop.")
            return {
                **state,
                "validation_passed": True,
                "validation_reason": f"No documents found after {max_retries} retries",
                "retry_count": retry_count + 1,
            }
        print(f"[validate_node] No docs (retry {retry_count}/{max_retries}) — will retry.")
        return {
            **state,
            "validation_passed": False,
            "validation_reason": "No documents retrieved",
            "retry_count": retry_count + 1,
        }

    llm = _make_llm()

    docs_summary = "\n---\n".join([
        f"Doc {i+1}: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}"
        for i, doc in enumerate(docs)
    ])

    validation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a document relevance validator. Assess if the documents are sufficient "
            "to answer the user's question. Consider: relevance, coverage, completeness.\n"
            "First line: PASS or FAIL\nSecond line: brief reason.",
        ),
        ("human", "Question: {query}\n\nDocuments:\n{docs_summary}"),
    ])

    try:
        raw = (validation_prompt | llm).invoke({
            "query": state["query"],
            "docs_summary": docs_summary,
        }).content
    except Exception as e:
        print(f"[validate_node] Validation LLM call failed: {e}. Treating as FAIL.")
        raw = "FAIL\nValidation API error, will retry or force exit"

    if isinstance(raw, list):
        raw = raw[0].get("text", "FAIL\nUnexpected format")
    raw = raw.strip()
    lines = raw.split("\n")
    decision = lines[0].upper().strip()
    reason = lines[1].strip() if len(lines) > 1 else decision
    validation_passed = "PASS" in decision

    print(f"[validate_node] Decision={decision}  Reason={reason}")

    updated_query = state["query"]

    if not validation_passed:
        if retry_count < max_retries:
            print(f"[validate_node] Retry {retry_count + 1}/{max_retries} — expanding query…")
            try:
                expand_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are a query expansion expert. Given validation feedback, expand the query "
                        "with related terms, synonyms, and alternative phrasings. "
                        "Return ONLY the expanded query text, nothing else.",
                    ),
                    ("human", "Original query: {query}\n\nFeedback: {feedback}"),
                ])
                expanded = (expand_prompt | llm).invoke({
                    "query": state["query"],
                    "feedback": reason,
                }).content
                if isinstance(expanded, list):
                    updated_query = expanded[0].get("text", state["query"]).strip()
                else:
                    updated_query = expanded.strip()
                print(f"[validate_node] Expanded query: {updated_query}")
            except Exception as e:
                print(f"[validate_node] Query expansion failed: {e}. Skipping expansion and proceeding.")
                updated_query = state["query"]
        else:
            print(f"[validate_node] Max retries reached — forcing PASS.")
            validation_passed = True

    return {
        **state,
        "validation_passed": validation_passed,
        "validation_reason": reason,
        "query": updated_query,
        "retry_count": retry_count + 1,
    }


# ── 5. Node 4: Generate Answer ────────────────────────────────────────────────
# Structured output via AIResponse schema.
# FIX: reranked_chunks populated from reranked_docs + rerank_scores.

def generate_answer_node(state: RAGState) -> RAGState:
    reranked_docs: List[Document] = state.get("reranked_docs", [])
    rerank_scores: List[float] = state.get("rerank_scores", [])

    if not reranked_docs:
        print("[generate_answer_node] No reranked docs — returning empty answer.")
        return {
            **state,
            "response": {
                "query": state["query"],
                "answer": "No relevant documents found for the given query.",
                "policy_citations": "",
                "page_no": "",
                "document_name": "",
                "reranked_chunks": [],
            },
        }

    llm = _make_llm()

    # Build structured output chain (excludes reranked_chunks — we add them manually)
    class _LLMAnswer(AIResponse.__class__):
        pass

    # Use a simpler pydantic model for the LLM structured output to avoid recursion
    from pydantic import BaseModel, Field as PField

    class _CoreAnswer(BaseModel):
        query: str = PField(description="The query submitted by the user")
        answer: str = PField(description="Generated answer from the LLM")
        policy_citations: str = PField(description="Cited policy or regulation reference")
        page_no: str = PField(description="Page number(s) referenced")
        document_name: str = PField(description="Name of the source document used")

    structured_llm = llm.with_structured_output(_CoreAnswer)

    context_parts = []
    for doc in reranked_docs:
        chunk_type = doc.metadata.get("chunk_type", "text")
        source_file = doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
        page_number = doc.metadata.get("page_number") or doc.metadata.get("page", "?")
        
        # Format context with chunk type indication for proper processing
        if chunk_type == "table":
            context_parts.append(
                f"[TABLE | Source: {source_file} | Page: {page_number}]\n{doc.page_content}"
            )
        elif chunk_type == "image":
            context_parts.append(
                f"[IMAGE | Source: {source_file} | Page: {page_number}]\n{doc.page_content}"
            )
        else:  # text or other
            context_parts.append(
                f"[TEXT | Source: {source_file} | Page: {page_number}]\n{doc.page_content}"
            )
    
    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful compliance assistant. Answer the user's question using ONLY the provided context. "
            "The context may include text passages, tables, and image descriptions labeled as [TEXT], [TABLE], or [IMAGE]. "
            "When citing tables or images, specifically mention their type (e.g., 'According to Table X...' or 'As shown in Figure X...'). "
            "Be precise and always cite the source document, page number, and content type. "
            "If the question cannot be answered with the provided context or relates to topics outside of the provided content, "
            "respond with 'I cannot answer this question based on the provided content.' "
            "Do not provide any information beyond the provided content, and do not hallucinate information.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])

    core: _CoreAnswer = (prompt | structured_llm).invoke({
        "context": context,
        "query": state["query"],
    })

    # Build reranked_chunks list for the response
    reranked_chunks = []
    for i, (doc, score) in enumerate(zip(reranked_docs, rerank_scores), start=1):
        reranked_chunks.append(RerankedChunk(
            rank=i,
            content=doc.page_content,
            source_file=doc.metadata.get("source_file") or doc.metadata.get("source"),
            page_number=doc.metadata.get("page_number") or doc.metadata.get("page"),
            chunk_type=doc.metadata.get("chunk_type", "text"),
            relevance_score=round(score, 4),
        ).model_dump())

    response = {
        **core.model_dump(),
        "reranked_chunks": reranked_chunks,
    }

    print(f"[generate_answer_node] Answer generated. Chunks returned: {len(reranked_chunks)}")
    return {**state, "response": response}


# ── 6. Router ─────────────────────────────────────────────────────────────────
def route_validation(state: RAGState) -> str:
    return "generate_answer" if state.get("validation_passed", False) else "retriever"


# ── 7. Build Graph ────────────────────────────────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "rerank")
    graph.add_edge("rerank", "validate")
    graph.add_conditional_edges(
        "validate",
        route_validation,
        {
            "generate_answer": "generate_answer",
            "retriever": "retriever",
        },
    )
    graph.add_edge("generate_answer", END)

    return graph.compile()


# Compiled once at import — reused across all requests
rag_graph = build_rag_graph()


# ── 8. Public entrypoint ──────────────────────────────────────────────────────
def run_vector_search_agent(query: str) -> dict:
    initial_state: RAGState = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "rerank_scores": [],
        "response": {},
        "validation_passed": False,
        "retry_count": 0,
        "validation_reason": "",
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]


# ── 9. Export Mermaid diagram ─────────────────────────────────────────────────
def export_rag_graph_as_mermaid_png(
    output_path: str = "src/api/v1/agents/rag_graph.png",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        graph_image = rag_graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_image)
        print(f"[export] Mermaid PNG saved → {output_path}")
    except Exception as e:
        mmd_path = output_path.replace(".png", ".mmd")
        mermaid_source = rag_graph.get_graph().draw_mermaid()
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(mermaid_source)
        print(f"[export] PNG render failed ({e}); Mermaid source saved → {mmd_path}")
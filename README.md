# capstone2

# project structure

multimodal-rag-system1/
├── main.py                          # FastAPI app entry point
├── pyproject.toml                   # Dependencies and project metadata
├── .env.example                     # Environment variable template
├── data/                            # Source documents (PDFs, images)
├── src/
│   ├── core/
│   │   └── db.py                    # PGVector store & embedding config
│   ├── ingestion/
│   │   ├── docling_parser.py        # Docling-based document parser
│   │   └── ingestion.py             # Ingestion pipeline orchestrator
│   ├── query/                       # Query pipeline (retrieval + generation)
│   └── api/
│       └── v1/
│           ├── routes/
│           │   └── query.py         # API route definitions
│           ├── schemas/
│           │   └── query_schema.py  # Pydantic request/response models
│           └── services/
│           |    └── query_service.py # RAG query logic
|           |__ agents/
|                  |___agents.py
|                  |__ embeddings.py
|                  |__ retrieval.py

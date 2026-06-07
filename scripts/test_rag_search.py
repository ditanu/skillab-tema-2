"""Manual RAG smoke test.

Run from the project root after PostgreSQL is up and migrations are applied:
    PYTHONPATH=src python scripts/test_rag_search.py
"""

from extraction_pipeline import RAGService
from extraction_pipeline.persistence import transaction


def main() -> None:
    query = "Ce spune contractul despre reziliere?"

    with transaction() as db:
        rag = RAGService(db)
        results = rag.search(query, top_k=3)

        print(f"Query: {query}")
        for chunk, score in results:
            preview = " ".join(chunk.content.split())[:120]
            print(
                f"- {chunk.document.filename} | chunk {chunk.chunk_index} "
                f"| score {score:.3f} | {preview}..."
            )


if __name__ == "__main__":
    main()

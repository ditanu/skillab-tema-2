"""RAG search service backed by sentence-transformers and pgvector."""

from __future__ import annotations

from sqlalchemy.orm import Session

from extraction_pipeline.persistence.models import DocumentChunk
from extraction_pipeline.persistence.repositories import DocumentChunkRepository
from extraction_pipeline.processing.embeddings import (
    MODEL_NAME,
    SentenceTransformerEmbeddingService,
)


DEFAULT_THRESHOLD = 0.4


class RAGService:
    """Encode queries and retrieve the most similar persisted document chunks."""

    def __init__(
        self,
        db: Session,
        embedding_service: SentenceTransformerEmbeddingService | None = None,
    ) -> None:
        self.repo = DocumentChunkRepository(db)
        self.embedding_service = embedding_service or SentenceTransformerEmbeddingService(
            MODEL_NAME
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Return top-k chunks ranked by cosine similarity."""
        query_embedding = self.embedding_service.encode(query)
        return self.repo.search_similar(query_embedding, limit=top_k)  # type: ignore[arg-type]

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> str:
        """Build a prompt-ready context string from relevant chunks."""
        results = self.search(query=query, top_k=top_k)
        relevant = [(chunk, score) for chunk, score in results if score >= threshold]

        return "\n\n".join(
            f"[{chunk.document.filename} | chunk {chunk.chunk_index} | score {score:.2f}]\n"
            f"{chunk.content}"
            for chunk, score in relevant
        )

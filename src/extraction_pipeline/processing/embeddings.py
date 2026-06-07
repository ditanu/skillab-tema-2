"""Sentence-transformers embedding helpers for RAG chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768


class SentenceTransformerEmbeddingService:
    """Lazy singleton wrapper around the course multilingual embedding model."""

    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        """Load the model only when embeddings are actually needed."""
        if SentenceTransformerEmbeddingService._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for RAG embeddings. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

            SentenceTransformerEmbeddingService._model = SentenceTransformer(self.model_name)

        return SentenceTransformerEmbeddingService._model

    def encode(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """Encode one text or a batch using normalized 768-dimensional embeddings."""
        single_input = isinstance(texts, str)
        values = [texts] if single_input else texts

        embeddings = self.model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = embeddings.tolist()

        if single_input:
            return result[0]

        return result


def embed_text(text: str) -> list[float]:
    """Convenience wrapper for one text."""
    result = SentenceTransformerEmbeddingService().encode(text)
    return result  # type: ignore[return-value]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper for a batch of texts."""
    result = SentenceTransformerEmbeddingService().encode(texts)
    return result  # type: ignore[return-value]

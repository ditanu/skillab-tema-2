"""Repository classes for document and chunk CRUD operations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import cast, delete as sql_delete, func, select, update as sql_update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, joinedload, selectinload

from extraction_pipeline.persistence.exceptions import (
    DocumentChunkNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidMetadataError,
)
from extraction_pipeline.persistence.models import Document, DocumentChunk


class DocumentRepository:
    """Single API for all Document database operations."""

    _UPDATABLE = {"filename", "content", "doc_metadata"}

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        filename: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create one document. The surrounding transaction owns commit/rollback."""
        self._validate_metadata(metadata or {})
        doc = Document(filename=filename, content=content, doc_metadata=metadata or {})
        self.db.add(doc)
        self.db.flush()
        self.db.refresh(doc)
        return doc

    def create_safe(
        self,
        filename: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create one document and raise domain errors for invalid input."""
        if self.get_by_filename(filename) is not None:
            raise DuplicateDocumentError(filename)

        return self.create(filename=filename, content=content, metadata=metadata or {})

    def create_with_chunks(
        self,
        filename: str,
        content: str,
        metadata: dict[str, Any] | None,
        chunks: list[dict[str, Any]],
    ) -> Document:
        """Create a document and its chunks in the same transaction."""
        doc = self.create_safe(filename=filename, content=content, metadata=metadata or {})

        for index, chunk_data in enumerate(chunks):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=chunk_data.get("chunk_index", index),
                content=chunk_data["content"],
                token_count=chunk_data.get("token_count"),
                embedding=chunk_data.get("embedding"),
                chunk_metadata=chunk_data.get("metadata", {}),
            )
            self.db.add(chunk)

        self.db.flush()
        self.db.refresh(doc)
        return doc

    def create_batch(self, documents: list[dict[str, Any]]) -> list[Document]:
        """Create many documents efficiently."""
        docs = [
            Document(
                filename=item["filename"],
                content=item["content"],
                doc_metadata=item.get("metadata", item.get("doc_metadata", {})),
            )
            for item in documents
        ]
        self.db.add_all(docs)
        self.db.flush()
        for doc in docs:
            self.db.refresh(doc)
        return docs

    def get_by_id(self, doc_id: int) -> Document | None:
        """Return a document by primary key, or None."""
        return self.db.get(Document, doc_id)

    def get_by_id_safe(self, doc_id: int) -> Document:
        """Return a document by id or raise DocumentNotFoundError."""
        doc = self.get_by_id(doc_id)
        if doc is None:
            raise DocumentNotFoundError(doc_id)
        return doc

    def get_with_chunks(self, doc_id: int) -> Document | None:
        """Return one document with chunks loaded."""
        statement = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == doc_id)
        )
        return self.db.execute(statement).scalar_one_or_none()

    def get_by_filename(self, filename: str) -> Document | None:
        """Return the first document matching a filename."""
        statement = select(Document).where(Document.filename == filename)
        return self.db.execute(statement).scalar_one_or_none()

    def get_all(self, skip: int = 0, limit: int = 100) -> tuple[list[Document], int]:
        """Return paginated documents plus total count."""
        total = self.count()
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(statement).scalars().all()), total

    def count(self) -> int:
        """Return the total number of documents."""
        return self.db.scalar(select(func.count(Document.id))) or 0

    def filter_by_metadata(self, key: str, value: str) -> list[Document]:
        """Filter documents by a JSONB metadata key."""
        statement = select(Document).where(Document.doc_metadata[key].as_string() == value)
        return list(self.db.execute(statement).scalars().all())

    def search(self, query: str, limit: int = 20) -> list[Document]:
        """Full-text search over filename and content using the generated tsvector."""
        ts_query = func.plainto_tsquery("romanian", query)
        statement = (
            select(Document)
            .where(Document.search_vector.op("@@")(ts_query))
            .order_by(func.ts_rank(Document.search_vector, ts_query).desc())
            .limit(limit)
        )
        return list(self.db.execute(statement).scalars().all())

    def update(self, doc_id: int, **fields: Any) -> Document | None:
        """Update allowed fields on a document."""
        clean = self._clean_update_fields(fields)
        if not clean:
            return self.get_by_id(doc_id)

        self.db.execute(sql_update(Document).where(Document.id == doc_id).values(**clean))
        self.db.flush()
        return self.get_by_id(doc_id)

    def update_metadata(self, doc_id: int, metadata: dict[str, Any]) -> Document | None:
        """Merge metadata atomically with PostgreSQL JSONB ||."""
        self._validate_metadata(metadata)
        self.db.execute(
            sql_update(Document)
            .where(Document.id == doc_id)
            .values(doc_metadata=Document.doc_metadata.op("||")(cast(metadata, JSONB)))
        )
        self.db.flush()
        return self.get_by_id(doc_id)

    def delete(self, doc_id: int) -> bool:
        """Delete one document. Related chunks are deleted by cascade."""
        result = self.db.execute(sql_delete(Document).where(Document.id == doc_id))
        self.db.flush()
        return (result.rowcount or 0) > 0

    def delete_batch(self, doc_ids: list[int]) -> int:
        """Delete many documents by id."""
        if not doc_ids:
            return 0

        result = self.db.execute(sql_delete(Document).where(Document.id.in_(doc_ids)))
        self.db.flush()
        return result.rowcount or 0

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise InvalidMetadataError("metadata must be a dict")

    def _clean_update_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        if "metadata" in fields:
            fields["doc_metadata"] = fields.pop("metadata")

        clean = {key: value for key, value in fields.items() if key in self._UPDATABLE}
        if "doc_metadata" in clean:
            self._validate_metadata(clean["doc_metadata"])
        return clean


class DocumentChunkRepository:
    """Single API for all DocumentChunk database operations."""

    _UPDATABLE = {"chunk_index", "content", "token_count", "embedding", "chunk_metadata"}

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
        embedding: list[float] | None = None,
    ) -> DocumentChunk:
        """Create one chunk for a document."""
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            embedding=embedding,
            chunk_metadata=metadata or {},
        )
        self.db.add(chunk)
        self.db.flush()
        self.db.refresh(chunk)
        return chunk

    def create_batch(self, chunks: list[dict[str, Any]]) -> list[DocumentChunk]:
        """Create many chunks efficiently."""
        db_chunks = [
            DocumentChunk(
                document_id=item["document_id"],
                chunk_index=item["chunk_index"],
                content=item["content"],
                token_count=item.get("token_count"),
                embedding=item.get("embedding"),
                chunk_metadata=item.get("metadata", item.get("chunk_metadata", {})),
            )
            for item in chunks
        ]
        self.db.add_all(db_chunks)
        self.db.flush()
        for chunk in db_chunks:
            self.db.refresh(chunk)
        return db_chunks

    def get_by_id(self, chunk_id: int) -> DocumentChunk | None:
        """Return a chunk by primary key, or None."""
        return self.db.get(DocumentChunk, chunk_id)

    def get_by_id_safe(self, chunk_id: int) -> DocumentChunk:
        """Return a chunk by id or raise DocumentChunkNotFoundError."""
        chunk = self.get_by_id(chunk_id)
        if chunk is None:
            raise DocumentChunkNotFoundError(chunk_id)
        return chunk

    def get_for_document(self, document_id: int) -> list[DocumentChunk]:
        """Return chunks for one document in their natural order."""
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(self.db.execute(statement).scalars().all())

    def search(self, query: str, limit: int = 20) -> list[DocumentChunk]:
        """Full-text search over chunk content."""
        ts_query = func.plainto_tsquery("romanian", query)
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.search_vector.op("@@")(ts_query))
            .order_by(func.ts_rank(DocumentChunk.search_vector, ts_query).desc())
            .limit(limit)
        )
        return list(self.db.execute(statement).scalars().all())

    def search_similar(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Vector similarity search using pgvector cosine distance."""
        distance = DocumentChunk.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("score")
        statement = (
            select(DocumentChunk, similarity)
            .options(joinedload(DocumentChunk.document))
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        return [(chunk, float(score)) for chunk, score in self.db.execute(statement).all()]

    def search_similar_with_threshold(
        self,
        embedding: list[float],
        min_score: float = 0.4,
        limit: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return cosine search results above a minimum similarity score."""
        return [
            (chunk, score)
            for chunk, score in self.search_similar(embedding=embedding, limit=limit)
            if score >= min_score
        ]

    def update(self, chunk_id: int, **fields: Any) -> DocumentChunk | None:
        """Update allowed fields on a chunk."""
        clean = self._clean_update_fields(fields)
        if not clean:
            return self.get_by_id(chunk_id)

        self.db.execute(
            sql_update(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
            .values(**clean)
        )
        self.db.flush()
        return self.get_by_id(chunk_id)

    def update_metadata(self, chunk_id: int, metadata: dict[str, Any]) -> DocumentChunk | None:
        """Merge chunk metadata atomically with PostgreSQL JSONB ||."""
        if not isinstance(metadata, dict):
            raise InvalidMetadataError("metadata must be a dict")

        self.db.execute(
            sql_update(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
            .values(chunk_metadata=DocumentChunk.chunk_metadata.op("||")(cast(metadata, JSONB)))
        )
        self.db.flush()
        return self.get_by_id(chunk_id)

    def delete(self, chunk_id: int) -> bool:
        """Delete one chunk."""
        result = self.db.execute(sql_delete(DocumentChunk).where(DocumentChunk.id == chunk_id))
        self.db.flush()
        return (result.rowcount or 0) > 0

    def delete_for_document(self, document_id: int) -> int:
        """Delete all chunks belonging to a document."""
        result = self.db.execute(
            sql_delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        self.db.flush()
        return result.rowcount or 0

    def _clean_update_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        if "metadata" in fields:
            fields["chunk_metadata"] = fields.pop("metadata")

        clean = {key: value for key, value in fields.items() if key in self._UPDATABLE}
        if "chunk_metadata" in clean and not isinstance(clean["chunk_metadata"], dict):
            raise InvalidMetadataError("metadata must be a dict")
        return clean

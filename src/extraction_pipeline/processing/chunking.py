"""Chunking logic for long documents."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
COURSE_SEPARATORS = ["\n\n", "\n", " ", ""]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separators=COURSE_SEPARATORS,
)


def should_chunk(document_text: str, max_size: int = 4000) -> bool:
    """Return True when the document is longer than the configured threshold."""
    return len(document_text) > max_size


def split_text(document_text: str, chunk_size: int | None = None) -> list[str]:
    """Split text into chunks, using the default course splitter unless overridden."""
    return _get_splitter(chunk_size).split_text(document_text)


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Split LangChain Documents while preserving and enriching metadata."""
    chunks = _get_splitter(chunk_size, chunk_overlap).split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata.update(metadata or {})
        chunk.metadata["chunk_id"] = index

    return chunks


def documents_to_text(documents: list[Document]) -> str:
    """Join LangChain document pages into one extraction input."""
    return "\n\n".join(doc.page_content for doc in documents if doc.page_content)


def _get_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return the course splitter or a same-style splitter with a custom chunk size."""
    if chunk_size is None and chunk_overlap is None:
        return splitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP,
        separators=COURSE_SEPARATORS,
    )

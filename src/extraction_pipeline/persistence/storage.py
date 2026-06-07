"""Bridge between the L3 document loader/chunker and the L4 repositories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy.orm import Session

from extraction_pipeline.persistence.models import Document
from extraction_pipeline.persistence.repositories import DocumentRepository
from extraction_pipeline.processing.chunking import chunk_documents, documents_to_text, should_chunk
from extraction_pipeline.processing.embeddings import embed_texts
from extraction_pipeline.processing.loaders import load_document
from extraction_pipeline.processing.loaders import LOADER_REGISTRY
from extraction_pipeline.processing.registry import EXTRACTION_REGISTRY


def store_document_file(
    file_path: str,
    doc_type: str | None,
    db: Session,
    replace_existing: bool = False,
) -> Document:
    """Load a file, prepare chunks, and save Document + DocumentChunk rows."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fișierul nu există: {file_path}")

    repo = DocumentRepository(db)
    existing = repo.get_by_filename(path.name)
    if existing is not None and not replace_existing:
        return existing

    if existing is not None:
        repo.delete(existing.id)

    documents = load_document(file_path)
    document_text = documents_to_text(documents)

    if not document_text.strip():
        raise ValueError("Documentul nu conține text extractibil.")

    resolved_doc_type = doc_type or infer_doc_type(path, document_text)
    if resolved_doc_type not in EXTRACTION_REGISTRY:
        raise ValueError(f"Tip document nesuportat: {resolved_doc_type}")

    route = EXTRACTION_REGISTRY[resolved_doc_type]
    metadata = {
        "doc_type": resolved_doc_type,
        "language": "ro",
        "source": str(path),
        "extension": path.suffix.lower(),
        "file_hash": _file_hash(path),
    }
    chunks = _prepare_chunks(
        documents=documents,
        document_text=document_text,
        metadata=metadata,
        chunk_size=route["chunk_size"],
        chunk_overlap=route.get("chunk_overlap"),
    )
    _attach_embeddings(chunks)

    return repo.create_with_chunks(
        filename=path.name,
        content=document_text,
        metadata=metadata,
        chunks=chunks,
    )


def store_document_auto(file_path: str, db: Session) -> Document:
    """Store one supported file and infer factura/contract routing automatically."""
    return store_document_file(file_path=file_path, doc_type=None, db=db)


def store_documents_from_path(
    path: str,
    db: Session,
    doc_type: str | None = None,
    recursive: bool = False,
    replace_existing: bool = False,
) -> list[Document]:
    """Store one file or all supported files from a directory."""
    root = Path(path)

    if root.is_file():
        return [
            store_document_file(
                str(root),
                doc_type=doc_type,
                db=db,
                replace_existing=replace_existing,
            )
        ]

    if not root.exists():
        raise FileNotFoundError(f"Calea nu există: {path}")

    pattern = "**/*" if recursive else "*"
    files = [
        candidate
        for candidate in sorted(root.glob(pattern))
        if candidate.is_file() and candidate.suffix.lower() in LOADER_REGISTRY
    ]

    return [
        store_document_file(
            str(file_path),
            doc_type=doc_type,
            db=db,
            replace_existing=replace_existing,
        )
        for file_path in files
    ]


def infer_doc_type(path: Path, document_text: str = "") -> str:
    """Infer the extraction route from filename first, then from document text."""
    haystack = f"{path.name}\n{document_text[:4000]}".lower()

    invoice_markers = (
        "factura",
        "facturi",
        "invoice",
        "furnizor",
        "client",
        "total",
        "tva",
    )
    contract_markers = (
        "contract",
        "agreement",
        "service agreement",
        "servicii",
        "consultanta",
        "prestator",
        "beneficiar",
        "termination",
        "reziliere",
    )

    invoice_score = sum(marker in haystack for marker in invoice_markers)
    contract_score = sum(marker in haystack for marker in contract_markers)

    if invoice_score > contract_score:
        return "factura"

    if contract_score > 0:
        return "contract"

    raise ValueError(
        f"Nu pot determina tipul documentului pentru {path}. "
        "Trimite doc_type='factura' sau doc_type='contract'."
    )


def _prepare_chunks(
    documents: list[LangChainDocument],
    document_text: str,
    metadata: dict[str, Any],
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[dict[str, Any]]:
    if should_chunk(document_text):
        chunked_documents = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata,
        )
    else:
        chunked_documents = [
            LangChainDocument(page_content=document_text, metadata={**metadata, "chunk_id": 0})
        ]

    return [
        {
            "chunk_index": index,
            "content": chunk.page_content,
            "metadata": dict(chunk.metadata),
        }
        for index, chunk in enumerate(chunked_documents)
        if chunk.page_content
    ]


def _attach_embeddings(chunks: list[dict[str, Any]]) -> None:
    """Generate one sentence-transformers embedding per chunk in a single batch."""
    if not chunks:
        return

    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk["embedding"] = embedding


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

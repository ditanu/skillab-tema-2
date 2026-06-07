"""Document loading, chunking, routing, and LLM extraction helpers."""

from extraction_pipeline.processing.chunking import (
    chunk_documents,
    documents_to_text,
    should_chunk,
    split_text,
)
from extraction_pipeline.processing.extractor import extract_with_schema
from extraction_pipeline.processing.loaders import load_document
from extraction_pipeline.processing.registry import EXTRACTION_REGISTRY

__all__ = [
    "EXTRACTION_REGISTRY",
    "chunk_documents",
    "documents_to_text",
    "extract_with_schema",
    "load_document",
    "should_chunk",
    "split_text",
]

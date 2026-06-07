"""Manual chunking smoke test.

Run from the project root:
    PYTHONPATH=src python scripts/test_chunking.py
"""

from pathlib import Path

from extraction_pipeline.processing.chunking import chunk_documents, documents_to_text, should_chunk
from extraction_pipeline.processing.loaders import load_document
from extraction_pipeline.processing.registry import EXTRACTION_REGISTRY


SAMPLE_DIR = Path("sample_docs")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def infer_doc_type(path: Path) -> str | None:
    """Infer document type from the sample filename."""
    name = path.name.lower()

    if "factura" in name or "invoice" in name:
        return "factura"

    if "contract" in name or "agreement" in name or "service" in name:
        return "contract"

    return None


def main() -> None:
    print("CHUNKING TEST")
    print("=" * 80)

    sample_files = [
        path
        for path in sorted(SAMPLE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for path in sample_files:
        doc_type = infer_doc_type(path)

        if doc_type is None:
            print(f"\nFile: {path}")
            print("Skipped: cannot infer a registered document type from filename.")
            continue

        chunk_size = EXTRACTION_REGISTRY[doc_type]["chunk_size"]
        chunk_overlap = EXTRACTION_REGISTRY[doc_type]["chunk_overlap"]

        docs = load_document(str(path))
        text = documents_to_text(docs)
        pipeline_chunks = (
            chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if should_chunk(text)
            else docs
        )

        print(f"\nFile: {path}")
        print(f"Type: {doc_type}")
        print(f"Characters: {len(text)}")
        print(f"Configured chunk_size: {chunk_size}")
        print(f"Configured chunk_overlap: {chunk_overlap}")
        print(f"Pipeline should_chunk(text, 4000): {should_chunk(text)}")
        print(f"Pipeline chunks produced: {len(pipeline_chunks)}")

        for index, chunk in enumerate(pipeline_chunks, start=1):
            preview = " ".join(chunk.page_content.split())[:90]
            print(f"  - chunk {index}: {len(chunk.page_content)} chars | {preview}...")

    csv_path = Path("sample_docs/facturi_export.csv")
    print(f"\nCSV sample present: {csv_path.exists()} ({csv_path})")
    print("CSV is not loaded by the current assignment registry: only .pdf, .docx, .txt.")


if __name__ == "__main__":
    main()

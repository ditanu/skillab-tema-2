"""Ingest supported documents dynamically into PostgreSQL + pgvector.

Examples:
    PYTHONPATH=src python scripts/ingest_documents.py sample_docs
    PYTHONPATH=src python scripts/ingest_documents.py sample_docs/sample-service-agreement.pdf
"""

from __future__ import annotations

import argparse

from extraction_pipeline.persistence import store_documents_from_path, transaction
from extraction_pipeline.processing.loaders import LOADER_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Store documents with chunk embeddings.")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--doc-type",
        choices=["factura", "contract"],
        default=None,
        help="Optional override. By default the type is inferred.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete matching existing documents first, then chunk and embed them again.",
    )
    args = parser.parse_args()

    with transaction() as db:
        documents = store_documents_from_path(
            args.path,
            db=db,
            doc_type=args.doc_type,
            recursive=args.recursive,
            replace_existing=args.rebuild,
        )

        print(f"Supported extensions: {', '.join(sorted(LOADER_REGISTRY))}")
        print(f"Stored documents: {len(documents)}")
        for doc in documents:
            print(f"- id={doc.id} file={doc.filename} chunks={len(doc.chunks)}")


if __name__ == "__main__":
    main()

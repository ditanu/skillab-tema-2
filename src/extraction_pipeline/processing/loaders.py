"""Document loading utilities using the Loader Registry Pattern."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader


LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}


def load_document(path: str) -> list[Document]:
    """Load a document with the loader registered for its file extension."""
    ext = Path(path).suffix.lower()

    if ext not in LOADER_REGISTRY:
        raise ValueError(f"Format nesuportat: {ext}")

    loader_cls = LOADER_REGISTRY[ext]
    return loader_cls(path).load()

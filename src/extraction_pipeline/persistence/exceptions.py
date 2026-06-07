"""Custom exceptions for the document persistence layer."""


class DocumentError(Exception):
    """Base class for all document persistence errors."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document id cannot be found."""

    def __init__(self, doc_id: int) -> None:
        super().__init__(f"Document with id={doc_id} not found")
        self.doc_id = doc_id


class DuplicateDocumentError(DocumentError):
    """Raised when a unique filename already exists."""

    def __init__(self, filename: str) -> None:
        super().__init__(f"Document with filename='{filename}' already exists")
        self.filename = filename


class InvalidMetadataError(DocumentError):
    """Raised when metadata is not a dictionary."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid metadata: {reason}")
        self.reason = reason


class DocumentChunkNotFoundError(DocumentError):
    """Raised when a chunk id cannot be found."""

    def __init__(self, chunk_id: int) -> None:
        super().__init__(f"DocumentChunk with id={chunk_id} not found")
        self.chunk_id = chunk_id

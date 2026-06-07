"""Registries for loaders and document-type routing."""

from extraction_pipeline.domain.schemas import Contract, Invoice


EXTRACTION_REGISTRY = {
    "factura": {
        "schema": Invoice,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "output_dir": "facturi",
        "file_prefix": "factura",
    },
    "contract": {
        "schema": Contract,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "output_dir": "contracte",
        "file_prefix": "contract",
    },
}

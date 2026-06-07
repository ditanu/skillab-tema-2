"""sentence transformer embeddings and hnsw index

Revision ID: 0002_st_embeddings_hnsw
Revises: 0001_create_documents_and_chunks
Create Date: 2026-06-01
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "0002_st_embeddings_hnsw"
down_revision: str | None = "0001_create_documents_and_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    op.execute("UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(384),
        existing_type=Vector(1536),
        existing_nullable=True,
        postgresql_using="embedding::vector(384)",
    )
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding_hnsw", table_name="document_chunks")
    op.execute("UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(1536),
        existing_type=Vector(384),
        existing_nullable=True,
        postgresql_using="embedding::vector(1536)",
    )
    op.create_index(
        "ix_document_chunks_embedding",
        "document_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="ivfflat",
        postgresql_with={"lists": 100},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )

"""switch sentence-transformer embeddings to 768 dimensions

Revision ID: 0004_mpnet_embeddings_768
Revises: 0003_reset_sample_documents
Create Date: 2026-06-07
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "0004_mpnet_embeddings_768"
down_revision: str | None = "0003_reset_sample_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(768),
        existing_type=Vector(384),
        existing_nullable=True,
        postgresql_using="NULL::vector(768)",
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
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(384),
        existing_type=Vector(768),
        existing_nullable=True,
        postgresql_using="NULL::vector(384)",
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

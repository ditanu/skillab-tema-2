"""create documents and chunks

Revision ID: 0001_create_documents_and_chunks
Revises:
Create Date: 2026-05-31
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_create_documents_and_chunks"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('romanian', coalesce(filename, '') || ' ' || coalesce(content, ''))",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("filename", name="uq_documents_filename"),
    )
    op.create_index("ix_documents_filename", "documents", ["filename"], unique=False)
    op.create_index(
        "ix_documents_search_vector",
        "documents",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_documents_metadata",
        "documents",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('romanian', coalesce(content, ''))", persisted=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_document_chunks_metadata",
        "document_chunks",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
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


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")
    op.drop_index("ix_document_chunks_metadata", table_name="document_chunks")
    op.drop_index("ix_document_chunks_search_vector", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_index("ix_documents_metadata", table_name="documents")
    op.drop_index("ix_documents_search_vector", table_name="documents")
    op.drop_index("ix_documents_filename", table_name="documents")
    op.drop_table("documents")

    op.execute("DROP EXTENSION IF EXISTS vector")

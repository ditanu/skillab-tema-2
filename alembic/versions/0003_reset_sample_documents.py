"""reset stored documents

Revision ID: 0003_reset_sample_documents
Revises: 0002_st_embeddings_hnsw
Create Date: 2026-06-01
"""

from collections.abc import Sequence

from alembic import op


revision: str = "0003_reset_sample_documents"
down_revision: str | None = "0002_st_embeddings_hnsw"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Development/data reset: deletes stored documents and cascades to chunks.
    op.execute("TRUNCATE TABLE documents RESTART IDENTITY CASCADE")


def downgrade() -> None:
    # Deleted document data cannot be reconstructed by a downgrade.
    pass

import ast
import operator
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

from tools.registry import register_tool
from tools.params_models import (
    CalculatorParams,
    DateTimeParams,
    DocumentSearchParams,
    KnowledgeSearchParams,
)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_MONTHS_RO = {
    1: "ianuarie",
    2: "februarie",
    3: "martie",
    4: "aprilie",
    5: "mai",
    6: "iunie",
    7: "iulie",
    8: "august",
    9: "septembrie",
    10: "octombrie",
    11: "noiembrie",
    12: "decembrie",
}


def _safe_eval_math(expression: str) -> float | int:
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Operator nepermis: {op_type.__name__}")

            left = eval_node(node.left)
            right = eval_node(node.right)

            return _ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Operator unar nepermis: {op_type.__name__}")

            operand = eval_node(node.operand)

            return _ALLOWED_OPERATORS[op_type](operand)

        raise ValueError("Expresie matematică nepermisă.")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed.body)


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluează expresii matematice simple atunci când este nevoie de calcule precise."""
    result = _safe_eval_math(params.expression)
    return str(result)


@register_tool
def get_current_datetime(params: DateTimeParams) -> str:
    """Returnează data și ora curentă pentru întrebări care depind de timp real."""
    now = datetime.now(ZoneInfo(params.timezone))
    month = _MONTHS_RO[now.month]
    return (
        f"Data curentă: {now.day} {month} {now.year}\n"
        f"Ora curentă: {now:%H:%M}\n"
        f"Fus orar: {params.timezone}"
    )


def _ensure_extraction_pipeline_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _search_documents_with_rag(params: DocumentSearchParams) -> str:
    _ensure_extraction_pipeline_importable()

    try:
        from extraction_pipeline import RAGService
        from extraction_pipeline.persistence.models import DocumentChunk
        from extraction_pipeline.persistence import transaction
        from sqlalchemy import func, select, text
        from sqlalchemy.orm import joinedload
    except Exception as error:
        return (
            "RAG nu este disponibil: nu pot importa pipeline-ul de extracție. "
            f"Detalii: {error}"
        )

    try:
        with transaction() as db:
            db.execute(text("SELECT 1"))
            embedded_chunks = db.scalar(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.embedding.is_not(None)
                )
            )

            if not embedded_chunks:
                return (
                    "Nu există fragmente cu embeddings în baza de date. "
                    "Rulează ingestia documentelor înainte de a folosi RAG."
                )

            rag = RAGService(db)
            requested_results = max(params.max_results, 8)
            results = rag.search(query=params.query, top_k=max(requested_results * 2, 12))
            expanded_results = _expand_results_with_neighbors(
                db=db,
                results=results,
                document_chunk_cls=DocumentChunk,
                joinedload=joinedload,
                limit=requested_results,
            )
    except Exception as error:
        return (
            "RAG nu este disponibil sau baza de date nu este pregătită. "
            "Verifică PostgreSQL, migrațiile Alembic și documentele ingestate. "
            f"Detalii: {error}"
        )

    relevant_results = [
        (chunks, score) for chunks, score in expanded_results if score >= params.min_score
    ]

    if not relevant_results:
        return (
            "Nu am găsit fragmente relevante în documentele încărcate pentru întrebarea dată."
        )

    formatted_results = []

    for index, (chunks, score) in enumerate(relevant_results, start=1):
        main_chunk = chunks[len(chunks) // 2]
        filename = (
            main_chunk.document.filename
            if main_chunk.document
            else f"document #{main_chunk.document_id}"
        )
        chunk_range = f"{chunks[0].chunk_index}-{chunks[-1].chunk_index}"
        preview = "\n\n".join(" ".join(chunk.content.split()) for chunk in chunks)
        formatted_results.append(
            "\n".join(
                [
                    f"Rezultat {index}:",
                    f"Sursă: {filename}",
                    f"Chunk-uri incluse: {chunk_range}",
                    f"Scor: {score:.3f}",
                    "Conținut:",
                    preview,
                ]
            )
        )

    return "\n\n".join(formatted_results)


def _expand_results_with_neighbors(
    db,
    results,
    document_chunk_cls,
    joinedload,
    limit: int,
    neighbor_window: int = 1,
) -> list[tuple[list, float]]:
    from sqlalchemy import select

    expanded = []
    seen_chunks = set()

    for chunk, score in results:
        if len(expanded) >= limit:
            break

        if chunk.id in seen_chunks:
            continue

        start = max(0, chunk.chunk_index - neighbor_window)
        end = chunk.chunk_index + neighbor_window
        statement = (
            select(document_chunk_cls)
            .options(joinedload(document_chunk_cls.document))
            .where(document_chunk_cls.document_id == chunk.document_id)
            .where(document_chunk_cls.chunk_index >= start)
            .where(document_chunk_cls.chunk_index <= end)
            .order_by(document_chunk_cls.chunk_index)
        )
        neighbor_chunks = list(db.execute(statement).scalars().all())

        for neighbor in neighbor_chunks:
            seen_chunks.add(neighbor.id)

        expanded.append((neighbor_chunks or [chunk], score))

    return expanded


def _search_documents_langchain(
    query: str,
    max_results: int = 8,
    min_score: float = 0.4,
) -> str:
    return _search_documents_with_rag(
        DocumentSearchParams(
            query=query,
            max_results=max_results,
            min_score=min_score,
        )
    )


search_documents_tool = StructuredTool.from_function(
    func=_search_documents_langchain,
    name="search_documents",
    description="Caută semantic în documentele încărcate în PostgreSQL + pgvector.",
    args_schema=DocumentSearchParams,
)


@register_tool
def search_documents(params: DocumentSearchParams) -> str:
    """Caută semantic în documentele încărcate în PostgreSQL + pgvector."""
    return _search_documents_with_rag(params)


@register_tool
def search_knowledge_base(params: KnowledgeSearchParams) -> str:
    """Compatibilitate: caută în documentele încărcate folosind RAG-ul real."""
    return _search_documents_with_rag(
        DocumentSearchParams(
            query=params.query,
            max_results=params.max_results,
        )
    )

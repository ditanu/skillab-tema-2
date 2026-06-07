from pydantic import BaseModel, Field


class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Expresia matematică de evaluat, de exemplu: '2 + 3 * 4'.",
        min_length=1,
    )


class DateTimeParams(BaseModel):
    timezone: str = Field(
        default="Europe/Bucharest",
        description="Fusul orar pentru data și ora curentă.",
    )


class KnowledgeSearchParams(BaseModel):
    query: str = Field(
        description=(
            "Întrebarea sau termenii de căutat în documentele persistate prin RAG."
        ),
        min_length=2,
    )
    max_results: int = Field(
        default=8,
        description="Numărul maxim de documente relevante returnate.",
        ge=1,
        le=10,
    )


class DocumentSearchParams(BaseModel):
    query: str = Field(
        description="Întrebarea sau termenii de căutat în documentele încărcate.",
        min_length=2,
    )
    max_results: int = Field(
        default=8,
        description="Numărul maxim de fragmente relevante returnate.",
        ge=1,
        le=10,
    )
    min_score: float = Field(
        default=0.4,
        description="Scorul minim de similaritate cosine pentru rezultate.",
        ge=0.0,
        le=1.0,
    )

"""LLM structured extraction helpers."""

from typing import Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError


SYSTEM_PROMPT = """Ești expert în extragerea datelor din documente.

Extrage toate câmpurile cerute.

Dacă o valoare nu există în document,
setează null și nu inventa informații."""


def extract_with_schema(
    text: str,
    schema: Type[BaseModel],
    llm: BaseChatModel,
    max_retries: int = 3,
) -> BaseModel:
    """Extract structured data from text using a LangChain structured output model."""
    structured_llm = llm.with_structured_output(schema)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Document:\n\n{text}"),
    ]
    last_validation_error: ValidationError | None = None

    for _ in range(max_retries):
        try:
            result = structured_llm.invoke(messages)

            if isinstance(result, schema):
                return result

            return schema.model_validate(result)
        except ValidationError as exc:
            last_validation_error = exc

    if last_validation_error is not None:
        raise last_validation_error

    raise RuntimeError("Extracția structurată nu a returnat un rezultat valid.")

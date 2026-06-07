"""Main LOAD -> CHUNK -> EXTRACT -> SAVE pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from extraction_pipeline.processing.chunking import chunk_documents, documents_to_text, should_chunk
from extraction_pipeline.processing.extractor import extract_with_schema
from extraction_pipeline.processing.loaders import load_document
from extraction_pipeline.processing.registry import EXTRACTION_REGISTRY


class ExtractionPipeline:
    """Production-ready document extraction pipeline."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        output_root: str = "extracted_data",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
    ) -> None:
        self._llm = llm
        self.model = model
        self.temperature = temperature
        self.output_root = Path(output_root)

    @property
    def llm(self) -> BaseChatModel:
        """Create the LLM lazily, which keeps the pipeline safer in batch workers."""
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        return self._llm

    def process(self, file_path: str, doc_type: str) -> dict[str, Any]:
        """Run LOAD -> CHUNK -> EXTRACT -> SAVE and always return a safe response."""
        try:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Fișierul nu există: {file_path}")

            if doc_type not in EXTRACTION_REGISTRY:
                raise ValueError(f"Tip document nesuportat: {doc_type}")

            route = EXTRACTION_REGISTRY[doc_type]
            schema = route["schema"]
            chunk_size = route["chunk_size"]
            chunk_overlap = route.get("chunk_overlap")

            documents = load_document(file_path)
            document_text = documents_to_text(documents)

            if not document_text.strip():
                raise ValueError("Documentul nu conține text extractibil.")

            texts_for_extraction = self._prepare_extraction_inputs(
                documents=documents,
                document_text=document_text,
                doc_type=doc_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                path=path,
            )
            extraction_input = "\n\n--- CHUNK ---\n\n".join(texts_for_extraction)

            result = extract_with_schema(extraction_input, schema, self.llm)
            output_path = self._save_result(result, route)

            return {
                "success": True,
                "data": result.model_dump(),
                "output_path": str(output_path),
            }

        except FileNotFoundError as exc:
            return {"success": False, "error": str(exc)}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ValidationError as exc:
            return {"success": False, "error": f"Eroare validare Pydantic: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Eroare neașteptată: {exc}"}

    def process_batch(
        self,
        files_to_process: Iterable[tuple[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Process many documents with the same safe response contract."""
        results: dict[str, list[dict[str, Any]]] = {"success": [], "errors": []}

        for file_path, doc_type in files_to_process:
            result = self.process(file_path, doc_type)
            bucket = "success" if result["success"] else "errors"
            results[bucket].append({"file": file_path, "doc_type": doc_type, "result": result})

        return results

    def _prepare_extraction_inputs(
        self,
        documents: list[Any],
        document_text: str,
        doc_type: str,
        chunk_size: int | None,
        chunk_overlap: int | None,
        path: Path,
    ) -> list[str]:
        """Apply chunking only when the document is above the course threshold."""
        if should_chunk(document_text):
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata={
                    "doc_type": doc_type,
                    "language": "ro",
                    "file_hash": self._file_hash(path),
                },
            )
            return [chunk.page_content for chunk in chunks if chunk.page_content]

        return [document_text]

    def _save_result(self, result: BaseModel, route: dict[str, Any]) -> Path:
        """Save the extracted Pydantic object as pretty UTF-8 JSON."""
        output_dir = self.output_root / route["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self._next_output_path(output_dir, route["file_prefix"])

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                result.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    @staticmethod
    def _next_output_path(output_dir: Path, prefix: str) -> Path:
        """Generate names like factura_001.json without overwriting previous outputs."""
        index = 1

        while True:
            candidate = output_dir / f"{prefix}_{index:03d}.json"
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute a stable hash for deduplication/debug metadata."""
        return hashlib.md5(path.read_bytes()).hexdigest()

"""Batch helpers inspired by the course multiprocessing examples."""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from extraction_pipeline.pipeline import ExtractionPipeline


def _process_one(file_path: str, doc_type: str, output_root: str) -> dict[str, Any]:
    """Create a fresh pipeline per worker so LLM clients are not shared unsafely."""
    pipeline = ExtractionPipeline(output_root=output_root)
    return pipeline.process(file_path, doc_type)


def batch_process(
    files_to_process: Iterable[tuple[str, str]],
    num_workers: int = 4,
    output_root: str = "extracted_data",
) -> dict[str, list[dict[str, Any]]]:
    """Process many documents concurrently and collect successes and errors."""
    results: dict[str, list[dict[str, Any]]] = {"success": [], "errors": []}
    file_pairs = list(files_to_process)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_input = {
            executor.submit(_process_one, file_path, doc_type, output_root): (file_path, doc_type)
            for file_path, doc_type in file_pairs
        }

        for future in as_completed(future_to_input):
            file_path, doc_type = future_to_input[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            bucket = "success" if result["success"] else "errors"
            results[bucket].append(
                {
                    "file": file_path,
                    "doc_type": doc_type,
                    "result": result,
                }
            )

    return results


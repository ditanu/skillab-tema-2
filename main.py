"""Example usage for Extraction Pipeline L3."""

from extraction_pipeline import ExtractionPipeline


def main() -> None:
    pipeline = ExtractionPipeline()

    samples = [
        ("sample_docs/factura_001.txt", "factura"),
        ("sample_docs/factura_002.txt", "factura"),
        ("sample_docs/contract_servicii.txt", "contract"),
        ("sample_docs/contract_consultanta.txt", "contract"),
    ]

    for file_path, doc_type in samples:
        result = pipeline.process(file_path, doc_type)
        print(f"{file_path}: {result}")


if __name__ == "__main__":
    main()

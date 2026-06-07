"""Manual JSON save smoke test.

Run from the project root:
    PYTHONPATH=src python scripts/test_save_json.py
"""

from extraction_pipeline.pipeline import ExtractionPipeline
from extraction_pipeline.domain.schemas import Contract, Invoice
from extraction_pipeline.processing.registry import EXTRACTION_REGISTRY


def main() -> None:
    pipeline = ExtractionPipeline()

    invoice = Invoice(
        numar="FV-TEST-001",
        data="15.03.2024",
        client="SC Digital Services SRL",
        furnizor="SC TechPro Solutions SRL",
        total=18088.00,
        produse=[
            "Laptop Dell Latitude 5540",
            'Monitor LG 27" 4K',
            "Tastatura mecanica Logitech",
        ],
    )

    contract = Contract(
        numar="CS-TEST-001",
        data_incheiere="01.03.2024",
        prestator="SC CloudTech Services SRL",
        beneficiar="SC Manufacturing Pro SRL",
        valoare=53550.00,
        durata_luni=6,
        obligatii_prestator=[
            "Implementare sistem ERP",
            "Migrare date",
            "Suport tehnic si mentenanta",
        ],
    )

    invoice_path = pipeline._save_result(invoice, EXTRACTION_REGISTRY["factura"])
    contract_path = pipeline._save_result(contract, EXTRACTION_REGISTRY["contract"])

    print("JSON files saved:")
    print(f"- {invoice_path}")
    print(f"- {contract_path}")


if __name__ == "__main__":
    main()

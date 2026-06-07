"""Pydantic schemas used for structured extraction."""

from typing import List

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    """
    Schema pentru extracție factură
    """

    numar: str = Field(description="Numărul facturii")
    data: str = Field(description="Data emiterii")
    client: str = Field(description="Numele clientului")
    furnizor: str = Field(description="Numele furnizorului")
    total: float = Field(description="Valoarea totală în RON")

    produse: List[str] = Field(
        default=[],
        description="Lista produselor sau serviciilor",
    )


class Contract(BaseModel):
    """
    Schema pentru extracție contract
    """

    numar: str = Field(description="Număr contract")
    data_incheiere: str = Field(description="Data semnării")
    prestator: str = Field(description="Numele prestatorului")
    beneficiar: str = Field(description="Numele beneficiarului")
    valoare: float = Field(description="Valoarea contractului")
    durata_luni: int = Field(description="Durata în luni")

    obligatii_prestator: List[str] = Field(
        default=[],
        description="Lista obligațiilor prestatorului",
    )


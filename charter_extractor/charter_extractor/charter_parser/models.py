"""Pydantic models for structured clause extraction."""

from __future__ import annotations
from pydantic import BaseModel, Field


class Clause(BaseModel):
    """A single legal clause extracted from the charter party document."""

    id: str = Field(description="Clause number/identifier (e.g. '1', '2', '3(a)')")
    title: str = Field(description="Clause title or heading")
    text: str = Field(description="Full clause text content, excluding strikethrough text")
    section: str = Field(
        default="",
        exclude=True,
        description="Document section (internal use for dedup, excluded from output)",
    )


class ExtractionResult(BaseModel):
    """Container for all extracted clauses, preserving document order."""

    clauses: list[Clause] = Field(
        default_factory=list,
        description="Extracted clauses in the order they appear in the document",
    )

    @property
    def count(self) -> int:
        return len(self.clauses)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

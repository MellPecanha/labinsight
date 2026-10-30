"""Contracts for grounded, non-clinical model explanations."""

from pydantic import BaseModel, ConfigDict, Field


class ExplanationRequest(BaseModel):
    """Question about the experiment, not an individual medical record."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=800)


class RetrievedSource(BaseModel):
    title: str
    source_url: str


class ExplanationResponse(BaseModel):
    answer: str
    sources: list[RetrievedSource]
    limitation: str

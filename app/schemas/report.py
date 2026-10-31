"""Contracts for deterministic comparison with a supplied report interval."""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml.preprocessing.biopro import FEATURE_METADATA


class ReportItem(BaseModel):
    """One result and the reference interval printed on the same report."""

    model_config = ConfigDict(extra="forbid")

    feature_code: str
    result: float
    unit: str = Field(min_length=1, max_length=30)
    reference_lower: float
    reference_upper: float

    @field_validator("feature_code")
    @classmethod
    def feature_must_be_supported(cls, value: str) -> str:
        if value not in FEATURE_METADATA:
            raise ValueError("feature_code is not supported by LabInsight")
        return value

    @field_validator("result", "reference_lower", "reference_upper")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("values must be finite numbers")
        return value

    @model_validator(mode="after")
    def lower_must_be_less_than_upper(self) -> "ReportItem":
        if self.reference_lower >= self.reference_upper:
            raise ValueError("reference_lower must be less than reference_upper")
        return self


class ReportEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReportItem] = Field(min_length=1, max_length=18)


class ReportItemEvaluation(BaseModel):
    feature_code: str
    test_name: str
    result: float
    unit: str
    reference_lower: float
    reference_upper: float
    position: Literal["below", "within", "above"]
    statement: str


class ReportEvaluationResponse(BaseModel):
    items: list[ReportItemEvaluation]
    limitation: str

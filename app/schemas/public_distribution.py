"""Contracts for non-clinical comparison with an NHANES descriptive band."""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.public_distribution import NHANES_PUBLIC_BANDS, PUBLIC_BAND_LABEL
from ml.preprocessing.biopro import FEATURE_METADATA


class PublicDistributionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_code: str
    result: float
    unit: str = Field(min_length=1, max_length=30)

    @field_validator("feature_code")
    @classmethod
    def feature_must_have_a_public_band(cls, value: str) -> str:
        if value not in NHANES_PUBLIC_BANDS:
            raise ValueError("feature_code is not supported by the public comparison band")
        return value

    @field_validator("result")
    @classmethod
    def result_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("result must be a finite number")
        return value

    @model_validator(mode="after")
    def unit_must_match_the_public_band(self) -> "PublicDistributionItem":
        expected_unit = FEATURE_METADATA[self.feature_code]["unit"]
        if self.unit != expected_unit:
            raise ValueError(f"unit must be {expected_unit} for this public comparison")
        return self


class PublicDistributionComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PublicDistributionItem] = Field(min_length=1, max_length=18)


class PublicDistributionComparisonItem(BaseModel):
    feature_code: str
    test_name: str
    result: float
    unit: str
    band_lower: float
    band_upper: float
    position: Literal["below", "within", "above"]
    statement: str


class PublicDistributionComparisonResponse(BaseModel):
    band_label: str = PUBLIC_BAND_LABEL
    items: list[PublicDistributionComparisonItem]
    limitation: str

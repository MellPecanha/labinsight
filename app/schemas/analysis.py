"""API contracts for statistical laboratory-panel analysis."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class LaboratoryPanel(BaseModel):
    """Selected BIOPRO_L measurements in their original NHANES units.

    A partial panel is accepted because the model was trained with median
    imputation.  The response always reports completeness so clients can make
    that limitation visible.
    """

    model_config = ConfigDict(extra="forbid")

    LBXSATSI: float | None = None
    LBDSALSI: float | None = None
    LBXSAPSI: float | None = None
    LBXSASSI: float | None = None
    LBXSC3SI: float | None = None
    LBDSBUSI: float | None = None
    LBXSCLSI: float | None = None
    LBDSCRSI: float | None = None
    LBXSGTSI: float | None = None
    LBDSIRSI: float | None = None
    LBXMAGN: float | None = None
    LBDSPHSI: float | None = None
    LBXSKSI: float | None = None
    LBXSNASI: float | None = None
    LBDSTBSI: float | None = None
    LBDSCASI: float | None = None
    LBDSTPSI: float | None = None
    LBDSUASI: float | None = None

    @field_validator("*")
    @classmethod
    def values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("laboratory values must be finite numbers")
        return value

    @model_validator(mode="after")
    def at_least_one_value_is_required(self) -> "LaboratoryPanel":
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("at least one laboratory value is required")
        return self


class AnalysisResponse(BaseModel):
    """Non-clinical response returned by the anomaly-detection endpoint."""

    model_version: str
    anomaly_score: float
    is_statistically_unusual: bool
    observed_feature_count: int
    missing_feature_count: int
    interpretation: str
    limitation: str

"""Load the local ML artifact and expose a safe analysis operation."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.core.config import MODEL_PATH, MODEL_VERSION
from app.schemas.analysis import AnalysisResponse, LaboratoryPanel
from ml.training.isolation_forest import IsolationForestExperiment


class ModelUnavailableError(RuntimeError):
    """Raised when the locally trained artifact cannot be used."""


class AnomalyService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise ModelUnavailableError(f"Model artifact was not found at {model_path}")
        try:
            experiment = joblib.load(model_path)
        except Exception as error:  # pragma: no cover - defensive boundary around external artifact
            raise ModelUnavailableError("Model artifact could not be loaded") from error
        if not isinstance(experiment, IsolationForestExperiment):
            raise ModelUnavailableError("Model artifact has an unsupported type")
        self.experiment = experiment

    def analyze(self, panel: LaboratoryPanel) -> AnalysisResponse:
        data = pd.DataFrame([panel.model_dump()], columns=self.experiment.feature_names)
        scored = self.experiment.score(data).iloc[0]
        return AnalysisResponse(
            model_version=MODEL_VERSION,
            anomaly_score=float(scored["anomaly_score"]),
            is_statistically_unusual=bool(scored["is_statistically_unusual"]),
            observed_feature_count=int(scored["observed_feature_count"]),
            missing_feature_count=int(scored["missing_feature_count"]),
            interpretation=(
                "The panel's pattern was statistically compared with the experimental "
                "NHANES BIOPRO_L reference dataset."
            ),
            limitation=(
                "This is an experimental statistical signal, not a diagnosis, clinical "
                "assessment, or treatment recommendation. Missing values are imputed."
            ),
        )


@lru_cache
def get_anomaly_service() -> AnomalyService:
    """Reuse the loaded artifact across requests."""
    return AnomalyService()

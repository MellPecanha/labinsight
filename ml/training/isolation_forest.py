"""Train and score a reproducible Isolation Forest experiment.

An Isolation Forest isolates observations using random feature splits.  Points
that need fewer splits to be isolated receive larger anomaly scores.  Its score
is statistical, not a medical finding and not a diagnosis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class IsolationForestExperiment:
    """Fitted pipeline and the threshold derived only from training scores."""

    pipeline: Pipeline
    feature_names: list[str]
    threshold: float
    anomaly_fraction: float

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return statistical scores plus missingness metadata for review."""
        _validate_columns(data, self.feature_names)
        ordered = data.loc[:, self.feature_names]
        # sklearn's score_samples is higher for more typical points.  Reverse
        # it so the output reads naturally: higher = more statistically unusual.
        anomaly_score = -self.pipeline.score_samples(ordered)
        return pd.DataFrame(
            {
                "anomaly_score": anomaly_score,
                "is_statistically_unusual": anomaly_score >= self.threshold,
                "observed_feature_count": ordered.notna().sum(axis=1),
                "missing_feature_count": ordered.isna().sum(axis=1),
            },
            index=ordered.index,
        )


def fit_isolation_forest(
    train_data: pd.DataFrame,
    *,
    anomaly_fraction: float = 0.02,
    random_state: int = 42,
    n_estimators: int = 300,
) -> IsolationForestExperiment:
    """Fit preprocessing and detector on train data only.

    ``anomaly_fraction`` is an operational review capacity (for example, 2%),
    not an estimate of disease prevalence.  The threshold is computed from the
    training-score quantile and then held fixed when scoring the holdout set.
    """
    if not 0 < anomaly_fraction < 1:
        raise ValueError("anomaly_fraction must be between 0 and 1")
    if train_data.empty:
        raise ValueError("train_data must not be empty")
    if train_data.isna().all(axis=1).any():
        raise ValueError("train_data must not contain rows with all features missing")

    feature_names = list(train_data.columns)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_data.loc[:, feature_names])
    train_scores = -pipeline.score_samples(train_data.loc[:, feature_names])
    threshold = float(np.quantile(train_scores, 1 - anomaly_fraction))
    return IsolationForestExperiment(
        pipeline=pipeline,
        feature_names=feature_names,
        threshold=threshold,
        anomaly_fraction=anomaly_fraction,
    )


def summarize_holdout(experiment: IsolationForestExperiment, holdout_data: pd.DataFrame) -> dict[str, float | int]:
    """Describe detector stability on unlabeled data; this is not accuracy."""
    scored = experiment.score(holdout_data)
    return {
        "records": len(scored),
        "flagged_records": int(scored["is_statistically_unusual"].sum()),
        "flagged_fraction": float(scored["is_statistically_unusual"].mean()),
        "score_median": float(scored["anomaly_score"].median()),
        "score_p95": float(scored["anomaly_score"].quantile(0.95)),
    }


def _validate_columns(data: pd.DataFrame, feature_names: list[str]) -> None:
    missing_columns = set(feature_names).difference(data.columns)
    if missing_columns:
        raise ValueError(f"data is missing model features: {sorted(missing_columns)}")

import numpy as np
import pandas as pd
import pytest

from ml.training.isolation_forest import fit_isolation_forest, summarize_holdout


def test_scores_preserve_index_and_expose_missingness() -> None:
    rng = np.random.default_rng(42)
    train = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    train.loc[0, "a"] = np.nan
    holdout = pd.DataFrame(rng.normal(size=(20, 3)), columns=train.columns, index=range(500, 520))

    experiment = fit_isolation_forest(train, anomaly_fraction=0.05, n_estimators=20)
    scored = experiment.score(holdout)

    assert scored.index.equals(holdout.index)
    assert set(scored.columns) == {
        "anomaly_score",
        "is_statistically_unusual",
        "observed_feature_count",
        "missing_feature_count",
    }
    assert scored["anomaly_score"].notna().all()
    assert summarize_holdout(experiment, holdout)["records"] == 20


def test_rejects_rows_without_any_observed_feature() -> None:
    data = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})

    with pytest.raises(ValueError, match="all features missing"):
        fit_isolation_forest(data)

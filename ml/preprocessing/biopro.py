"""Preparation utilities for the public NHANES BIOPRO_L dataset.

This module deliberately retains partially observed laboratory panels.  Missing
values are imputed later inside the modelling pipeline, after the train/test
split, to avoid leaking information from the holdout set.
"""

from pathlib import Path

import pandas as pd


SAS_ZERO_ARTIFACT = 5.397605346934028e-79

SELECTED_FEATURES = [
    "LBXSATSI",
    "LBDSALSI",
    "LBXSAPSI",
    "LBXSASSI",
    "LBXSC3SI",
    "LBDSBUSI",
    "LBXSCLSI",
    "LBDSCRSI",
    "LBXSGTSI",
    "LBDSIRSI",
    "LBXMAGN",
    "LBDSPHSI",
    "LBXSKSI",
    "LBXSNASI",
    "LBDSTBSI",
    "LBDSCASI",
    "LBDSTPSI",
    "LBDSUASI",
]


def load_biopro(path: str | Path) -> pd.DataFrame:
    """Read BIOPRO_L and replace the known SAS XPORT zero-reading artifact."""
    dataset = pd.read_sas(path, format="xport")
    return dataset.replace(SAS_ZERO_ARTIFACT, 0.0)


def select_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the selected panel, excluding only rows with no observed result.

    No outlier is removed here.  A row with one or more results remains in the
    experiment and is later accompanied by its missing-value count in reports.
    """
    missing_columns = set(SELECTED_FEATURES).difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"BIOPRO_L is missing expected columns: {sorted(missing_columns)}")

    panel = dataset.loc[:, SELECTED_FEATURES].copy()
    return panel.dropna(how="all")

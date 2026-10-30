"""Preparation utilities for the public NHANES BIOPRO_L dataset.

This module deliberately retains partially observed laboratory panels.  Missing
values are imputed later inside the modelling pipeline, after the train/test
split, to avoid leaking information from the holdout set.
"""

from pathlib import Path

import pandas as pd


SAS_ZERO_ARTIFACT = 5.397605346934028e-79

FEATURE_METADATA = {
    "LBXSATSI": {"name": "Alanina aminotransferase (ALT)", "unit": "U/L", "group": "Enzimas e proteínas"},
    "LBDSALSI": {"name": "Albumina", "unit": "g/L", "group": "Enzimas e proteínas"},
    "LBXSAPSI": {"name": "Fosfatase alcalina", "unit": "U/L", "group": "Enzimas e proteínas"},
    "LBXSASSI": {"name": "Aspartato aminotransferase (AST)", "unit": "U/L", "group": "Enzimas e proteínas"},
    "LBXSGTSI": {"name": "Gama-glutamil transferase (GGT)", "unit": "U/L", "group": "Enzimas e proteínas"},
    "LBDSTPSI": {"name": "Proteína total", "unit": "g/L", "group": "Enzimas e proteínas"},
    "LBXSC3SI": {"name": "Bicarbonato", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBXSCLSI": {"name": "Cloreto", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBXMAGN": {"name": "Magnésio", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBDSPHSI": {"name": "Fósforo", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBXSKSI": {"name": "Potássio", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBXSNASI": {"name": "Sódio", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBDSCASI": {"name": "Cálcio", "unit": "mmol/L", "group": "Eletrólitos e minerais"},
    "LBDSBUSI": {"name": "Nitrogênio ureico no sangue", "unit": "mmol/L", "group": "Função renal e metabólica"},
    "LBDSCRSI": {"name": "Creatinina", "unit": "µmol/L", "group": "Função renal e metabólica"},
    "LBDSIRSI": {"name": "Ferro", "unit": "µmol/L", "group": "Função renal e metabólica"},
    "LBDSTBSI": {"name": "Bilirrubina total", "unit": "µmol/L", "group": "Função renal e metabólica"},
    "LBDSUASI": {"name": "Ácido úrico", "unit": "µmol/L", "group": "Função renal e metabólica"},
}

SELECTED_FEATURES = [
    "LBXSATSI", "LBDSALSI", "LBXSAPSI", "LBXSASSI", "LBXSC3SI", "LBDSBUSI",
    "LBXSCLSI", "LBDSCRSI", "LBXSGTSI", "LBDSIRSI", "LBXMAGN", "LBDSPHSI",
    "LBXSKSI", "LBXSNASI", "LBDSTBSI", "LBDSCASI", "LBDSTPSI", "LBDSUASI",
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

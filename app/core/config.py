"""Central, environment-aware application settings."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("LABINSIGHT_MODEL_PATH", PROJECT_ROOT / "data/processed/isolation_forest.joblib")
)
MODEL_VERSION = "isolation-forest-biopro-l-2021-2023-v1"

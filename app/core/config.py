"""Central, environment-aware application settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
MODEL_PATH = Path(
    os.getenv("LABINSIGHT_MODEL_PATH", PROJECT_ROOT / "data/processed/isolation_forest.joblib")
)
MODEL_VERSION = "isolation-forest-biopro-l-2021-2023-v1"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "rag/knowledge_base"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

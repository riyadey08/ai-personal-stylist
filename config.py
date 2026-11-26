"""Central configuration for the AI Personal Stylist app.

All paths are relative to this package so the app works from any working
directory. Secrets and the model name come from environment variables
(see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

KB_PATH = DATA_DIR / "fashion_knowledge_base.csv"
TRAINING_DATA_PATH = DATA_DIR / "body_features_with_metadata.csv"

INDEX_PATH = MODELS_DIR / "faiss_index.bin"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PCA_PATH = MODELS_DIR / "pca.joblib"
METADATA_PATH = MODELS_DIR / "retrieval_metadata.csv"
FEATURES_PATH = MODELS_DIR / "feature_order.joblib"

# Gemini. The app requires a key; see .env.example.
API_KEY_ENV = "GOOGLE_API_KEY"
MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-3.6-flash"

OCCASIONS = ["Any", "Casual", "College", "Office", "Party", "Travel"]
SEASONS = ["All", "Summer", "Winter", "Autumn", "Spring"]

BODY_SHAPES = ["Pear", "Rectangle", "Inverted Triangle"]

# Multi-photo fusion: fitted-clothes photos are trusted more for body width.
FITTED_PHOTO_WEIGHT = 2.0
MIN_QUALITY_SCORE = 9

# Retrieval defaults.
DEFAULT_K = 5
MAX_CONTEXT_ROWS = 20
MAX_CONTEXT_CHARS = 6000


def model_name() -> str:
    return os.getenv(MODEL_ENV, DEFAULT_MODEL)


def ensure_dirs() -> None:
    for directory in (MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

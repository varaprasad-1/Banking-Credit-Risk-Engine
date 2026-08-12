"""
prediction.py -- Load saved model artifacts and generate predictions.

Provides reusable functions used by both the Streamlit app and external scripts.
Never retrains -- always loads pre-saved artifacts.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    DEFAULT_MODEL_PATH,
    PREPROCESSING_PATH,
    MODEL_METADATA_PATH,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading (cached in module-level singletons)
# ─────────────────────────────────────────────────────────────────────────────
_model        = None
_preprocessor = None
_metadata     = None


def load_model(path: str = DEFAULT_MODEL_PATH):
    global _model
    if _model is None:
        with open(path, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
    return _model


def load_preprocessor(path: str = PREPROCESSING_PATH):
    global _preprocessor
    if _preprocessor is None:
        with open(path, "rb") as f:
            _preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
    return _preprocessor


def load_metadata(path: str = MODEL_METADATA_PATH) -> dict:
    global _metadata
    if _metadata is None:
        with open(path, "rb") as f:
            _metadata = pickle.load(f)
        logger.info(f"Metadata loaded from {path}")
    return _metadata


def artifacts_exist() -> bool:
    return (
        os.path.isfile(DEFAULT_MODEL_PATH)
        and os.path.isfile(PREPROCESSING_PATH)
        and os.path.isfile(MODEL_METADATA_PATH)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(input_dict: dict, metadata: dict) -> tuple[bool, str]:
    """
    Check that all required features are present and have valid types.
    Returns (is_valid, error_message).
    """
    num_feats = metadata.get("num_features", [])
    cat_feats = metadata.get("cat_features", [])
    all_feats = num_feats + cat_feats

    missing = [f for f in all_feats if f not in input_dict]
    if missing:
        return False, f"Missing required fields: {missing}"

    for f in num_feats:
        v = input_dict.get(f)
        try:
            float(v)
        except (TypeError, ValueError):
            return False, f"Field '{f}' must be numeric, got: {v!r}"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

def classify_risk(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "LOW"
    elif prob <= HIGH_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def predict(input_dict: dict) -> dict:
    """
    Predict default probability for a single customer profile.

    Parameters
    ----------
    input_dict : dict
        Feature values keyed by column name.

    Returns
    -------
    dict with keys:
        default_probability (float 0-1)
        risk_level          ("LOW" | "MEDIUM" | "HIGH")
        risk_score          (float 0-100)
        risk_color          (str hex)
    """
    model        = load_model()
    preprocessor = load_preprocessor()
    metadata     = load_metadata()

    # Validate
    is_valid, err = validate_input(input_dict, metadata)
    if not is_valid:
        raise ValueError(err)

    df_row = pd.DataFrame([input_dict])

    # Only keep columns the preprocessor was trained on
    known_cols = metadata["num_features"] + metadata["cat_features"]
    for col in known_cols:
        if col not in df_row.columns:
            df_row[col] = np.nan

    df_row = df_row[known_cols]

    X = preprocessor.transform(df_row)
    prob = float(model.predict_proba(X)[0, 1])

    risk_level = classify_risk(prob)
    color_map  = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}

    return {
        "default_probability": round(prob, 4),
        "risk_level":          risk_level,
        "risk_score":          round(prob * 100, 1),
        "risk_color":          color_map[risk_level],
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run prediction on an entire DataFrame.
    Returns original df with added columns: default_prob, risk_level, risk_score.
    """
    model        = load_model()
    preprocessor = load_preprocessor()
    metadata     = load_metadata()

    known_cols = metadata["num_features"] + metadata["cat_features"]
    available  = [c for c in known_cols if c in df.columns]
    df_feat    = df[available].copy()

    # Fill missing required columns with NaN so the pipeline can impute
    for col in known_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
    df_feat = df_feat[known_cols]

    X     = preprocessor.transform(df_feat)
    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["default_prob"] = np.round(probs, 4)
    out["risk_level"]   = [classify_risk(p) for p in probs]
    out["risk_score"]   = np.round(probs * 100, 1)
    return out


if __name__ == "__main__":
    if not artifacts_exist():
        print("Models not trained yet. Run: python src/train_model.py")
        sys.exit(1)

    meta = load_metadata()
    sample = {}
    for f in meta["num_features"]:
        sample[f] = 0.0
    for f in meta["cat_features"]:
        sample[f] = "missing"

    result = predict(sample)
    print("Sample prediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

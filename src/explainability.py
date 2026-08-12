"""
explainability.py -- Feature-importance-based explanations for credit risk predictions.

SHAP is used when available and compatible.
Falls back to a coefficient/importance-based approach if SHAP is unavailable.

IMPORTANT: Results are described as "contributing factors", NOT causal claims.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.prediction import load_model, load_preprocessor, load_metadata

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _try_shap_explanation(model, X_row: np.ndarray, feature_names: list[str]) -> list[tuple]:
    """
    Try to get SHAP values for a single row.
    Returns list of (feature_name, shap_value) sorted by |shap_value| desc.
    """
    try:
        import shap
        # TreeExplainer works for RF, XGBoost, GBM
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_row)
            # For binary classification shap_values returns list [class0, class1]
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            vals = shap_vals[0]
        except Exception:
            # LinearExplainer for logistic regression
            explainer = shap.LinearExplainer(model, X_row)
            shap_vals = explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            vals = shap_vals[0]

        pairs = list(zip(feature_names, vals.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs
    except Exception as e:
        logger.debug(f"SHAP unavailable: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: global feature importance
# ─────────────────────────────────────────────────────────────────────────────

def _importance_explanation(model, X_row: np.ndarray, feature_names: list[str]) -> list[tuple]:
    """
    Uses global feature importances + input-value sign to approximate per-instance explanation.
    Not as accurate as SHAP, but reliable across all sklearn estimators.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        # Uniform fallback
        importances = np.ones(len(feature_names))

    # Weight by feature value deviation from zero (post-scaling)
    contributions = importances * X_row[0]
    pairs = list(zip(feature_names, contributions.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _prettify(feature_name: str) -> str:
    """Convert encoded feature names (e.g. 'num__loan_int_rate') to readable labels."""
    name = feature_name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ").replace("  ", " ").title()
    return name


def explain_prediction(input_dict: dict, top_n: int = 5) -> dict:
    """
    Generate risk factor explanation for a single customer.

    Returns
    -------
    dict with:
        risk_factors    : list of dicts  (push risk UP)
        positive_factors: list of dicts  (push risk DOWN)
        method          : "SHAP" | "feature_importance"
    """
    model         = load_model()
    preprocessor  = load_preprocessor()
    metadata      = load_metadata()
    feature_names = metadata["feature_names"]

    known_cols = metadata["num_features"] + metadata["cat_features"]
    df_row = pd.DataFrame([input_dict])
    for col in known_cols:
        if col not in df_row.columns:
            df_row[col] = np.nan
    df_row = df_row[known_cols]

    X_row = preprocessor.transform(df_row)

    # Try SHAP first, fall back to importance
    pairs = _try_shap_explanation(model, X_row, feature_names)
    method = "SHAP"
    if not pairs:
        pairs = _importance_explanation(model, X_row, feature_names)
        method = "feature_importance"

    risk_factors     = []
    positive_factors = []

    for feat_name, val in pairs:
        label = _prettify(feat_name)
        if val > 0:
            risk_factors.append({
                "feature":     label,
                "raw_feature": feat_name,
                "contribution": round(float(val), 4),
                "description": f"Contributing factor toward higher predicted risk.",
            })
        else:
            positive_factors.append({
                "feature":     label,
                "raw_feature": feat_name,
                "contribution": round(float(val), 4),
                "description": f"Contributing factor toward lower predicted risk.",
            })

    return {
        "risk_factors":     risk_factors[:top_n],
        "positive_factors": positive_factors[:top_n],
        "method":           method,
    }

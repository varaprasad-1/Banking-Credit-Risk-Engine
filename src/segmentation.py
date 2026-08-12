"""
segmentation.py -- K-Means customer segmentation.

Steps:
  1. Select financial/demographic numerical features from the dataset
  2. Handle missing values + scale
  3. Evaluate k using Elbow method + Silhouette score
  4. Fit K-Means with the best k
  5. Save segmentation model
  6. Generate and return segment profiles

Segment labels are assigned AFTER examining actual cluster characteristics.
Labels are NOT pre-assigned blindly.
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

from src.config import (
    SEGMENTATION_MODEL_PATH,
    N_CLUSTERS_RANGE,
    N_CLUSTERS,
    AUTO_SELECT_CLUSTERS,
    RANDOM_STATE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection for segmentation
# ─────────────────────────────────────────────────────────────────────────────

# Prefer financial/demographic features for segmentation
PREFERRED_SEG_FEATURES = [
    "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "person_age", "person_emp_length", "cb_person_cred_hist_length",
    "annual_inc", "annual_income", "income", "loan_amount",
    "credit_history_length", "age",
]


def select_segmentation_features(df: pd.DataFrame, target_col: str, id_cols: list[str]) -> list[str]:
    """Select numerical features suitable for segmentation."""
    exclude = set(id_cols + [target_col] if target_col else id_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [c for c in num_cols if c not in exclude]

    # Prefer known financial/demographic features
    preferred = [c for c in PREFERRED_SEG_FEATURES if c in available]
    remaining = [c for c in available if c not in preferred]

    seg_features = preferred + remaining
    # Take up to 8 features (too many hurt cluster quality)
    seg_features = seg_features[:8]

    if not seg_features:
        seg_features = available[:5]

    logger.info(f"Segmentation features selected: {seg_features}")
    return seg_features


# ─────────────────────────────────────────────────────────────────────────────
# Optimal k selection
# ─────────────────────────────────────────────────────────────────────────────

def select_optimal_k(X_scaled: np.ndarray) -> tuple[int, dict]:
    """
    Test k in N_CLUSTERS_RANGE.
    Returns (best_k, evaluation_data_dict).
    """
    k_min, k_max = N_CLUSTERS_RANGE
    inertias     = []
    sil_scores   = []
    k_range      = list(range(k_min, k_max + 1))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            sil = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
            sil_scores.append(sil)
        else:
            sil_scores.append(0.0)

    # Pick k with highest silhouette score
    best_idx = int(np.argmax(sil_scores))
    best_k   = k_range[best_idx]

    eval_data = {
        "k_range":    k_range,
        "inertias":   inertias,
        "sil_scores": sil_scores,
        "best_k":     best_k,
    }
    logger.info(
        f"Optimal k selected: {best_k}  "
        f"(Silhouette={sil_scores[best_idx]:.4f})"
    )
    return best_k, eval_data


# ─────────────────────────────────────────────────────────────────────────────
# Segment label assignment (data-driven)
# ─────────────────────────────────────────────────────────────────────────────

def assign_segment_labels(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign meaningful labels to clusters based on actual cluster statistics.
    Labels are determined by cluster characteristics, not pre-assigned.
    """
    df = profiles_df.copy()
    n_clusters = len(df)

    # Determine which characterisation columns are available
    has_income  = "person_income_mean" in df.columns or any("income" in c for c in df.columns)
    has_default = "default_rate" in df.columns
    has_loan    = any("loan_amnt" in c for c in df.columns)

    # Build a simple scoring for each cluster
    # Higher income → more stable; higher default rate → higher risk
    labels = []
    for _, row in df.iterrows():
        # Extract available stats
        income_col  = next((c for c in df.columns if "income" in c and "mean" in c), None)
        default_col = "default_rate" if "default_rate" in df.columns else None
        loan_col    = next((c for c in df.columns if "loan_amnt" in c and "mean" in c), None)

        income  = row[income_col]  if income_col  else None
        default = row[default_col] if default_col else None
        loan    = row[loan_col]    if loan_col    else None

        # Relative ranks
        if income_col:
            income_rank  = df[income_col].rank(pct=True)[row.name]
        else:
            income_rank  = 0.5

        if default_col:
            default_rank = df[default_col].rank(pct=True)[row.name]
        else:
            default_rank = 0.5

        # Label logic
        if default_rank >= 0.75:
            label = "High-Risk Customer"
        elif income_rank >= 0.75 and default_rank < 0.50:
            label = "Premium Customer"
        elif income_rank < 0.35 and default_rank >= 0.50:
            label = "Emerging Customer"
        else:
            label = "Stable Customer"

        labels.append(label)

    df["segment_label"] = labels
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_segmentation(df: pd.DataFrame, target_col: str, id_cols: list[str]) -> dict:
    """
    Fit K-Means segmentation on the dataset.
    Returns a dict with model, scaler, features, profiles, and evaluation data.
    """
    seg_features = select_segmentation_features(df, target_col, id_cols)

    df_seg = df[seg_features].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(df_seg)

    # Scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Optimal k
    if AUTO_SELECT_CLUSTERS:
        best_k, eval_data = select_optimal_k(X_scaled)
    else:
        best_k    = N_CLUSTERS
        eval_data = {"best_k": best_k}

    # Fit final model
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["segment"] = labels

    # Build cluster profiles
    agg_dict = {f: ["mean", "median", "std"] for f in seg_features}
    if target_col in df_out.columns:
        agg_dict[target_col] = "mean"

    profiles = df_out.groupby("segment").agg(agg_dict).round(2)
    profiles.columns = ["_".join(c).strip() for c in profiles.columns.values]
    profiles["cluster_size"] = df_out.groupby("segment").size()
    if target_col in df_out.columns:
        profiles.rename(columns={f"{target_col}_mean": "default_rate"}, inplace=True)
    profiles = profiles.reset_index()

    profiles = assign_segment_labels(profiles)

    seg_artifact = {
        "model":        km,
        "scaler":       scaler,
        "imputer":      imputer,
        "seg_features": seg_features,
        "n_clusters":   best_k,
        "profiles":     profiles,
        "eval_data":    eval_data,
        "target_col":   target_col,
        "labels_series": labels.tolist(),
    }

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SEGMENTATION_MODEL_PATH, "wb") as f:
        pickle.dump(seg_artifact, f)
    logger.info(f"Segmentation model saved -> {SEGMENTATION_MODEL_PATH}")

    print(f"\n  Segmentation: {best_k} clusters identified")
    print("  Cluster sizes:")
    for _, row in profiles.iterrows():
        label = row.get("segment_label", f"Segment {int(row['segment'])}")
        print(f"    • {label}: {int(row['cluster_size'])} customers")

    return seg_artifact


# ─────────────────────────────────────────────────────────────────────────────
# Predict segment for a new customer
# ─────────────────────────────────────────────────────────────────────────────

def predict_segment(input_dict: dict) -> dict:
    """Assign a new customer to a segment."""
    with open(SEGMENTATION_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    seg_features = artifact["seg_features"]
    imputer      = artifact["imputer"]
    scaler       = artifact["scaler"]
    model        = artifact["model"]
    profiles     = artifact["profiles"]

    row = {f: input_dict.get(f, np.nan) for f in seg_features}
    X   = pd.DataFrame([row])[seg_features].values
    X   = imputer.transform(X)
    X   = scaler.transform(X)

    cluster_id = int(model.predict(X)[0])

    # Find label from profiles
    match = profiles[profiles["segment"] == cluster_id]
    if not match.empty:
        label = match.iloc[0].get("segment_label", f"Segment {cluster_id}")
    else:
        label = f"Segment {cluster_id}"

    return {
        "segment_id":    cluster_id,
        "segment_label": label,
    }

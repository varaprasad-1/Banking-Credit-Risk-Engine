"""
preprocessing.py -- Scikit-learn preprocessing pipeline.

Handles:
  • Missing values (median imputation for numerics, constant for categoricals)
  • Standard scaling of numerical features
  • One-hot encoding of categorical features (handle_unknown='ignore')
  • Duplicate row removal
  • ID column removal
  • Data-leakage prevention (pipeline fit only on training set)

The fitted pipeline is saved to disk and MUST be reused during prediction.
"""

import logging
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import (
    PREPROCESSING_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import (
    validate_dataset,
    detect_numerical_columns,
    detect_categorical_columns,
    detect_id_columns,
    detect_target_column,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows and obvious data-quality issues.
    Returns a cleaned copy; does NOT modify in place.
    """
    original_len = len(df)
    df = df.copy()

    # Drop exact duplicates
    df = df.drop_duplicates()
    n_dropped = original_len - len(df)
    if n_dropped:
        logger.info(f"Removed {n_dropped} duplicate rows.")

    # Drop columns where > 60 % of values are missing
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > 0.60].index.tolist()
    if high_missing:
        df = df.drop(columns=high_missing)
        logger.warning(f"Dropped high-missing columns (>60%): {high_missing}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline construction
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    num_features: list[str],
    cat_features: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numeric features
      - Imputes + one-hot-encodes categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_features:
        transformers.append(("num", numeric_pipeline, num_features))
    if cat_features:
        transformers.append(("cat", categorical_pipeline, cat_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # drop any columns not explicitly listed
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer, cat_features: list[str]) -> list[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            # Get OHE feature names
            ohe: OneHotEncoder = transformer.named_steps["encoder"]
            ohe_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
    return feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Prepare training data
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> dict:
    """
    Full preprocessing for training:
      1. Clean dataframe
      2. Detect columns
      3. Build & fit preprocessing pipeline (on train set only)
      4. Return splits + pipeline + metadata

    Returns dict with keys:
      X_train, X_test, y_train, y_test,
      feature_names, num_features, cat_features,
      target_col, preprocessor
    """
    df = clean_dataframe(df)
    report = validate_dataset(df)

    target_col = report["target_column"]
    if target_col is None:
        raise ValueError(
            "Cannot identify the target column. "
            "Set TARGET_COLUMN in src/config.py."
        )

    id_cols  = report["id_columns"]
    num_cols = report["numerical_columns"]
    cat_cols = report["categorical_columns"]

    logger.info(f"Target      : {target_col}")
    logger.info(f"Numerical   : {num_cols}")
    logger.info(f"Categorical : {cat_cols}")
    logger.info(f"ID columns  : {id_cols}")

    # Features
    X = df.drop(columns=[target_col] + id_cols, errors="ignore")
    y = df[target_col].astype(int)

    # Remove any remaining ID-like columns from X
    X = X.drop(columns=id_cols, errors="ignore")

    # Recalculate which of X's columns are numeric / categorical
    # (after dropping target and IDs)
    num_features = [c for c in num_cols if c in X.columns]
    cat_features = [c for c in cat_cols if c in X.columns]

    # Guard against empty features
    if not num_features and not cat_features:
        raise ValueError("No usable features found after removing target and IDs.")

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError:
        # Fallback if stratify fails (e.g., too few samples per class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    logger.info(
        f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  "
        f"Default rate (train): {y_train.mean():.1%}"
    )

    # Build & fit pipeline on TRAIN data only
    preprocessor = build_preprocessing_pipeline(num_features, cat_features)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = get_feature_names_out(preprocessor, cat_features)

    return {
        "X_train": X_train_t,
        "X_test":  X_test_t,
        "y_train": y_train.values,
        "y_test":  y_test.values,
        "feature_names": feature_names,
        "num_features":  num_features,
        "cat_features":  cat_features,
        "target_col":    target_col,
        "preprocessor":  preprocessor,
        "X_train_raw":   X_train,
        "X_test_raw":    X_test,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────

def save_preprocessing_pipeline(preprocessor: ColumnTransformer, path: str = PREPROCESSING_PATH):
    import os, pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved preprocessing pipeline -> {path}")


def load_preprocessing_pipeline(path: str = PREPROCESSING_PATH) -> ColumnTransformer:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Transform a single-row dict (for inference)
# ──────────────────────────────────────────────────────────────────────────────

def transform_input(
    input_dict: dict,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """
    Transform a single customer profile dict into a preprocessed numpy array.
    Handles missing/extra keys gracefully.
    """
    df_row = pd.DataFrame([input_dict])
    try:
        return preprocessor.transform(df_row)
    except Exception as e:
        raise ValueError(f"Input transformation failed: {e}") from e

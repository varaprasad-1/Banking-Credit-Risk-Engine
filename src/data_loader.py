"""
data_loader.py -- Dataset discovery, loading, validation, and schema introspection.

Supports any CSV with a binary/numeric default/credit-risk target column.
Does NOT assume any particular column names -- everything is detected or configurable.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DATA_DIR,
    DATASET_CANDIDATES,
    TARGET_COLUMN,
    TARGET_COLUMN_CANDIDATES,
    ID_COLUMN_PATTERNS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_dataset() -> str | None:
    """
    Search the data/ directory for a credit-risk CSV.
    Returns the absolute path to the first matching file, or None.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Check configured candidates
    for filename in DATASET_CANDIDATES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(path):
            logger.info(f"Dataset found: {path}")
            return path

    # 2. Scan for any CSV
    csvs = sorted(Path(DATA_DIR).glob("*.csv"))
    if csvs:
        path = str(csvs[0])
        logger.info(f"Using first CSV found in data/: {path}")
        return path

    logger.warning(
        "No dataset found in data/. "
        "Please place credit_risk.csv in the data/ directory."
    )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load CSV into a DataFrame. Raises FileNotFoundError if not found."""
    if path is None:
        path = find_dataset()
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            "Dataset not found. Place credit_risk.csv in the data/ directory."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Column Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Heuristically identify columns that are likely identifiers."""
    id_cols = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in ID_COLUMN_PATTERNS:
            if pattern in col_lower:
                id_cols.append(col)
                break
        else:
            # High-cardinality integer columns with monotonic-ish values
            if (
                df[col].dtype in [np.int64, np.int32, np.float64]
                and df[col].nunique() == len(df)
            ):
                id_cols.append(col)
    return list(dict.fromkeys(id_cols))  # deduplicate, preserve order


def detect_target_column(df: pd.DataFrame) -> str | None:
    """
    Identify the target (default) column.
    Priority: config override -> known name candidates -> binary numeric column heuristic.
    """
    # 1. Config override
    if TARGET_COLUMN and TARGET_COLUMN in df.columns:
        logger.info(f"Using configured target column: '{TARGET_COLUMN}'")
        return TARGET_COLUMN

    # 2. Known name candidates
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate.lower() in df_cols_lower:
            col = df_cols_lower[candidate.lower()]
            logger.info(f"Auto-detected target column: '{col}'")
            return col

    # 3. Binary numeric heuristic (0/1 with <60% ones)
    for col in df.columns:
        if df[col].dtype in [np.int64, np.int32, np.float64, int, float]:
            vals = df[col].dropna().unique()
            if set(vals).issubset({0, 1}):
                ratio = df[col].mean()
                if 0.01 < ratio < 0.90:
                    logger.info(
                        f"Heuristically identified target column: '{col}' "
                        f"(default rate ≈ {ratio:.1%})"
                    )
                    return col

    logger.error(
        "Could not detect a target column. Set TARGET_COLUMN in src/config.py."
    )
    return None


def detect_numerical_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Return numerical feature columns (excluding target, IDs, etc.)."""
    exclude = set(exclude or [])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]


def detect_categorical_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Return categorical/object feature columns."""
    exclude = set(exclude or [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also detect low-cardinality integers as potential categoricals
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude:
            continue
        if df[col].nunique() <= 10 and df[col].nunique() >= 2:
            unique_vals = sorted(df[col].dropna().unique())
            if all(isinstance(v, (int, np.integer)) and v >= 0 for v in unique_vals):
                # Could be ordinal/categorical — keep as numeric, note it
                pass
    return [c for c in cat_cols if c not in exclude]


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run data quality checks. Returns a validation report dict.
    Does NOT raise on quality issues -- caller decides what to do.
    """
    report = {}

    report["shape"]       = df.shape
    report["n_rows"]      = df.shape[0]
    report["n_cols"]      = df.shape[1]
    report["columns"]     = list(df.columns)
    report["dtypes"]      = df.dtypes.astype(str).to_dict()

    # Missing values
    missing      = df.isnull().sum()
    missing_pct  = (missing / len(df) * 100).round(2)
    report["missing_values"]  = missing[missing > 0].to_dict()
    report["missing_pct"]     = missing_pct[missing_pct > 0].to_dict()
    report["has_missing"]     = bool(missing.sum() > 0)

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dupes
    report["has_duplicates"]  = n_dupes > 0

    # Column classification
    id_cols  = detect_id_columns(df)
    tgt_col  = detect_target_column(df)
    exclude  = set(id_cols + ([tgt_col] if tgt_col else []))
    num_cols = detect_numerical_columns(df, exclude=exclude)
    cat_cols = detect_categorical_columns(df, exclude=exclude)

    report["id_columns"]          = id_cols
    report["target_column"]       = tgt_col
    report["numerical_columns"]   = num_cols
    report["categorical_columns"] = cat_cols

    # Target class balance
    if tgt_col:
        vc = df[tgt_col].value_counts(normalize=True)
        report["target_distribution"] = vc.to_dict()
        report["default_rate"]        = float(df[tgt_col].mean())
        report["class_imbalanced"]    = bool(
            df[tgt_col].value_counts(normalize=True).min() < 0.15
        )

    logger.info(
        f"Validation -- rows: {report['n_rows']}, "
        f"missing: {sum(report['missing_values'].values())}, "
        f"duplicates: {report['duplicate_rows']}"
    )
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Schema extraction (for UI generation)
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_schema(df: pd.DataFrame, target_col: str, id_cols: list[str]) -> dict:
    """
    Return a dict describing each feature (type, range, categories, etc.)
    Used by app.py to build input widgets dynamically.
    """
    exclude = set(id_cols + [target_col])
    schema = {}
    for col in df.columns:
        if col in exclude:
            continue
        col_data = df[col].dropna()
        if df[col].dtype == object or df[col].dtype.name == "category":
            schema[col] = {
                "type": "categorical",
                "categories": sorted(col_data.unique().tolist()),
                "default": col_data.mode().iloc[0] if len(col_data) > 0 else "",
            }
        else:
            schema[col] = {
                "type": "numerical",
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "default": float(col_data.median()),
            }
    return schema


# ──────────────────────────────────────────────────────────────────────────────
# Quick summary (for README / debug)
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(df: pd.DataFrame):
    report = validate_dataset(df)
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Rows             : {report['n_rows']:,}")
    print(f"  Columns          : {report['n_cols']}")
    print(f"  Target column    : {report['target_column']}")
    print(f"  Numerical cols   : {report['numerical_columns']}")
    print(f"  Categorical cols : {report['categorical_columns']}")
    print(f"  ID columns       : {report['id_columns']}")
    print(f"  Missing values   : {report['has_missing']} -> {report['missing_values']}")
    print(f"  Duplicates       : {report['duplicate_rows']}")
    if "default_rate" in report:
        print(f"  Default rate     : {report['default_rate']:.1%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    df = load_dataset()
    print_dataset_summary(df)

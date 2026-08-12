# -*- coding: utf-8 -*-
"""
train_model.py -- End-to-end training script.

Usage:
    python src/train_model.py

Workflow:
  1. Load & validate dataset
  2. Clean & preprocess (no leakage)
  3. Train Logistic Regression, Random Forest, XGBoost (if available)
  4. Evaluate every model on the test set
  5. Select best model (ROC-AUC primary, then F1-recall weighted)
  6. Save: best model, preprocessing pipeline, model metadata
  7. Print training summary

Do NOT retrain in Streamlit -- load saved artifacts only.
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

# Fix Windows console encoding issues
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── ensure project root is on path ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATA_DIR, MODELS_DIR,
    DEFAULT_MODEL_PATH, PREPROCESSING_PATH, MODEL_METADATA_PATH,
    RANDOM_STATE,
)
from src.data_loader import load_dataset, find_dataset, print_dataset_summary
from src.preprocessing import prepare_data, save_preprocessing_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation (only used if no real dataset exists)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(path: str, n_samples: int = 10_000):
    """
    Generate a realistic credit-risk CSV if no real dataset is found.
    Clearly documented as SYNTHETIC data.
    Column schema mirrors the Kaggle Credit Risk Dataset columns.
    """
    print("\n[WARNING] No real dataset found. Generating SYNTHETIC credit risk data.")
    print("   For real results, place credit_risk.csv in the data/ folder.\n")

    rng = np.random.default_rng(RANDOM_STATE)

    n = n_samples
    age        = rng.integers(18, 75, size=n)
    income     = rng.lognormal(mean=10.8, sigma=0.55, size=n).astype(int)
    income     = np.clip(income, 12_000, 500_000)
    emp_length = np.clip(rng.poisson(lam=5, size=n), 0, 40).astype(float)

    intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    intent  = rng.choice(intents, size=n, p=[0.22, 0.20, 0.18, 0.12, 0.15, 0.13])

    grades      = ['A', 'B', 'C', 'D', 'E', 'F']
    grade_probs = [0.28, 0.27, 0.21, 0.13, 0.08, 0.03]
    grade       = rng.choice(grades, size=n, p=grade_probs)

    loan_amnt = np.clip(
        rng.lognormal(mean=9.2, sigma=0.7, size=n).astype(int), 500, 50_000
    )

    grade_rate_map = {
        'A': (5.0, 8.5), 'B': (8.5, 11.5), 'C': (11.5, 14.5),
        'D': (14.5, 17.5), 'E': (17.5, 21.0), 'F': (21.0, 25.0),
    }
    loan_int_rate = np.array([
        round(float(rng.uniform(*grade_rate_map[g])), 2) for g in grade
    ])

    pct_income  = np.round(loan_amnt / income, 4)
    hist_length = np.clip(age - rng.integers(17, 24, size=n), 1, 45)
    cb_default  = rng.choice(['Y', 'N'], size=n, p=[0.15, 0.85])

    # Logistic-style risk score → binary target
    grade_risk = {'A': 0.0, 'B': 0.1, 'C': 0.2, 'D': 0.35, 'E': 0.5, 'F': 0.65}
    g_risk = np.array([grade_risk[g] for g in grade])

    logit = (
        -2.5
        + 2.5 * (cb_default == 'Y').astype(float)
        + 2.0 * (pct_income > 0.40).astype(float)
        + 1.5 * g_risk
        + 0.8 * (loan_int_rate > 15).astype(float)
        + 0.5 * (emp_length < 2).astype(float)
        - 0.8 * (income > 100_000).astype(float)
        + rng.normal(0, 0.8, size=n)
    )
    prob        = 1 / (1 + np.exp(-logit))
    loan_status = (prob > 0.50).astype(int)

    df = pd.DataFrame({
        'person_age':                age,
        'person_income':             income,
        'person_emp_length':         emp_length,
        'loan_intent':               intent,
        'loan_grade':                grade,
        'loan_amnt':                 loan_amnt,
        'loan_int_rate':             loan_int_rate,
        'loan_percent_income':       pct_income,
        'cb_person_default_on_file': cb_default,
        'cb_person_cred_hist_length': hist_length,
        'loan_status':               loan_status,
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"   Synthetic dataset saved -> {path}")
    print(f"   Rows: {len(df):,}  |  Default rate: {loan_status.mean():.1%}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def _evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics = {
        "model_name": name,
        "accuracy":   round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }
    return metrics


def _selection_score(m: dict) -> float:
    """Composite score: 50% ROC-AUC + 30% Recall + 20% F1 (credit-risk aware)."""
    return 0.50 * m["roc_auc"] + 0.30 * m["recall"] + 0.20 * m["f1"]


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str | None = None):
    print("\n" + "=" * 65)
    print("  BANKING CREDIT RISK MODEL TRAINING")
    print("=" * 65)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    if dataset_path is None:
        dataset_path = find_dataset()

    if dataset_path is None or not os.path.isfile(dataset_path):
        # Generate synthetic if no real data provided
        synthetic_path = os.path.join(DATA_DIR, "credit_risk.csv")
        generate_synthetic_dataset(synthetic_path)
        dataset_path = synthetic_path

    df = load_dataset(dataset_path)
    print_dataset_summary(df)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    print("Building preprocessing pipeline & splitting data...")
    data = prepare_data(df)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]
    target_col    = data["target_col"]
    preprocessor  = data["preprocessor"]

    cw = _get_class_weights(y_train)

    # ── 3. Train models ──────────────────────────────────────────────────────
    print("\nTraining models (this may take a minute)...\n")
    trained_models = {}

    # Logistic Regression
    print("  [1/3] Logistic Regression...", end=" ", flush=True)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    trained_models["Logistic Regression"] = lr
    print("done")

    # Random Forest
    print("  [2/3] Random Forest...", end=" ", flush=True)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    trained_models["Random Forest"] = rf
    print("done")

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier
        print("  [3/3] XGBoost...", end=" ", flush=True)
        scale_pos = int(np.sum(y_train == 0)) / max(1, int(np.sum(y_train == 1)))
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        trained_models["XGBoost"] = xgb
        print("done")
    except ImportError:
        print("  [3/3] XGBoost not installed -- skipping.")

    # ── 4. Evaluate all models ───────────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 65)

    all_metrics = {}
    for name, model in trained_models.items():
        m = _evaluate(name, model, X_test, y_test)
        all_metrics[name] = m
        print(
            f"  {name:<23} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f}"
        )

    print("-" * 65)

    # ── 5. Select best model ─────────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda n: _selection_score(all_metrics[n]))
    best_model = trained_models[best_name]
    best_metrics = all_metrics[best_name]

    print(f"\n[BEST] Best model selected: {best_name}")
    print(
        f"   Selection score = 0.50*AUC + 0.30*Recall + 0.20*F1 "
        f"= {_selection_score(best_metrics):.4f}"
    )

    # Feature importances
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        feat_imp = dict(sorted(
            zip(feature_names, imp.tolist()),
            key=lambda x: x[1], reverse=True,
        ))
    elif hasattr(best_model, "coef_"):
        coef = np.abs(best_model.coef_[0])
        feat_imp = dict(sorted(
            zip(feature_names, coef.tolist()),
            key=lambda x: x[1], reverse=True,
        ))

    # ── 6. Save artifacts ────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved model -> {DEFAULT_MODEL_PATH}")

    save_preprocessing_pipeline(preprocessor, PREPROCESSING_PATH)

    metadata = {
        "best_model_name":  best_name,
        "target_column":    target_col,
        "feature_names":    feature_names,
        "num_features":     data["num_features"],
        "cat_features":     data["cat_features"],
        "all_metrics":      all_metrics,
        "best_metrics":     best_metrics,
        "feature_importance": feat_imp,
        "dataset_path":     dataset_path,
        "n_train":          int(len(X_train)),
        "n_test":           int(len(X_test)),
        "default_rate":     float(y_train.mean()),
    }
    with open(MODEL_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata -> {MODEL_METADATA_PATH}")

    # ── 7. Train segmentation ────────────────────────────────────────────────
    print("\nTraining customer segmentation model...")
    try:
        from src.segmentation import train_segmentation
        id_cols = metadata.get("id_columns_detected", [])
        train_segmentation(df, target_col, id_cols)
    except Exception as e:
        print(f"  [WARNING] Segmentation training warning: {e}")

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Dataset rows (train/test) : {len(X_train):,} / {len(X_test):,}")
    print(f"  Target column             : {target_col}")
    print(f"  Best model                : {best_name}")
    print(f"  Accuracy                  : {best_metrics['accuracy']:.4f}")
    print(f"  Precision                 : {best_metrics['precision']:.4f}")
    print(f"  Recall                    : {best_metrics['recall']:.4f}")
    print(f"  F1-Score                  : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC                   : {best_metrics['roc_auc']:.4f}")
    print("=" * 65 + "\n")

    return metadata


if __name__ == "__main__":
    train()

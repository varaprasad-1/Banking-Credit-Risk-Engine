"""
config.py -- Central configuration for Banking Credit Risk & Cross-Sell Engine.
All tuneable parameters are kept here. Do NOT scatter config values in other files.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Accepted dataset filenames (searched in order)
DATASET_CANDIDATES = [
    "credit_risk.csv",
    "credit_risk_dataset.csv",
    "credit_default.csv",
    "loan_data.csv",
    "german_credit.csv",
]

# ─────────────────────────────────────────────
# TARGET COLUMN
# ─────────────────────────────────────────────
# The script will auto-detect the target; set this manually if auto-detect fails.
TARGET_COLUMN = None  # e.g. "loan_status" or "default"

# Possible names for the target column (searched in order, case-insensitive)
TARGET_COLUMN_CANDIDATES = [
    "loan_status", "default", "default_on_payment",
    "credit_default", "target", "label", "y",
    "defaulted", "loan_default", "bad_flag",
]

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
DEFAULT_MODEL_PATH      = os.path.join(MODELS_DIR, "default_model.pkl")
PREPROCESSING_PATH      = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "segmentation_model.pkl")
MODEL_METADATA_PATH     = os.path.join(MODELS_DIR, "model_metadata.pkl")

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20       # 80/20 train-test split

# Class imbalance strategy: "smote", "class_weight", or "none"
IMBALANCE_STRATEGY = "class_weight"

# ─────────────────────────────────────────────
# RISK THRESHOLDS
# (PROTOTYPE values — NOT real regulatory thresholds)
# ─────────────────────────────────────────────
LOW_THRESHOLD  = 0.30   # prob < LOW  -> LOW risk
HIGH_THRESHOLD = 0.60   # prob > HIGH -> HIGH risk
# LOW_THRESHOLD ≤ prob ≤ HIGH_THRESHOLD → MEDIUM risk

# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────
N_CLUSTERS_RANGE = (2, 8)    # range tested during elbow/silhouette analysis
N_CLUSTERS       = 4         # default; overridden by auto-selection if enabled
AUTO_SELECT_CLUSTERS = True  # set False to always use N_CLUSTERS

# ─────────────────────────────────────────────
# ID COLUMN DETECTION
# ─────────────────────────────────────────────
ID_COLUMN_PATTERNS = [
    "id", "_id", "customer_id", "loan_id", "applicant_id",
    "account_id", "uuid", "index",
]

# ─────────────────────────────────────────────
# CROSS-SELL PRODUCTS
# ─────────────────────────────────────────────
BANKING_PRODUCTS = [
    "Savings Account",
    "Credit Card",
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Fixed Deposit",
    "Investment Account",
    "Insurance",
    "Premium Banking Account",
]

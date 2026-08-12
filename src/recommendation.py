"""
recommendation.py -- Risk-aware banking product cross-sell engine.

METHODOLOGY: Customer Segmentation + Risk-Aware Business Rules.

IMPORTANT LIMITATION:
  The Credit Risk Dataset does NOT contain transaction or product purchase history.
  Therefore, collaborative filtering / market-basket analysis cannot be applied.
  This engine uses:
    - Predicted default probability (risk level)
    - Customer segment label
    - Available financial features (income, loan amount, etc.)
    - Business rules that encode domain knowledge

  This architecture allows a real recommendation model (e.g., matrix factorization,
  neural collaborative filtering) to be added later once purchase history is available.

PRODUCTS:
  Savings Account | Credit Card | Personal Loan | Home Loan | Auto Loan |
  Fixed Deposit   | Investment Account | Insurance | Premium Banking Account

RISK AWARENESS:
  High-risk customers are NOT recommended additional debt products.
  Savings-oriented and protection products are prioritised for high-risk customers.
"""

import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import LOW_THRESHOLD, HIGH_THRESHOLD, BANKING_PRODUCTS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Product catalogue
# ─────────────────────────────────────────────────────────────────────────────

PRODUCT_CATALOGUE = {
    "Savings Account": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],   # suitable for all
        "description": "A liquid, interest-bearing savings account.",
        "base_score":  70,
    },
    "Credit Card": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM"],
        "description": "Revolving credit facility with rewards programme.",
        "base_score":  75,
    },
    "Personal Loan": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM"],
        "description": "Unsecured personal loan with flexible tenure.",
        "base_score":  65,
    },
    "Home Loan": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW"],
        "description": "Mortgage loan for property purchase or construction.",
        "base_score":  80,
    },
    "Auto Loan": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM"],
        "description": "Vehicle financing up to 90% of on-road price.",
        "base_score":  72,
    },
    "Fixed Deposit": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],
        "description": "Guaranteed returns with capital protection.",
        "base_score":  68,
    },
    "Investment Account": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM"],
        "description": "Market-linked investment with growth potential.",
        "base_score":  73,
    },
    "Insurance": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],
        "description": "Life and health protection coverage.",
        "base_score":  74,
    },
    "Premium Banking Account": {
        "min_risk":    "LOW",
        "risk_levels": ["LOW"],
        "description": "Exclusive banking with priority service and higher limits.",
        "base_score":  85,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring adjustments
# ─────────────────────────────────────────────────────────────────────────────

def _income_tier(income: float | None) -> str:
    if income is None:
        return "unknown"
    if income > 120_000:
        return "high"
    elif income > 50_000:
        return "medium"
    else:
        return "low"


def _score_product(
    product: str,
    risk_level: str,
    default_prob: float,
    segment_label: str,
    income: float | None,
    loan_amnt: float | None,
    loan_pct_income: float | None,
    extra_features: dict,
) -> tuple[int, str]:
    """
    Compute recommendation score (0-100) and reason string for one product.
    Returns (score, reason).
    """
    meta = PRODUCT_CATALOGUE[product]

    if risk_level not in meta["risk_levels"]:
        return 0, "Not recommended for current risk profile."

    score  = meta["base_score"]
    reason = meta["description"]
    inc_t  = _income_tier(income)

    # ── Risk adjustments ────────────────────────────────────────────────────
    if risk_level == "LOW":
        score += 10
        reason = f"Low default risk ({default_prob*100:.1f}%) makes this a strong fit."
    elif risk_level == "MEDIUM":
        score -= 5
        reason = f"Moderate risk ({default_prob*100:.1f}%). Standard eligibility criteria apply."
    # HIGH risk: already filtered above for credit/loan products

    # ── Income adjustments ──────────────────────────────────────────────────
    if inc_t == "high":
        if product in ["Premium Banking Account", "Investment Account", "Home Loan"]:
            score += 12
            reason += " High income supports eligibility."
        elif product in ["Fixed Deposit", "Savings Account"]:
            score += 8
    elif inc_t == "low":
        if product in ["Premium Banking Account", "Home Loan", "Investment Account"]:
            score = max(score - 20, 10)
            reason += " Lower income may limit access to premium products."

    # ── Loan-to-income adjustments ──────────────────────────────────────────
    if loan_pct_income is not None:
        if loan_pct_income > 0.40 and product in ["Personal Loan", "Auto Loan", "Credit Card"]:
            score -= 15
            reason += " High existing loan-to-income ratio is a concern."
        elif loan_pct_income < 0.15 and product in ["Personal Loan", "Credit Card"]:
            score += 8
            reason += " Low existing debt burden supports additional credit."

    # ── Segment adjustments ─────────────────────────────────────────────────
    if "Premium" in segment_label and product == "Premium Banking Account":
        score += 15
    if "High-Risk" in segment_label:
        if product in ["Savings Account", "Fixed Deposit", "Insurance"]:
            score += 10
            reason += " Financial safety products are recommended for risk-conscious customers."
        if product in ["Personal Loan", "Credit Card", "Auto Loan"]:
            score -= 20  # Penalise additional credit for high-risk
    if "Emerging" in segment_label and product == "Savings Account":
        score += 8

    return min(100, max(0, score)), reason


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def recommend_products(
    risk_level: str,
    default_probability: float,
    segment_label: str,
    income: float | None = None,
    loan_amnt: float | None = None,
    loan_pct_income: float | None = None,
    extra_features: dict | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Return top-N product recommendations for a customer.

    Parameters
    ----------
    risk_level           : "LOW" | "MEDIUM" | "HIGH"
    default_probability  : float 0-1
    segment_label        : string from segmentation module
    income               : annual income (optional)
    loan_amnt            : loan amount (optional)
    loan_pct_income      : loan as % of income (optional)
    extra_features       : any additional feature dict (optional)
    top_n                : number of recommendations to return

    Returns
    -------
    List of dicts with keys:
        product, score, reason, eligibility_note
    """
    extra = extra_features or {}
    recommendations = []

    for product in BANKING_PRODUCTS:
        score, reason = _score_product(
            product        = product,
            risk_level     = risk_level,
            default_prob   = default_probability,
            segment_label  = segment_label,
            income         = income,
            loan_amnt      = loan_amnt,
            loan_pct_income= loan_pct_income,
            extra_features = extra,
        )
        if score <= 0:
            continue

        # Eligibility note
        if risk_level == "HIGH" and product in ["Personal Loan", "Credit Card", "Auto Loan"]:
            elig_note = "[WARN]️ Not recommended: High default risk detected."
        elif risk_level == "HIGH" and product in ["Home Loan", "Investment Account", "Premium Banking Account"]:
            elig_note = "[WARN]️ Not recommended: Insufficient risk profile."
        elif risk_level == "LOW" and score >= 80:
            elig_note = "[OK] Strong match."
        elif score >= 70:
            elig_note = "[OK] Good match."
        else:
            elig_note = "ℹ️ Conditional match -- review eligibility."

        recommendations.append({
            "product":          product,
            "score":            score,
            "reason":           reason,
            "eligibility_note": elig_note,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level opportunity sizing
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_opportunities(df_with_risk: "pd.DataFrame") -> dict:
    """
    Given a DataFrame with columns [risk_level, segment_label, person_income, ...],
    estimate how many customers in the dataset are suitable for each product.
    Returns a dict: product -> count.
    """
    import pandas as pd

    results = {}
    for product in BANKING_PRODUCTS:
        meta = PRODUCT_CATALOGUE[product]
        allowed_risk = meta["risk_levels"]
        if "risk_level" in df_with_risk.columns:
            count = int(df_with_risk["risk_level"].isin(allowed_risk).sum())
        else:
            count = len(df_with_risk)
        results[product] = count
    return results

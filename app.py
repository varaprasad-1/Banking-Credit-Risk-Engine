"""
app.py — Banking Credit Default Risk & Cross-Sell Engine
Multi-page Streamlit application.

Run with:  streamlit run app.py

Pages:
  1. Dashboard
  2. Risk Assessment
  3. Cross-Sell Recommendations
  4. Customer Segmentation
  5. Model Performance
  6. Data Explorer
  7. About
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Credit Risk & Cross-Sell Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #1d4ed8;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(29,78,216,0.12);
}
.kpi-label { color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { color: #0f172a; font-size: 1.85rem; font-weight: 800; line-height: 1; }
.kpi-sub   { color: #94a3b8; font-size: 0.78rem; margin-top: 4px; }

/* ── Section header ────────────────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
    border-left: 4px solid #1d4ed8;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 20px;
    border: 1px solid #dbeafe;
    border-left: 4px solid #1d4ed8;
}
.section-header h2 { color: #1e3a8a; margin: 0; font-size: 1.15rem; font-weight: 700; }
.section-header p  { color: #64748b; margin: 2px 0 0; font-size: 0.82rem; }

/* ── Risk badges ────────────────────────────────────────────────────────── */
.badge-low    { background:#dcfce7; color:#15803d; border:1px solid #86efac; padding:6px 16px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-medium { background:#fef9c3; color:#a16207; border:1px solid #fde047; padding:6px 16px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-high   { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; padding:6px 16px; border-radius:20px; font-weight:700; display:inline-block; }

/* ── Product recommendation card ────────────────────────────────────────── */
.rec-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1d4ed8;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.rec-card:hover { border-left-color: #3b82f6; box-shadow: 0 4px 16px rgba(29,78,216,0.10); }
.rec-name  { color: #1d4ed8; font-size: 1.0rem; font-weight: 700; margin-bottom: 4px; }
.rec-score { float: right; background: #dbeafe; color: #1e40af; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }
.rec-reason { color: #475569; font-size: 0.88rem; margin-top: 4px; }
.rec-elig   { color: #94a3b8; font-size: 0.78rem; margin-top: 6px; }

/* ── Factor lists ────────────────────────────────────────────────────────── */
.factor-risk { background:#fff1f2; border-left:3px solid #ef4444; padding:8px 12px; border-radius:6px; margin-bottom:6px; color:#b91c1c; font-size:0.88rem; }
.factor-pos  { background:#f0fdf4; border-left:3px solid #22c55e; padding:8px 12px; border-radius:6px; margin-bottom:6px; color:#15803d; font-size:0.88rem; }

/* ── Warning box ────────────────────────────────────────────────────────── */
.warning-box { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:12px 16px; color:#92400e; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    from src.config import DEFAULT_MODEL_PATH, PREPROCESSING_PATH, MODEL_METADATA_PATH, SEGMENTATION_MODEL_PATH
    artifacts = {}
    try:
        with open(DEFAULT_MODEL_PATH, "rb") as f:
            artifacts["model"] = pickle.load(f)
        with open(PREPROCESSING_PATH, "rb") as f:
            artifacts["preprocessor"] = pickle.load(f)
        with open(MODEL_METADATA_PATH, "rb") as f:
            artifacts["metadata"] = pickle.load(f)
        artifacts["trained"] = True
    except FileNotFoundError:
        artifacts["trained"] = False

    try:
        with open(SEGMENTATION_MODEL_PATH, "rb") as f:
            artifacts["seg"] = pickle.load(f)
    except FileNotFoundError:
        artifacts["seg"] = None

    return artifacts


@st.cache_data(show_spinner=False)
def load_dataset_cached():
    from src.data_loader import find_dataset, load_dataset, validate_dataset
    path = find_dataset()
    if path is None:
        return None, None
    df = load_dataset(path)
    report = validate_dataset(df)
    return df, report


@st.cache_data(show_spinner=False)
def get_batch_predictions(_df, _artifacts):
    """Run predictions on whole dataset for dashboard stats."""
    if not _artifacts.get("trained") or _df is None:
        return None
    try:
        from src.prediction import batch_predict
        return batch_predict(_df)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def models_not_trained_warning():
    st.error(
        "⚠️ **Models not trained yet.**\n\n"
        "Run the training script first:\n"
        "```\npython src/train_model.py\n```"
    )


def kpi_card(label: str, value: str, sub: str = ""):
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def risk_badge(level: str):
    cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(level, "badge-medium")
    return f'<span class="{cls}">{level} RISK</span>'


CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(color="#334155", family="Inter"),
    margin=dict(t=40, l=10, r=10, b=10),
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 Credit Risk Engine")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Risk Assessment",
            "🎯 Cross-Sell Recommendations",
            "👥 Customer Segmentation",
            "📈 Model Performance",
            "🗂️ Data Explorer",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Banking Credit Risk &  Cross-Sell Engine")

# Load artifacts and data
artifacts = load_artifacts()
df_raw, data_report = load_dataset_cached()
df_pred = get_batch_predictions(df_raw, artifacts)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("""
    <div class="section-header">
        <h2>📊 Portfolio Analytics Dashboard</h2>
        <p>Real-time credit risk overview derived from the loaded dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.warning("No dataset loaded. Run `python src/train_model.py` to generate data.")
        st.stop()

    target_col = data_report.get("target_column")

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_customers = len(df_raw)
    default_rate = df_raw[target_col].mean() if target_col else 0.0

    if df_pred is not None and "risk_level" in df_pred.columns:
        high_risk     = int((df_pred["risk_level"] == "HIGH").sum())
        avg_prob      = float(df_pred["default_prob"].mean())
        medium_risk   = int((df_pred["risk_level"] == "MEDIUM").sum())
        low_risk      = int((df_pred["risk_level"] == "LOW").sum())
    else:
        high_risk   = int(df_raw[target_col].sum()) if target_col else 0
        avg_prob    = float(default_rate)
        medium_risk = 0
        low_risk    = total_customers - high_risk

    cross_sell_opps = medium_risk + low_risk

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Total Customers", f"{total_customers:,}", "in dataset"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Historical Default Rate", f"{default_rate:.1%}", "actual labels"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("High-Risk Customers", f"{high_risk:,}", "predicted"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Avg Default Probability", f"{avg_prob:.1%}", "model output"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Cross-Sell Opportunities", f"{cross_sell_opps:,}", "low + medium risk"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        if target_col and target_col in df_raw.columns:
            vc = df_raw[target_col].value_counts().reset_index()
            vc.columns = ["Status", "Count"]
            vc["Status"] = vc["Status"].map({0: "No Default", 1: "Default"})
            fig = px.pie(
                vc, names="Status", values="Count",
                title="Default Distribution",
                color="Status",
                color_discrete_map={"No Default": "#22c55e", "Default": "#ef4444"},
                hole=0.4,
            )
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if df_pred is not None and "risk_level" in df_pred.columns:
            rl_counts = df_pred["risk_level"].value_counts().reset_index()
            rl_counts.columns = ["Risk Level", "Count"]
            colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
            fig2 = px.bar(
                rl_counts, x="Risk Level", y="Count",
                title="Predicted Risk Level Distribution",
                color="Risk Level",
                color_discrete_map=colors,
            )
            fig2.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Scatter: income vs loan amount (if cols exist) ────────────────────
    income_col   = next((c for c in ["person_income", "income", "annual_inc"] if c in df_raw.columns), None)
    loan_col     = next((c for c in ["loan_amnt", "loan_amount"] if c in df_raw.columns), None)

    if income_col and loan_col and target_col:
        sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
        fig3 = px.scatter(
            sample,
            x=income_col, y=loan_col,
            color=target_col,
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            title=f"{income_col.replace('_',' ').title()} vs {loan_col.replace('_',' ').title()} by Default Status",
            opacity=0.6,
            labels={str(target_col): "Default"},
        )
        fig3.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: Risk Assessment
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Risk Assessment":
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Customer Risk Assessment</h2>
        <p>Enter customer profile to predict default probability and risk level</p>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts.get("trained"):
        models_not_trained_warning()
        st.stop()

    metadata = artifacts["metadata"]
    num_features = metadata.get("num_features", [])
    cat_features = metadata.get("cat_features", [])

    # ── Dynamic form from actual feature schema ────────────────────────────
    st.markdown("### 📋 Customer Profile")

    # Get feature stats from metadata if dataset is available
    feat_stats = {}
    if df_raw is not None:
        for col in num_features:
            if col in df_raw.columns:
                feat_stats[col] = {
                    "min":    float(df_raw[col].dropna().min()),
                    "max":    float(df_raw[col].dropna().max()),
                    "mean":   float(df_raw[col].dropna().mean()),
                    "median": float(df_raw[col].dropna().median()),
                }
        for col in cat_features:
            if col in df_raw.columns:
                feat_stats[col] = {"categories": sorted(df_raw[col].dropna().unique().tolist())}

    col1, col2 = st.columns(2)
    input_values = {}

    all_features = num_features + cat_features
    half = len(all_features) // 2

    def nice_label(col: str) -> str:
        return col.replace("_", " ").title()

    with col1:
        for feat in all_features[:half]:
            if feat in cat_features:
                cats = feat_stats.get(feat, {}).get("categories", ["unknown"])
                input_values[feat] = st.selectbox(nice_label(feat), cats, key=f"inp_{feat}")
            else:
                stats = feat_stats.get(feat, {"min": 0.0, "max": 1e6, "median": 50.0})
                mn, mx, md = stats["min"], stats["max"], stats["median"]
                input_values[feat] = st.number_input(
                    nice_label(feat), min_value=float(mn), max_value=float(mx),
                    value=float(md), key=f"inp_{feat}"
                )

    with col2:
        for feat in all_features[half:]:
            if feat in cat_features:
                cats = feat_stats.get(feat, {}).get("categories", ["unknown"])
                input_values[feat] = st.selectbox(nice_label(feat), cats, key=f"inp_{feat}")
            else:
                stats = feat_stats.get(feat, {"min": 0.0, "max": 1e6, "median": 50.0})
                mn, mx, md = stats["min"], stats["max"], stats["median"]
                input_values[feat] = st.number_input(
                    nice_label(feat), min_value=float(mn), max_value=float(mx),
                    value=float(md), key=f"inp_{feat}"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Assess Customer", use_container_width=True, type="primary"):
        try:
            from src.prediction import predict
            from src.explainability import explain_prediction
            from src.recommendation import recommend_products
            from src.segmentation import predict_segment
            from src.config import SEGMENTATION_MODEL_PATH

            result = predict(input_values)
            explanation = explain_prediction(input_values)

            prob       = result["default_probability"]
            risk_level = result["risk_level"]
            risk_score = result["risk_score"]
            color      = result["risk_color"]

            # Segment
            seg_result = {"segment_id": 0, "segment_label": "Unknown"}
            if os.path.isfile(SEGMENTATION_MODEL_PATH):
                try:
                    seg_result = predict_segment(input_values)
                except Exception:
                    pass

            st.markdown("---")
            st.markdown("## 📊 Risk Assessment Results")

            r1, r2, r3 = st.columns(3)
            r1.metric("Default Probability", f"{prob*100:.1f}%")
            r2.markdown(
                f"**Risk Level**<br>{risk_badge(risk_level)}",
                unsafe_allow_html=True
            )
            r3.metric("Customer Segment", seg_result["segment_label"])

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Default Risk Score", "font": {"color": "#334155", "size": 14}},
                number={"suffix": "%", "font": {"color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": color},
                    "bgcolor": "#f1f5f9",
                    "bordercolor": "#cbd5e1",
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(34,197,94,0.2)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                        {"range": [60, 100],"color": "rgba(239,68,68,0.2)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": risk_score},
                }
            ))
            fig_gauge.update_layout(height=230, **CHART_LAYOUT)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk factors
            rf_col, pf_col = st.columns(2)
            with rf_col:
                st.markdown("#### ⚠️ Risk Contributing Factors")
                st.caption(f"*Explanation method: {explanation['method']}. These are contributing factors, not causal claims.*")
                if explanation["risk_factors"]:
                    for f in explanation["risk_factors"]:
                        st.markdown(f'<div class="factor-risk">⬆ {f["feature"]}<br><small>{f["description"]}</small></div>', unsafe_allow_html=True)
                else:
                    st.info("No major risk factors detected.")

            with pf_col:
                st.markdown("#### ✅ Positive Contributing Factors")
                if explanation["positive_factors"]:
                    for f in explanation["positive_factors"]:
                        st.markdown(f'<div class="factor-pos">⬇ {f["feature"]}<br><small>{f["description"]}</small></div>', unsafe_allow_html=True)
                else:
                    st.info("No positive factors detected.")

            # Recommendations
            st.markdown("---")
            st.markdown("#### 🎯 Recommended Products")
            income_val = input_values.get(
                next((f for f in ["person_income", "income", "annual_inc"] if f in input_values), ""), None
            )
            loan_val   = input_values.get(
                next((f for f in ["loan_amnt", "loan_amount"] if f in input_values), ""), None
            )
            lpi_val    = input_values.get(
                next((f for f in ["loan_percent_income"] if f in input_values), ""), None
            )
            recs = recommend_products(
                risk_level          = risk_level,
                default_probability = prob,
                segment_label       = seg_result["segment_label"],
                income              = float(income_val) if income_val is not None else None,
                loan_amnt           = float(loan_val) if loan_val is not None else None,
                loan_pct_income     = float(lpi_val) if lpi_val is not None else None,
            )
            for rec in recs:
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-score">Score: {rec['score']}</span>
                    <div class="rec-name">💳 {rec['product']}</div>
                    <div class="rec-reason">{rec['reason']}</div>
                    <div class="rec-elig">{rec['eligibility_note']}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Assessment failed: {e}")
            st.exception(e)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: Cross-Sell Recommendations
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Cross-Sell Recommendations":
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Cross-Sell Recommendation Engine</h2>
        <p>Risk-aware product recommendations based on customer segmentation + business rules</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ℹ️ <strong>Methodology Note:</strong> The Credit Risk dataset does not contain product purchase history.
    This engine uses <strong>Customer Segmentation + Risk-Aware Business Rules</strong>.
    A collaborative filtering or neural recommendation model can be added once transaction data is available.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Quick profile inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_risk = st.selectbox("Customer Risk Level", ["LOW", "MEDIUM", "HIGH"])
        demo_prob = {"LOW": 0.15, "MEDIUM": 0.45, "HIGH": 0.75}[demo_risk]
    with c2:
        demo_segment = st.selectbox("Customer Segment", ["Premium Customer", "Stable Customer", "Emerging Customer", "High-Risk Customer"])
    with c3:
        demo_income = st.number_input("Annual Income ($)", min_value=10_000, max_value=500_000, value=65_000, step=5_000)

    from src.recommendation import recommend_products, compute_portfolio_opportunities, PRODUCT_CATALOGUE
    recs = recommend_products(
        risk_level          = demo_risk,
        default_probability = demo_prob,
        segment_label       = demo_segment,
        income              = demo_income,
        top_n               = 9,
    )

    st.markdown("### 🛒 Recommended Products")
    cols = st.columns(3)
    for i, rec in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="rec-card">
                <span class="rec-score">{rec['score']}/100</span>
                <div class="rec-name">{rec['product']}</div>
                <div class="rec-reason">{rec['reason']}</div>
                <div class="rec-elig">{rec['eligibility_note']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Portfolio-Level Product Opportunity Sizing")
    st.caption("Number of customers in the dataset eligible for each product (by risk level).")

    if df_pred is not None:
        opps = compute_portfolio_opportunities(df_pred)
    elif df_raw is not None:
        # Estimate using default rate as proxy
        opps = {}
        from src.recommendation import PRODUCT_CATALOGUE as PC
        for prod, meta in PC.items():
            allowed = meta["risk_levels"]
            opps[prod] = len(df_raw)  # conservative; real risk breakdown needs predictions
    else:
        opps = {p: 0 for p in PRODUCT_CATALOGUE}

    opp_df = pd.DataFrame(list(opps.items()), columns=["Product", "Suitable Customers"])
    fig_opp = px.bar(
        opp_df.sort_values("Suitable Customers", ascending=True),
        x="Suitable Customers", y="Product", orientation="h",
        title="Cross-Sell Opportunity by Product",
        color="Suitable Customers", color_continuous_scale="Blues",
    )
    fig_opp.update_layout(**CHART_LAYOUT, height=400)
    st.plotly_chart(fig_opp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: Customer Segmentation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown("""
    <div class="section-header">
        <h2>👥 Customer Segmentation</h2>
        <p>K-Means clustering of customers by financial profile</p>
    </div>
    """, unsafe_allow_html=True)

    seg = artifacts.get("seg")
    if seg is None:
        st.error("Segmentation model not found. Run `python src/train_model.py` first.")
        st.stop()

    profiles   = seg["profiles"]
    n_clusters = seg["n_clusters"]
    eval_data  = seg.get("eval_data", {})

    st.metric("Number of Segments", n_clusters)

    # Profiles table
    st.markdown("### 📋 Segment Profiles")
    display_cols = ["segment", "segment_label", "cluster_size"]
    available_feat_cols = [c for c in profiles.columns if c not in display_cols and "std" not in c]
    show_cols = display_cols + available_feat_cols[:8]
    show_cols = [c for c in show_cols if c in profiles.columns]
    st.dataframe(profiles[show_cols].rename(columns={"segment": "Cluster ID", "segment_label": "Label", "cluster_size": "Size"}), use_container_width=True)

    # Cluster sizes pie
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            profiles,
            names="segment_label",
            values="cluster_size",
            title="Cluster Size Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_pie.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        if "default_rate" in profiles.columns:
            fig_dr = px.bar(
                profiles, x="segment_label", y="default_rate",
                title="Default Rate by Segment",
                color="segment_label",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"default_rate": "Default Rate", "segment_label": "Segment"},
            )
            fig_dr.update_yaxes(tickformat=".1%")
            fig_dr.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_dr, use_container_width=True)

    # Elbow / Silhouette
    if "k_range" in eval_data:
        st.markdown("### 🔬 Optimal Cluster Selection")
        ec1, ec2 = st.columns(2)
        with ec1:
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(x=eval_data["k_range"], y=eval_data["inertias"], mode="lines+markers", line=dict(color="#3b82f6")))
            fig_el.update_layout(title="Elbow Method (Inertia)", xaxis_title="k", yaxis_title="Inertia", **CHART_LAYOUT)
            st.plotly_chart(fig_el, use_container_width=True)
        with ec2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=eval_data["k_range"], y=eval_data["sil_scores"], mode="lines+markers", line=dict(color="#22c55e")))
            fig_sil.add_vline(x=eval_data["best_k"], line_dash="dash", line_color="#f59e0b", annotation_text=f"Best k={eval_data['best_k']}")
            fig_sil.update_layout(title="Silhouette Score", xaxis_title="k", yaxis_title="Score", **CHART_LAYOUT)
            st.plotly_chart(fig_sil, use_container_width=True)

    # Feature comparison across segments
    seg_features = seg.get("seg_features", [])
    mean_cols = [c for c in profiles.columns if "_mean" in c]
    if mean_cols:
        st.markdown("### 📊 Feature Comparison Across Segments")
        feat_plot_df = profiles[["segment_label"] + mean_cols].copy()
        feat_plot_df_melt = feat_plot_df.melt(id_vars="segment_label", var_name="Feature", value_name="Mean Value")
        fig_comp = px.bar(
            feat_plot_df_melt,
            x="Feature", y="Mean Value",
            color="segment_label",
            barmode="group",
            title="Mean Feature Values by Segment",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_comp.update_layout(**CHART_LAYOUT, height=420)
        st.plotly_chart(fig_comp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="section-header">
        <h2>📈 Model Performance</h2>
        <p>Evaluation metrics for all trained models and selection rationale</p>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts.get("trained"):
        models_not_trained_warning()
        st.stop()

    meta        = artifacts["metadata"]
    all_metrics = meta.get("all_metrics", {})
    best_name   = meta.get("best_model_name", "Unknown")
    best_m      = meta.get("best_metrics", {})
    feat_imp    = meta.get("feature_importance", {})

    # Model comparison table
    st.markdown("### 🏆 Model Comparison")
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model":     f"{'⭐ ' if name == best_name else ''}{name}",
            "Accuracy":  m.get("accuracy", "N/A"),
            "Precision": m.get("precision", "N/A"),
            "Recall":    m.get("recall", "N/A"),
            "F1-Score":  m.get("f1", "N/A"),
            "ROC-AUC":   m.get("roc_auc", "N/A"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.info(
        f"**Selected model: {best_name}** — chosen by composite score: "
        f"50% ROC-AUC + 30% Recall + 20% F1 (credit-risk aware weighting, "
        f"prioritising default detection over accuracy alone)."
    )

    mc1, mc2 = st.columns(2)

    with mc1:
        # Confusion matrix
        cm = best_m.get("confusion_matrix")
        if cm:
            st.markdown("#### Confusion Matrix")
            fig_cm, ax = plt.subplots(figsize=(4, 3))
            fig_cm.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
                ax=ax,
            )
            ax.tick_params(colors="#334155")
            ax.set_title("Confusion Matrix", color="#0f172a", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_cm)

    with mc2:
        # ROC Curve
        fpr = best_m.get("roc_fpr")
        tpr = best_m.get("roc_tpr")
        if fpr and tpr:
            st.markdown("#### ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="#94a3b8", dash="dash"), name="Random"))
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#3b82f6", width=2),
                                         name=f"AUC={best_m['roc_auc']:.4f}"))
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **CHART_LAYOUT,
                height=320,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # Feature importance
    if feat_imp:
        st.markdown("### 🔬 Feature Importance (Top 15)")
        top_feat = dict(list(feat_imp.items())[:15])
        fi_df = pd.DataFrame(list(top_feat.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ").str.title()
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Feature Importance",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(**CHART_LAYOUT, height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Per-class metrics
    cr = best_m.get("classification_report", {})
    if cr:
        st.markdown("### 📋 Classification Report")
        cr_rows = []
        for cls, stats in cr.items():
            if isinstance(stats, dict):
                cr_rows.append({"Class": cls, **{k: round(v, 4) for k, v in stats.items()}})
        if cr_rows:
            st.dataframe(pd.DataFrame(cr_rows).set_index("Class"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: Data Explorer
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Data Explorer":
    st.markdown("""
    <div class="section-header">
        <h2>🗂️ Data Explorer</h2>
        <p>Explore and analyse the loaded credit risk dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.warning("No dataset found. Run `python src/train_model.py` to generate one.")
        st.stop()

    target_col = data_report.get("target_column")

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",            f"{df_raw.shape[0]:,}")
    col2.metric("Columns",         df_raw.shape[1])
    col3.metric("Missing Values",  f"{df_raw.isnull().sum().sum():,}")
    col4.metric("Duplicate Rows",  f"{df_raw.duplicated().sum():,}")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Preview", "📊 Statistics", "❓ Missing Values", "🎯 Target Distribution"])

    with tab1:
        # Search/filter
        filter_col = st.selectbox("Filter by column", ["— none —"] + list(df_raw.columns))
        if filter_col != "— none —" and df_raw[filter_col].dtype == object:
            vals = st.multiselect("Select values", sorted(df_raw[filter_col].unique().tolist()))
            display_df = df_raw[df_raw[filter_col].isin(vals)] if vals else df_raw
        else:
            display_df = df_raw
        st.dataframe(display_df.head(500), use_container_width=True)
        st.caption(f"Showing up to 500 of {len(display_df):,} rows.")

    with tab2:
        st.dataframe(df_raw.describe(include="all").round(3), use_container_width=True)

    with tab3:
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("✅ No missing values detected.")
        else:
            miss_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Pct": (missing.values / len(df_raw) * 100).round(2)})
            st.dataframe(miss_df, use_container_width=True)
            fig_miss = px.bar(miss_df, x="Column", y="Pct", title="Missing Value % by Column", color="Pct", color_continuous_scale="Reds")
            fig_miss.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_miss, use_container_width=True)

    with tab4:
        if target_col:
            vc = df_raw[target_col].value_counts()
            fig_t = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Distribution of '{target_col}'",
                           labels={"x": target_col, "y": "Count"}, color=vc.index.astype(str),
                           color_discrete_sequence=["#22c55e", "#ef4444"])
            fig_t.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_t, use_container_width=True)
            st.metric("Default Rate", f"{df_raw[target_col].mean():.2%}")
        else:
            st.warning("Target column not detected.")


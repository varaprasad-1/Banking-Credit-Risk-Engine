# Banking Credit Default Risk & Cross-Sell Engine

> An end-to-end machine learning application that predicts customer loan default probability and recommends suitable banking products using customer segmentation and risk-aware business rules.

---

## Problem Statement

Banks need to:
1. **Predict** which loan applicants are likely to default, so they can price risk correctly and make sound credit decisions.
2. **Identify** which banking products each customer segment should be offered — without recommending products that increase financial stress for already high-risk customers.

---

## Objectives

- Build a production-quality credit default risk prediction model trained on real or realistic data
- Select the best model using credit-risk aware metrics (not accuracy alone)
- Segment customers into meaningful groups using unsupervised learning
- Recommend banking products based on risk level, customer segment, and financial profile
- Deliver a professional, interactive Streamlit dashboard for analysts and relationship managers

---

## Features

| Feature | Details |
|---|---|
| Credit Risk Prediction | Logistic Regression, Random Forest, XGBoost; best selected by AUC/Recall/F1 |
| Risk Levels | LOW (<30%), MEDIUM (30-60%), HIGH (>60%) — configurable thresholds |
| Explainability | SHAP (preferred) or feature-importance with "contributing factor" language |
| Customer Segmentation | K-Means with Elbow + Silhouette k-selection |
| Product Recommendations | 9 banking products with risk-aware business rules |
| Multi-page Streamlit App | Dashboard, Risk Assessment, Cross-Sell, Segmentation, Model Perf, Data Explorer |

---

## Architecture

```
Banking-Credit-Risk-Engine/
|
+-- data/
|   +-- credit_risk.csv               # Dataset (real or synthetic)
|
+-- models/                           # Saved training artifacts
|   +-- default_model.pkl             # Best trained classifier
|   +-- preprocessing_pipeline.pkl    # Fitted sklearn pipeline
|   +-- segmentation_model.pkl        # K-Means + scaler
|   +-- model_metadata.pkl            # Metrics, feature names, etc.
|
+-- src/
|   +-- config.py                     # All configuration in one place
|   +-- data_loader.py                # Dataset discovery, loading, validation
|   +-- preprocessing.py             # Feature pipeline (impute, scale, encode)
|   +-- train_model.py               # Training script (run once)
|   +-- prediction.py                # Inference module (no retraining)
|   +-- explainability.py            # SHAP / feature-importance explanations
|   +-- segmentation.py             # K-Means customer clustering
|   +-- recommendation.py           # Risk-aware product recommendations
|
+-- notebooks/
|   +-- EDA_Model_Training.ipynb    # Analysis notebook
|
+-- app.py                           # Streamlit application
+-- requirements.txt
+-- README.md
+-- .streamlit/config.toml
```

---

## Dataset

### If You Have a Real Dataset

Place your credit risk CSV in:
```
data/credit_risk.csv
```

The system will **automatically detect** the target column, numerical features, and categorical features. No hard-coded column assumptions.

Set the target column in `src/config.py` if auto-detection fails:
```python
TARGET_COLUMN = "loan_status"   # or whatever your column is named
```

### If No Dataset Is Provided

The training script generates a **synthetic** dataset (10,000 rows) with realistic credit risk patterns. The synthetic data is clearly documented and the README notes this limitation.

**Supported column schemas:**
- Kaggle Credit Risk Dataset: `person_age`, `person_income`, `loan_amnt`, `loan_status`, etc.
- German Credit Dataset: `credit_amount`, `default`, etc.
- Any CSV with a binary 0/1 default column

---

## ML Workflow

```
Raw CSV
  -> Data validation & cleaning
  -> Feature detection (numeric, categorical, IDs)
  -> Train/test split (stratified, 80/20)
  -> Preprocessing pipeline (impute + scale + OHE) — fit on TRAIN only
  -> Train: Logistic Regression, Random Forest, XGBoost
  -> Evaluate all on test set
  -> Select best: 50% ROC-AUC + 30% Recall + 20% F1
  -> Save model + pipeline + metadata
  -> K-Means segmentation (Elbow + Silhouette for k)
  -> Save segmentation model
```

---

## Algorithms

| Task | Algorithm | Why |
|---|---|---|
| Default Risk | Logistic Regression | Interpretable baseline |
| Default Risk | Random Forest | High recall, handles non-linearity |
| Default Risk | XGBoost | State-of-the-art gradient boosting |
| Segmentation | K-Means | Effective for financial profile clustering |
| Explainability | SHAP TreeExplainer | Per-instance feature attribution |

---

## Evaluation Metrics

Because this is **credit risk prediction**, accuracy alone is insufficient:

| Metric | Weight in Selection | Reason |
|---|---|---|
| ROC-AUC | 50% | Overall discrimination ability |
| Recall | 30% | Missing a default (false negative) is costly |
| F1-Score | 20% | Balance of precision and recall |
| Accuracy | 0% | Misleading with imbalanced classes |

---

## Cross-Sell Methodology

### Important Limitation

The Credit Risk dataset does **NOT** contain product purchase history or transaction data. Therefore:
- Collaborative filtering cannot be applied
- Market-basket analysis / Apriori rules cannot be computed from real co-occurrence

### What Is Used Instead

**Customer Segmentation + Risk-Aware Business Rules**

1. Customer is assigned to a K-Means segment based on financial profile
2. Risk level and default probability are computed from the ML model
3. Business rules encode domain knowledge:
   - Low-risk, high-income customers → Premium Banking, Investment, Home Loan
   - Medium-risk customers → Credit Card, Auto Loan, Fixed Deposit
   - High-risk customers → Savings Account, Fixed Deposit, Insurance (no additional debt)
4. Scores are computed per-product and top-N returned

### Future Enhancement Path

Once product purchase/transaction data is available, replace the rule layer with:
- Matrix factorization (collaborative filtering)
- Neural collaborative filtering
- Sequence-aware recommendation (RNN/Transformer)

The architecture was designed to allow this drop-in replacement.

---

## Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
# Run the training pipeline (do this once before launching the app)
python src/train_model.py
```

This will:
1. Load or generate the credit risk dataset
2. Train Logistic Regression, Random Forest, and XGBoost
3. Select the best model
4. Save all artifacts to `models/`
5. Train K-Means segmentation
6. Print a training summary with actual metrics

---

## Running the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Screenshots

*(Add screenshots here after running the application)*

**Dashboard:**
![Dashboard](screenshots/dashboard.png)

**Risk Assessment:**
![Risk Assessment](screenshots/risk_assessment.png)

**Cross-Sell Recommendations:**
![Cross-Sell](screenshots/cross_sell.png)

---

## Limitations

| Limitation | Description |
|---|---|
| No real transaction history | Product recommendations use business rules, not learned co-occurrence patterns |
| Risk thresholds are prototypes | LOW <30%, MEDIUM 30-60%, HIGH >60% are NOT regulatory thresholds |
| Synthetic data (if no CSV) | Synthetic data follows realistic distributions but is not real banking data |
| Explainability | SHAP values show correlation patterns, not causal relationships |
| No real-time data | App uses a static trained model; no online learning |

---

## Future Enhancements

- [ ] Real product transaction data integration for collaborative filtering
- [ ] SHAP waterfall plots in the UI
- [ ] What-If simulator (adjust loan terms and see risk change)
- [ ] PDF report export
- [ ] Batch prediction upload (CSV)
- [ ] Model retraining via UI
- [ ] A/B testing framework for recommendation strategies
- [ ] Drift detection for model monitoring

---

## Technical Notes

- Models are trained once and saved as `.pkl` files
- Streamlit loads artifacts with `@st.cache_resource` — no retraining on each page visit
- The preprocessing pipeline is always applied consistently (no data leakage)
- All `print()` calls in training use ASCII to avoid Windows cp1252 encoding issues

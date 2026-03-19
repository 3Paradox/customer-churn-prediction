# 📉 Customer Churn Prediction & Business Intelligence System

> An end-to-end machine learning system that predicts which telecom customers will churn,
> explains *why* using SHAP values, and quantifies the exact revenue at risk —
> deployed as an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Reporting_Layer-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| F1 Score (Churn Class) | **0.85+** |
| PR-AUC | **0.72+** |
| ROC-AUC | **0.84+** |
| Monthly Revenue at Risk Identified | **$139,130** |
| Annual Revenue at Risk | **$1,670,000+** |
| Dataset Size | 7,043 customers · 21 raw features · ~58 engineered features |

**Top 3 Churn Drivers (SHAP):**
1. **Contract Type** — Month-to-month customers churn at 42% vs 3% for 2-year contracts
2. **Tenure** — Customers in their first 12 months are the highest risk group
3. **Monthly Charges** — Churners pay $74/month on average vs $61 for stayers

---

## 🚀 What Makes This Different

Most churn projects stop at a confusion matrix. This one goes further:

**1. SHAP Explainability**
Every prediction comes with a individual-level explanation — not just *who* will churn
but *which features* drove that specific customer's risk score. Built using
`shap.TreeExplainer` with waterfall plots for each customer.

**2. Revenue Impact Calculator**
Predictions are translated into dollars using:
`revenue_at_risk = predicted_churn_probability × MonthlyCharges`
An interactive Streamlit slider lets you model: *"If we retain X% of high-risk customers
with a Y% discount, we save $Z/month."*

**3. RFM Feature Engineering**
Applied the classic marketing segmentation framework (Recency · Frequency · Monetary)
to create 4 composite features on top of the raw data — demonstrating domain knowledge
beyond standard ML pipelines.

**4. SQL Reporting Layer**
All insights are written to a SQLite database with 3 structured tables and reusable
query functions in `src/sql_queries.py` — showing end-to-end data engineering thinking,
not just model building.

**5. Customer Risk Segmentation**
Every customer is classified into High Risk (prob > 0.7), Medium Risk (0.4–0.7),
or Safe (< 0.4) with specific retention actions mapped to each tier.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (scale_pos_weight) |
| Explainability | SHAP (TreeExplainer, waterfall plots) |
| Data Storage | SQLite (sqlite3) |
| Dashboard | Streamlit |
| Serialisation | pickle |

---

## 📁 Project Structure
```
churn_project/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← raw Kaggle download
│   ├── telco_clean.csv                          ← cleaned (Stage 1 output)
│   └── telco_features.csv                       ← engineered features (Stage 2 output)
├── notebooks/
│   ├── 01_eda.ipynb                             ← EDA + data cleaning
│   ├── 02_feature_engineering.ipynb             ← RFM scoring + OHE + SQL layer
│   ├── 03_modeling.ipynb                        ← XGBoost + GridSearchCV + evaluation
│   └── 04_shap_explainability.ipynb             ← SHAP + revenue impact calculator
├── src/
│   └── sql_queries.py                           ← reusable SQL query functions
├── outputs/
│   ├── churn_model.pkl                          ← trained XGBoost model
│   ├── churn_analysis.db                        ← SQLite reporting database
│   ├── shap_summary.png                         ← global SHAP beeswarm plot
│   ├── shap_bar.png                             ← top 15 features bar chart
│   ├── shap_waterfall_high.png                  ← high risk customer explanation
│   ├── shap_waterfall_mid.png                   ← borderline customer explanation
│   └── shap_waterfall_low.png                   ← safe customer explanation
└── app.py                                       ← Streamlit dashboard (3 pages)
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/3Paradox/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install xgboost shap streamlit pandas scikit-learn imbalanced-learn plotly seaborn openpyxl
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Place it in the `data/` folder

### 4. Run the notebooks in order
```bash
jupyter notebook
```
Run these notebooks top to bottom, in order:
```
notebooks/01_eda.ipynb                  ← generates telco_clean.csv
notebooks/02_feature_engineering.ipynb  ← generates telco_features.csv + SQLite DB
notebooks/03_modeling.ipynb             ← generates churn_model.pkl
notebooks/04_shap_explainability.ipynb  ← generates SHAP plots + updates DB
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Key EDA Findings

- **Contract type is the strongest predictor:** Month-to-month customers churn at ~42%,
  one-year at ~11%, and two-year at just ~3% — a 14x difference between extremes.

- **Early tenure is the danger zone:** Customers in their first 12 months have the
  highest churn rate. After 24 months, churn drops sharply — the first year is the
  critical retention window.

- **Churners are high-value customers:** Customers who leave pay an average of $74/month
  vs $61 for those who stay. This makes churn a revenue quality problem, not just a
  volume problem. Monthly revenue lost: ~$139,130 ($1.67M annually).

- **Add-on services are protective:** Customers without OnlineSecurity or TechSupport
  churn at ~41%. With either service, churn drops to ~15% — a 2.7x difference that
  points directly to a bundling retention strategy.

---

## 💼 Skills Demonstrated

This project was built to showcase a complete, production-oriented data science skill set:

**Data Engineering**
- Real-world data quality debugging (TotalCharges dtype bug, tenure=0 edge case)
- Feature engineering from domain knowledge (RFM framework)
- SQLite database design and reusable query functions

**Machine Learning**
- Handling class imbalance with `scale_pos_weight`
- Hyperparameter tuning with `GridSearchCV` (360 model fits, 5-fold CV)
- Correct metric selection: PR-AUC over ROC-AUC for imbalanced datasets
- Model serialisation with `pickle`

**Explainability & Business Translation**
- SHAP TreeExplainer for global and individual-level feature attribution
- Revenue impact quantification: probability × monthly charges = dollars at risk
- Customer segmentation with actionable retention tiers

**Software Engineering**
- Modular code organisation (`src/`, `notebooks/`, `outputs/`)
- Cached data loading in Streamlit (`@st.cache_data`, `@st.cache_resource`)
- Reusable SQL query module with typed function signatures

**Visualisation & Communication**
- 7 publication-quality EDA plots (matplotlib/seaborn)
- Interactive Streamlit dashboard with 3 pages (plotly)
- Interview-ready summary cells with actual numbers baked in

---
## 📸 Screenshots

### Overview — KPI Cards & Risk Segments
![Overview 1](outputs/screenshots/overview_1.png)

### Overview — Tenure Analysis & SHAP Drivers
![Overview 2](outputs/screenshots/overview_2.png)

### Customer Explorer — Risk Badge & Profile
![Explorer 1](outputs/screenshots/explorer_1.png)

### Customer Explorer — Retention Recommendation & SHAP Waterfall
![Explorer 2](outputs/screenshots/explorer_2.png)

### Customer Explorer — SHAP Waterfall (continued)
![Explorer 3](outputs/screenshots/explorer_3.png)

### Revenue Calculator — Sliders & Projected Impact
![Calculator 1](outputs/screenshots/calculator_1.png)

### Revenue Calculator — Top 10 Priority Customers
![Calculator 2](outputs/screenshots/calculator_2.png)

---

## 📌 Data Source

IBM Telco Customer Churn Dataset — available on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
7,043 customers · 21 features · 26.5% churn rate.

---

## 👤 Author

**Tushar Gupta**
Third-year Plastic Technology student at HBTU Kanpur, transitioning into Data Science.

[![GitHub](https://img.shields.io/badge/GitHub-3Paradox-181717?style=flat&logo=github)](https://github.com/3Paradox)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/tushar-gupta-8b1317373)
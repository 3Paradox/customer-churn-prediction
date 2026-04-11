import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_HIGH = '#E8614C'
COLOR_MED  = '#F0A500'
COLOR_SAFE = '#4C9BE8'

@st.cache_data
def load_data():
    df = pd.read_csv('data/telco_features.csv')
    if 'predicted_churn_prob' not in df.columns:
        model = load_model()
        drop_cols = [c for c in ['customerID','Churn','Churn_Binary'] if c in df.columns]
        X = df.drop(columns=drop_cols)
        import json
        with open('outputs/feature_columns.json') as f:
            feature_cols = json.load(f)
        X = X[feature_cols]
        df['predicted_churn_prob'] = model.predict_proba(X)[:, 1]
        df['revenue_at_risk']      = df['predicted_churn_prob'] * df['MonthlyCharges']
        df['segment'] = df['predicted_churn_prob'].apply(
            lambda p: 'High Risk' if p > 0.7 else ('Medium Risk' if p >= 0.4 else 'Safe')
        )
    return df

@st.cache_resource
def load_model():
    with open('outputs/churn_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_clean():

    return pd.read_csv('data/telco_clean.csv')

df       = load_data()

@st.cache_resource
def get_comparison_models(_X_tr, _y_tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=1)
    lr.fit(_X_tr, _y_tr)
    rf.fit(_X_tr, _y_tr)
    return lr, rf
model    = load_model()
df_clean = load_clean()

drop_cols = [c for c in ['customerID','Churn','Churn_Binary',
                          'predicted_churn_prob','revenue_at_risk','segment'] if c in df.columns]
X_all = df.drop(columns=drop_cols)
import json
with open('outputs/feature_columns.json') as f:
    feature_cols = json.load(f)
X_all = X_all[feature_cols]

with st.sidebar:
    st.markdown("## 📉 Churn Intelligence")
    st.markdown("**End-to-End ML Portfolio Project**")
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Features Used",   f"{X_all.shape[1]}")
    st.metric("Churn Rate",      f"{df['Churn_Binary'].mean()*100:.1f}%")
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown("**XGBoost Classifier**")
    st.markdown("Optimised with Optuna · PR-AUC 0.6652")
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/3Paradox/customer-churn-prediction)")
    st.markdown("---")
    st.caption("Built with Python · XGBoost · SHAP · Streamlit")

page = st.radio(
    label="Navigate",
    options=["📊 Overview", "🔍 Customer Explorer", "💰 Revenue Calculator", "📈 Model Performance"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ═══════════════════════════════════════════════
# PAGE 1 - OVERVIEW
# ═══════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📉 Customer Churn Intelligence Dashboard")
    st.markdown("Real-time churn risk analysis · Powered by XGBoost + SHAP")

    total_customers  = len(df)
    churn_rate_pct   = df['Churn_Binary'].mean() * 100
    high_risk_count  = (df['segment'] == 'High Risk').sum()
    monthly_rev_risk = df[df['segment'] == 'High Risk']['revenue_at_risk'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers",         f"{total_customers:,}")
    c2.metric("⚠️ Churn Rate",              f"{churn_rate_pct:.1f}%")
    c3.metric("💰 Monthly Revenue at Risk", f"${monthly_rev_risk:,.0f}")
    c4.metric("🔴 High Risk Customers",     f"{high_risk_count:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Risk Segments")
        seg_counts = df['segment'].value_counts().reindex(['High Risk','Medium Risk','Safe'])
        fig_pie = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            color=seg_counts.index,
            color_discrete_map={
                'High Risk': COLOR_HIGH,
                'Medium Risk': COLOR_MED,
                'Safe': COLOR_SAFE
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, margin=dict(t=20,b=20,l=20,r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Contract Type")
        df_plot = df.copy()
        contract_cols = [c for c in df.columns if c.startswith('Contract_')]

        def get_contract(row):
            for col in contract_cols:
                if row.get(col, 0) == 1:
                    return col.replace('Contract_', '')
            return 'Unknown'

        if contract_cols:
            df_plot['contract_type'] = df_plot[contract_cols].apply(get_contract, axis=1)
        else:
            df_plot['contract_type'] = df_clean['Contract'].values

        contract_churn = (df_plot
            .groupby('contract_type')['Churn_Binary']
            .mean().reset_index()
            .rename(columns={'Churn_Binary':'churn_rate'})
            .sort_values('churn_rate', ascending=False))
        contract_churn['churn_pct'] = contract_churn['churn_rate'] * 100

        fig_c = px.bar(
            contract_churn, x='contract_type', y='churn_pct',
            color='churn_pct',
            color_continuous_scale=['#4C9BE8','#F0A500','#E8614C'],
            text=contract_churn['churn_pct'].apply(lambda x: f'{x:.1f}%'),
            labels={'contract_type':'Contract Type','churn_pct':'Churn Rate (%)'}
        )
        fig_c.update_traces(textposition='outside')
        fig_c.update_layout(coloraxis_showscale=False, margin=dict(t=20,b=20), height=350)
        st.plotly_chart(fig_c, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn Rate by Tenure Group")
        df_t = df.copy()
        df_t['tenure_group'] = pd.cut(
            df_t['tenure'],
            bins=[0,12,24,36,48,72],
            labels=['0-12','13-24','25-36','37-48','48+'],
            right=True
        )
        tenure_churn = (df_t
            .groupby('tenure_group', observed=True)['Churn_Binary']
            .mean().reset_index()
            .rename(columns={'Churn_Binary':'churn_rate'}))
        tenure_churn['churn_pct'] = tenure_churn['churn_rate'] * 100

        fig_t = px.bar(
            tenure_churn, x='tenure_group', y='churn_pct',
            color='churn_pct',
            color_continuous_scale=['#4C9BE8','#F0A500','#E8614C'],
            text=tenure_churn['churn_pct'].apply(lambda x: f'{x:.1f}%'),
            labels={'tenure_group':'Tenure Group (months)','churn_pct':'Churn Rate (%)'}
        )
        fig_t.update_traces(textposition='outside')
        fig_t.update_layout(coloraxis_showscale=False, margin=dict(t=20,b=20), height=350)
        st.plotly_chart(fig_t, use_container_width=True)

    with col4:
        st.subheader("Top Churn Drivers Globally")
        if os.path.exists('outputs/shap_bar.png'):
            st.image('outputs/shap_bar.png', use_container_width=True)
        else:
            st.info("Run Stage 4 notebook to generate SHAP bar chart.")

# ═══════════════════════════════════════════════
# PAGE 2 - CUSTOMER EXPLORER
# ═══════════════════════════════════════════════

elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Explorer")
    st.markdown("Drill into any individual customer — see their risk score, key drivers, and recommended action.")

    if 'customerID' in df.columns:
        customer_ids = df['customerID'].tolist()
    elif 'customerID' in df_clean.columns:
        customer_ids = df_clean['customerID'].tolist()
    else:
        customer_ids = [f"Customer_{i}" for i in range(len(df))]

    selected_id = st.selectbox("Select Customer ID", customer_ids)

    if 'customerID' in df.columns:
        row_idx = df[df['customerID'] == selected_id].index[0]
        pos_idx = df.index.get_loc(row_idx)
    else:
        pos_idx = customer_ids.index(selected_id)
        row_idx = df.index[pos_idx]

    customer_row = df.loc[row_idx]

    if 'customerID' in df_clean.columns:
        clean_row = df_clean[df_clean['customerID'] == selected_id].iloc[0]
    else:
        clean_row = df_clean.iloc[pos_idx]

    churn_prob  = float(customer_row['predicted_churn_prob'])
    rev_at_risk = float(customer_row['revenue_at_risk'])
    segment     = customer_row['segment']

    st.markdown("---")
    badge_col, detail_col = st.columns([1, 2])

    with badge_col:
        color = COLOR_HIGH if segment == 'High Risk' else (COLOR_MED if segment == 'Medium Risk' else COLOR_SAFE)
        emoji = "🔴" if segment == 'High Risk' else ("🟡" if segment == 'Medium Risk' else "🟢")
        st.markdown(
            f"""<div style='background:{color};padding:24px;border-radius:12px;
            text-align:center;color:white;font-size:24px;font-weight:bold;'>
            {emoji} {segment.upper()}<br>
            <span style='font-size:40px'>{churn_prob*100:.1f}%</span><br>
            <span style='font-size:14px'>churn probability</span>
            </div>""", unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("💸 Revenue at Risk", f"${rev_at_risk:.2f}/month")

    with detail_col:
        st.markdown("#### Customer Profile")
        d1, d2, d3 = st.columns(3)
        d1.metric("Tenure",          f"{int(clean_row['tenure'])} months")
        d2.metric("Monthly Charges", f"${clean_row['MonthlyCharges']:.2f}")
        d3.metric("Total Charges",   f"${clean_row['TotalCharges']:.2f}")

        d4, d5, d6 = st.columns(3)
        d4.metric("Contract",         str(clean_row['Contract']))
        d5.metric("Internet Service", str(clean_row['InternetService']))
        d6.metric("RFM Score",        str(int(customer_row.get('rfm_score', 0))))

        st.markdown("#### Churn Probability")
        st.progress(float(churn_prob))
        st.caption(f"{churn_prob*100:.2f}% probability of churning")

    st.markdown("---")
    st.markdown("#### 💡 Retention Recommendation")
    if segment == 'High Risk':
        st.error(f"🚨 **Immediate Action** — Offer 20% discount on annual contract upgrade. "
                 f"Worth ${rev_at_risk*12:,.0f}/year to retain.")
    elif segment == 'Medium Risk':
        st.warning("⚠️ **Monitor & Engage** — Offer TechSupport or OnlineSecurity bundle. "
                   "Add-ons reduce churn from ~41% to ~15%.")
    else:
        st.success("✅ **No Immediate Action** — Customer is stable. Monitor at next billing cycle.")

    st.markdown("---")
    st.markdown("#### 🧠 Why This Prediction? — SHAP Explanation")
    st.caption("Each bar shows how much a feature pushed this prediction toward or away from churn.")

    with st.spinner("Generating SHAP explanation..."):
        try:
            X_customer = X_all.iloc[[pos_idx]].copy()
            X_display  = X_customer.copy()
            X_display.columns = [c.replace('_', ' ') for c in X_customer.columns]
            local_explainer = shap.TreeExplainer(model)
            sv = local_explainer(X_display)
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(sv[0], max_display=12, show=False)
            plt.title(f"SHAP Explanation — {selected_id}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()
        except Exception as e:
            st.error(f"Could not generate SHAP plot: {e}")

# ═══════════════════════════════════════════════
# PAGE 3 - REVENUE CALCULATOR
# ═══════════════════════════════════════════════

elif page == "💰 Revenue Calculator":
    st.title("💰 Revenue Impact Calculator")
    st.markdown("Model the financial return of different retention strategies.")

    high_risk_df         = df[df['segment'] == 'High Risk'].copy()
    high_risk_count      = len(high_risk_df)
    high_risk_avg_charge = high_risk_df['MonthlyCharges'].mean()
    total_rev_at_risk    = high_risk_df['revenue_at_risk'].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 High Risk Customers",   f"{high_risk_count:,}")
    m2.metric("💵 Avg Monthly Charge",    f"${high_risk_avg_charge:.2f}")
    m3.metric("⚠️ Total Revenue at Risk", f"${total_rev_at_risk:,.0f}/mo")

    st.markdown("---")
    st.subheader("🎛️ Retention Scenario Settings")
    s1, s2 = st.columns(2)

    with s1:
        retain_pct = st.slider(
            "What % of High Risk customers can we retain?",
            min_value=0, max_value=100, value=20, step=5, format="%d%%"
        )
    with s2:
        discount_pct = st.slider(
            "Average discount offered to retained customers (%)",
            min_value=0, max_value=50, value=15, step=5, format="%d%%"
        )

    st.markdown("---")

    customers_retained = int(high_risk_count * retain_pct / 100)
    gross_saved        = customers_retained * high_risk_avg_charge
    discount_cost      = gross_saved * (discount_pct / 100)
    net_saved_monthly  = gross_saved - discount_cost
    net_saved_annual   = net_saved_monthly * 12

    st.info(":bulb: **Why contact so many customers?** Optimal threshold = 0.13 because missing a churner costs \$74/month but a false alarm costs only \$10 (7.4:1 ratio). Casting a wider net maximises profit — standard telecom practice.")
    st.subheader("📈 Projected Impact")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("👥 Customers Retained",  f"{customers_retained:,}")
    r2.metric("💰 Gross Revenue Saved", f"${gross_saved:,.0f}/mo")
    r3.metric("🏷️ Net After Discounts", f"${net_saved_monthly:,.0f}/mo",
              delta=f"-${discount_cost:,.0f} discount cost")
    r4.metric("📅 Annual Net Impact",   f"${net_saved_annual:,.0f}")

    st.markdown("---")
    st.subheader("🏆 Top 10 Highest Revenue-at-Risk Customers")
    st.caption("Priority targets for your retention campaign.")

    display_cols = ['predicted_churn_prob','MonthlyCharges','revenue_at_risk','segment','tenure']
    if 'customerID' in df.columns:
        display_cols = ['customerID'] + display_cols

    top10 = (high_risk_df
        .sort_values('revenue_at_risk', ascending=False)
        .head(10)[display_cols].copy())

    top10['predicted_churn_prob'] = top10['predicted_churn_prob'].apply(lambda x: f'{x*100:.1f}%')
    top10['revenue_at_risk']      = top10['revenue_at_risk'].apply(lambda x: f'${x:.2f}')
    top10['MonthlyCharges']       = top10['MonthlyCharges'].apply(lambda x: f'${x:.2f}')
    top10.columns = [c.replace('_',' ').title() for c in top10.columns]

    st.dataframe(top10.reset_index(drop=True), use_container_width=True, height=380)

    st.markdown("---")

# ═══════════════════════════════════════════════
# PAGE 4 - MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    st.markdown("Honest evaluation of the XGBoost churn model across all key metrics.")

    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 confusion_matrix, roc_auc_score, f1_score,
                                 precision_score, recall_score, brier_score_loss)
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt

    df_perf = load_data()
    drop_cols = ['customerID', 'Churn', 'Churn_Binary', 'segment', 'predicted_churn_prob', 'revenue_at_risk']
    X_p = df_perf.drop(columns=[c for c in drop_cols if c in df_perf.columns])
    y_p = df_perf['Churn_Binary']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42, stratify=y_p)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("ROC-AUC",    f"{roc_auc_score(y_te, y_prob):.4f}")
    col2.metric("PR-AUC",     f"{auc(*precision_recall_curve(y_te, y_prob)[1::-1]):.4f}")
    col3.metric("F1 (Churn)", f"{f1_score(y_te, y_pred):.4f}")
    col4.metric("Recall",     f"{recall_score(y_te, y_pred):.4f}")
    col5.metric("Precision", f"{precision_score(y_te, y_pred):.4f}")

    col6.metric("Brier Score",f"{brier_score_loss(y_te, y_prob):.4f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#E05C5C', lw=2, label=f'XGBoost (AUC = {auc(fpr, tpr):.4f})')
        ax.plot([0,1],[0,1],'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, color='#E05C5C', lw=2, label=f'XGBoost (PR-AUC = {auc(rec, prec):.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)
    with c3:
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap='Reds')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['No Churn','Churn'])
        ax.set_yticklabels(['No Churn','Churn'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix', fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        fop, mpv = calibration_curve(y_te, y_prob, n_bins=10)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(mpv, fop, 's-', color='#E05C5C', label='XGBoost')
        ax.plot([0,1],[0,1],'k--', label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve', fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Model Comparison — Why XGBoost?")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=1)
    lr, rf = get_comparison_models(X_tr, y_tr)

    rows = []
    for name, m in [('Logistic Regression', lr), ('Random Forest', rf), ('XGBoost (tuned)', model)]:
        pb = m.predict_proba(X_te)[:, 1]
        pd_ = m.predict(X_te)
        p2, r2, _ = precision_recall_curve(y_te, pb)
        rows.append({
            'Model': name,
            'ROC-AUC': round(roc_auc_score(y_te, pb), 4),
            'PR-AUC':  round(auc(r2, p2), 4),
            'F1 (Churn)': round(f1_score(y_te, pd_), 4),
            'Brier Score': round(brier_score_loss(y_te, pb), 4)
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)
    st.caption("XGBoost wins on PR-AUC and F1 after hyperparameter tuning — the metrics that matter most for imbalanced churn data.")

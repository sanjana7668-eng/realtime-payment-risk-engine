import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Barclays Risk Engine", layout="wide")

st.title("💳 Real-Time Payment Risk & Anomaly Scoring Engine")
st.markdown("Simulated live transaction triage, cost-loss optimization & explainable AI risk scoring.")

@st.cache_resource
def load_assets():
    model = joblib.load('xgb_fraud_model.pkl')
    samples = pd.read_csv('sample_transactions.csv')
    explainer = shap.TreeExplainer(model)
    return model, samples, explainer

model, samples, explainer = load_assets()

# Sidebar Controls
st.sidebar.header("1. Inspector Settings")
options = [f"Transaction #{i}" for i in range(len(samples))]
selected_option = st.sidebar.selectbox("Select Sample Transaction", options, index=5)
tx_index = int(selected_option.split("#")[1])

st.sidebar.divider()
st.sidebar.header("2. Risk Appetite & Loss Matrix")
threshold = st.sidebar.slider("Decision Cutoff Threshold (%)", min_value=1.0, max_value=99.0, value=35.0, step=1.0) / 100.0
friction_cost = st.sidebar.number_input("Compliance Review Cost ($/review)", value=15.0)

sample_tx = samples.iloc[[tx_index]]

# Live Model Scoring
prob = model.predict_proba(sample_tx)[0][1]
risk_score = prob * 100

# Unscale dollar amount estimate for realistic financial metrics
raw_amount = float(np.abs(sample_tx['scaled_amount'].values[0]) * 120 + 25)

# Financial Loss Matrix Calculations
expected_fraud_loss = prob * raw_amount

st.subheader(f"Transaction Reference: TXN-{tx_index + 1000}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Est. Transaction Value", f"${raw_amount:.2f}")
col2.metric("Fraud Risk Score", f"{risk_score:.2f}%")
col3.metric("Expected Fraud Loss", f"${expected_fraud_loss:.2f}")

if prob >= threshold:
    col4.error("🚨 FLAGGED FOR COMPLIANCE REVIEW")
else:
    col4.success("✅ APPROVED")

st.divider()

# Expected Value Analysis Section
st.subheader("📊 Financial Cost-Benefit Triage Analysis")
c1, c2 = st.columns(2)
with c1:
    st.info(f"**Target Threshold Selected:** `{threshold * 100:.1f}%`\n\nIf flagged, holding this transaction costs **${friction_cost:.2f}** in operational review friction but protects **${raw_amount:.2f}** in gross exposure.")
with c2:
    if prob >= threshold:
        st.warning(f"**Action Plan:** Auto-route to Tier-2 Fraud Triage desk. Risk exposure probability exceeds corporate risk appetite cutoff of {threshold * 100:.1f}%.")
    else:
        st.success(f"**Action Plan:** Auto-cleared for immediate settlement. Risk profile sits within normal operating bounds.")

st.divider()

# Explainable AI Plot
st.subheader("Explainable AI (XAI): Feature Drivers for Risk Score")
shap_values = explainer(sample_tx)

fig, ax = plt.subplots(figsize=(8, 4))
shap.plots.waterfall(shap_values[0], max_display=7, show=False)
st.pyplot(fig)

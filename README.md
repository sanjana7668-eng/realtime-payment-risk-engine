# 💳 Real-Time Payment Risk & Anomaly Scoring Engine

An end-to-end quantitative risk engine simulating real-time payment triage, dynamic cost-loss threshold optimization, and explainable AI (XAI) risk scoring for high-throughput payment architectures.

---

## 📌 Architectural Highlights

* **Cost-Sensitive ML Core**: XGBoost classifier trained on 284k+ highly imbalanced transaction records (0.17% baseline fraud rate), optimized for Precision-Recall AUC (PR-AUC 0.83+) via weighted positive log-loss scaling (`scale_pos_weight`).
* **Explainable AI (XAI)**: Integrated SHAP (`TreeExplainer`) waterfall attribution to provide localized, feature-level probability push factors for compliance auditability.
* **Dynamic Loss Matrix Optimization**: Customizable financial loss matrix calculating expected gross loss vs. compliance friction costs ($C_{FP}$ vs. $C_{FN}$) to dynamically drive decision triage cutoffs.
* **Interactive Dashboard**: Streamlit interface deployed for real-time transaction inspection and business risk appetite tuning.

---

## 📊 Financial Cost-Loss Optimization Matrix

Standard classification models optimize for binary accuracy or F1-score, treating all false decisions equally. In institutional quantitative risk engineering, decision cutoffs must minimize expected monetary loss across asymmetric outcomes:

$$\text{Expected Loss} = P(\text{Fraud}) \cdot \text{Transaction Value}$$

$$\text{Total System Cost} = C_{FP} \cdot N_{FP} + \sum_{i \in \text{FN}} \text{Amount}_i$$

* **False Positive Cost ($C_{FP}$)**: Operational review friction cost (e.g., $15.00 manual compliance check).
* **False Negative Cost ($C_{FN}$)**: Full financial exposure lost to undetected fraudulent settlement.

---

## 🚀 Local Deployment

1. **Clone Repository**:
   ```bash
   git clone https://github.com/sanjana7668-eng/realtime-payment-risk-engine.git
   cd realtime-payment-risk-engine
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Streamlit Risk Engine**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, XGBoost
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend/UI**: Streamlit
- **Data Processing**: Pandas, NumPy

---

## 📁 Project Structure

```text
├── app.py                     # Streamlit risk engine application
├── payment_risk_engine.ipynb  # Full model training & XAI pipeline notebook
├── xgb_fraud_model.pkl        # Serialized XGBoost classification model
├── scaler.pkl                 # Scaler artifact for feature normalization
├── sample_transactions.csv    # Sample data for live triage testing
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation
```

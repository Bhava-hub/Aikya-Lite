import os

import numpy as np
import pandas as pd
import shap
import streamlit as st
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from model import FraudNet
from utils import find_optimal_thresholds

st.set_page_config(page_title="Explainable Federated Fraud Detection", layout="wide")
st.title("Explainable Federated Fraud Detection")
st.caption("Federated learning across 4 simulated banks, with SHAP-based explanations — modeled on JPMorgan/BNY's Project Aikya")

@st.cache_resource
def load_model_and_data(model_mtime):
    test_df = pd.read_csv("data/global_test.csv")
    X = test_df.drop("Class", axis=1)
    y = test_df["Class"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = FraudNet(input_dim=X_scaled.shape[1])
    model.load_state_dict(torch.load("federated_model.pt"))
    model.eval()
    return test_df, X, X_scaled, y, model

test_df, X, X_scaled, y, model = load_model_and_data(os.path.getmtime("federated_model.pt"))

def predict_fn(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor).numpy()
    return expit(logits).flatten()

@st.cache_data
def compute_threshold_comparison(_X_scaled, y, amounts, _model):
    x_tensor = torch.tensor(_X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = _model(x_tensor).numpy()
    probs = expit(logits).flatten()
    return find_optimal_thresholds(probs, y, amounts), probs

# ---------- Transaction picker + prediction ----------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Pick a transaction")
    filter_choice = st.radio("Show:", ["All", "Known fraud only", "Known legitimate only"])

    if filter_choice == "Known fraud only":
        indices = np.where(y == 1)[0]
    elif filter_choice == "Known legitimate only":
        indices = np.where(y == 0)[0]
    else:
        indices = np.arange(len(y))

    selected_idx = st.selectbox("Transaction index", indices[:200])

with col2:
    st.subheader("Prediction")
    sample = X_scaled[selected_idx:selected_idx+1]
    prob = predict_fn(sample)[0]
    actual_label = "FRAUD" if y[selected_idx] == 1 else "Legitimate"

    st.metric("Fraud probability", f"{prob:.2%}")
    st.write(f"**Actual label:** {actual_label}")
    st.write(f"**Model flags as (cost-optimal):** {'FRAUD' if prob >= 0.9981 else 'Legitimate'}")

# ---------- SHAP explanation ----------
st.subheader("Why did the model decide this?")

with st.spinner("Computing SHAP explanation..."):
    background = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(sample, nsamples=100)

contributions = list(zip(X.columns.tolist(), np.array(shap_vals).flatten()))
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, val in contributions[:8]:
    direction = "🔴 toward FRAUD" if val > 0 else "🔵 toward NOT FRAUD"
    st.write(f"**{feat}**: {val:+.4f} — {direction}")

st.caption("Note: features (V1–V28) are PCA-anonymized for privacy — see project notes for details.")

# ---------- Threshold strategy comparison ----------
st.divider()
st.subheader("Threshold Strategy: F1-Optimal vs. Cost-Optimal")

amounts = test_df["Amount"].values
comparison, all_probs = compute_threshold_comparison(X_scaled, y, amounts, model)

col_a, col_b = st.columns(2)
for col, name in zip([col_a, col_b], comparison.keys()):
    r = comparison[name]
    with col:
        st.markdown(f"**{name}** (threshold: {r['threshold']:.4f})")
        st.metric("Total cost", f"${r['total_cost']:,.2f}")
        st.write(f"Missed-fraud cost: ${r['fn_cost']:,.2f}")
        st.write(f"False-alarm cost: ${r['fp_cost']:,.2f}")
        st.write(f"Recall: {r['recall']:.2%}")

savings = comparison["F1-optimal"]["total_cost"] - comparison["Cost-optimal"]["total_cost"]
savings_pct = savings / comparison["F1-optimal"]["total_cost"] * 100
st.success(f"Cost-based thresholding saves ${savings:,.2f} ({savings_pct:.1f}%) vs. F1-optimal, on this test set.")

st.caption("Cost model: missed fraud = transaction amount × 3.75 (LexisNexis fraud-cost multiplier); false alarm = $32.50 (industry-average alert investigation cost).")
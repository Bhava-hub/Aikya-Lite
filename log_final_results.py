import torch
import pandas as pd
import numpy as np
import mlflow
from scipy.special import expit
from sklearn.metrics import classification_report
from model import FraudNet
from utils import find_optimal_thresholds, total_cost

MODEL_PATH = "federated_model.pt"
RUN_LABEL = "clean_pos_weight_seeded"  # change this per experiment you want to track

mlflow.set_experiment("aikya-lite-federated-fraud")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
test_df = pd.read_csv("data/global_test.csv")
X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"].values
amounts = test_df["Amount"].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test.values)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

model = FraudNet(input_dim=X_test.shape[1])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    probs = expit(model(X_tensor).numpy()).flatten()

results = find_optimal_thresholds(probs, y_test, amounts)
cost_r = results["Cost-optimal"]
f1_r = results["F1-optimal"]

preds = (probs >= cost_r["threshold"]).astype(int)
report = classification_report(y_test, preds, output_dict=True)

with mlflow.start_run(run_name=RUN_LABEL):
    mlflow.log_param("model_path", MODEL_PATH)
    mlflow.log_metric("cost_optimal_threshold", cost_r["threshold"])
    mlflow.log_metric("cost_optimal_total_cost", cost_r["total_cost"])
    mlflow.log_metric("cost_optimal_recall", cost_r["recall"])
    mlflow.log_metric("f1_optimal_threshold", f1_r["threshold"])
    mlflow.log_metric("f1_optimal_total_cost", f1_r["total_cost"])
    mlflow.log_metric("f1_optimal_recall", f1_r["recall"])
    mlflow.log_metric("precision_at_cost_threshold", report["1"]["precision"])
    mlflow.log_metric("recall_at_cost_threshold", report["1"]["recall"])
    mlflow.log_metric("f1_at_cost_threshold", report["1"]["f1-score"])
    mlflow.log_artifact(MODEL_PATH)

print(f"Logged run '{RUN_LABEL}' to MLflow.")
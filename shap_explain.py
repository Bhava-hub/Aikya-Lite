import torch
import pandas as pd
import numpy as np
import shap
from model import FraudNet

# Load test data
test_df = pd.read_csv("data/global_test.csv")
X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"].values
feature_names = X_test.columns.tolist()

# Scale — same caveat as before: fit fresh here for this PoC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test.values)

# Load the federated model
model = FraudNet(input_dim=X_scaled.shape[1])
model.load_state_dict(torch.load("federated_model.pt"))
model.eval()

# SHAP needs a plain function: input array -> output probabilities
def predict_fn(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs.flatten()

# Background set: a small random sample SHAP uses as a "baseline" reference
background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]

explainer = shap.KernelExplainer(predict_fn, background)

# Pick 3 actual fraud cases to explain
fraud_indices = np.where(y_test == 1)[0][:3]
samples_to_explain = X_scaled[fraud_indices]

print("Computing SHAP values for 3 fraud cases (this may take a minute)...")
shap_values = explainer.shap_values(samples_to_explain, nsamples=100)

# Print feature contributions for each case
for i, idx in enumerate(fraud_indices):
    print(f"\n--- Fraud case #{idx} ---")
    contributions = list(zip(feature_names, shap_values[i].flatten()))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top 5 features driving this prediction:")
    for feat, val in contributions[:5]:
        direction = "→ pushed toward FRAUD" if val > 0 else "→ pushed toward NOT FRAUD"
        print(f"  {feat}: {val:+.4f} {direction}")


shap_values_array = np.array(shap_values)
if shap_values_array.ndim == 3:
    shap_values_array = shap_values_array.squeeze(-1)  # drop the extra output dimension


# Save a summary plot across all 3 cases
shap.summary_plot(shap_values_array, samples_to_explain, feature_names=feature_names, show=False)
import matplotlib.pyplot as plt
plt.savefig("shap_summary.png", bbox_inches="tight")
print("\nSaved shap_summary.png")

# model = the neural network object/architecture created in Python.
# It defines the layers, connections, and how input data flows through the network.
#
# .pt file = a PyTorch file used to save information about a trained model.
# In this case, it contains the model's learned parameters (weights and biases),
# which are the numerical values the model learned during training.
#
# torch.load("federated_model.pt") reads those saved parameters from the file,
# and load_state_dict() puts those parameters into the model's corresponding layers.
#
# In simple terms:
# model = the structure of the neural network
# .pt file = the learned values of that network
# load_state_dict() = puts the learned values back into the network


"""
this dataset's features (V1 through V28) aren't real transaction details like "merchant category" or "location" — they're the output of PCA (Principal Component Analysis), applied by the dataset's original creators specifically to anonymize the real features for privacy. 
So "V14 pushed toward fraud" is true and meaningful mathematically,
but you can't say what V14 actually represents in real-world terms (the original bank data owners never disclosed that, precisely for privacy/security reasons).
 This is actually a great, honest point for your README: "This demonstrates the explainability mechanism works — in a real deployment with real (non-anonymized) features, a compliance officer would see actual meaningful feature names like 'transaction location' or 'merchant category' instead of V14, making the explanation directly actionable."

"""
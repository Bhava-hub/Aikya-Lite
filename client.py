import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import flwr as fl
from model import FraudNet

bank_id = sys.argv[1]  # e.g. "0", "1", "2", "3"

# Load this bank's own data only
df = pd.read_csv(f"data/bank_{bank_id}.csv")
X = df.drop("Class", axis=1).values
y = df["Class"].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = FraudNet(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def get_params():
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_params(params):
    keys = model.state_dict().keys()
    state_dict = dict(zip(keys, [torch.tensor(p) for p in params]))
    model.load_state_dict(state_dict, strict=True)

class FraudClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_params()

    def fit(self, parameters, config):
        set_params(parameters)
        model.train()
        for epoch in range(5):  # local epochs per round
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"[Bank {bank_id}] local loss: {loss.item():.4f}")
        return get_params(), len(X), {}

    def evaluate(self, parameters, config):
        set_params(parameters)
        model.eval()
        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y).item()
        return loss, len(X), {}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FraudClient())



"""
We considered adding Differential Privacy (DP) to our Federated Learning project because, although the banks never share their raw transaction data, the model updates they send could potentially leak some information about their data.

However, implementing DP properly is more complicated than simply adding random noise. 
A proper implementation needs to limit the influence of each individual transaction using per-example gradient clipping 
and use a privacy accountant to calculate a trustworthy privacy value (ε). 
Libraries such as Opacus can help with this,
but integrating them correctly into our Flower-based federated training would require significant additional work and testing.

Since our project is a proof of concept using simulated banks and a public dataset, 
adding full DP was not necessary to demonstrate our main goal: showing that Federated Learning can train a useful fraud detection model without sharing the banks' raw data.

Therefore, we decided to keep DP as future work. We didn't leave it out because DP is unimportant; 
we left it out because implementing it correctly would add considerable complexity and time to the project.

In one sentence

"We considered Differential Privacy, but decided to leave it as future work because implementing it correctly requires additional complexity, 
while our current project already demonstrates the main benefit of Federated Learning: training across banks without sharing their raw data."

"""
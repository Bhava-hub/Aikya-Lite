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
pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum()])  # auto-computed per bank
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#change from tensor to numpy values
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
        #to set parameters given from server to client
        set_params(parameters)
        model.train()
        for epoch in range(10):  # local epochs per round
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            #update weights
            loss.backward()
            optimizer.step()
        print(f"[Bank {bank_id}] local loss: {loss.item():.4f}")
        return get_params(), len(X), {}
# for telling the model just to test and not learn or update any weights
    def evaluate(self, parameters, config):
        set_params(parameters)
        model.eval()
        #ask model not to learn
        with torch.no_grad():
            output = model(X)
            #calculates the loss of global model
            loss = criterion(output, y).item()
        return loss, len(X), {}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FraudClient())
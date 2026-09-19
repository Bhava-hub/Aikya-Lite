import flwr as fl
import numpy as np
from model import FraudNet
import torch

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert Flower's parameter format back to numpy arrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Load into a FraudNet model and save its state_dict
            model = FraudNet(input_dim=30)
            keys = model.state_dict().keys()
            state_dict = dict(zip(keys, [torch.tensor(p) for p in aggregated_ndarrays]))
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), "federated_model.pt")
            print(f"[Server] Saved model after round {server_round}")

        return aggregated_parameters, metrics

strategy = SaveModelStrategy(
    min_fit_clients=4,
    min_available_clients=4,
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)
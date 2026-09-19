import torch
import torch.nn as nn

class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        #constructing a neural network with  32 neurons and feeding our features to it
        self.fc1 = nn.Linear(input_dim, 32)
        #converting it into 16 neurons
        self.fc2 = nn.Linear(32, 16)
        #converting to 1 neuron
        self.fc3 = nn.Linear(16, 1)
        #relu for converting -ve values to 0
        self.relu = nn.ReLU()
        #compressing values from 0 to 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
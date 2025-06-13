import torch
import torch.nn as nn


class RepPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_layers=2):
        super().__init__()

        layers = []
        # Input layer
        layers.extend([nn.Linear(obs_dim, hidden), nn.ReLU()])

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])

        # Output layer - changed to 2 dimensions for binary classification
        layers.append(nn.Linear(hidden, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Use softmax for probability distribution over 2 classes
        return torch.softmax(self.net(x), dim=-1)

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

        # Output layer
        layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class PPOCompatiblePolicy(nn.Module):
    """
    Policy network that matches the policy part of stable-baselines3 PPO MlpPolicy.
    This allows direct parameter loading from supervised training to PPO.
    Only includes the policy network components - value network is PPO-specific.
    """

    def __init__(self, obs_dim, hidden=128, n_layers=2, n_actions=2):
        super().__init__()

        policy_layers = []
        policy_layers.extend([nn.Linear(obs_dim, hidden), nn.ReLU()])
        for _ in range(n_layers - 1):
            policy_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])

        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Sequential(*policy_layers)

        self.action_net = nn.Linear(hidden, n_actions)

    def forward(self, x):
        policy_features = self.mlp_extractor.policy_net(x)

        action_logits = self.action_net(policy_features)

        return torch.softmax(action_logits, dim=-1)

    def get_action_logits(self, x):
        """Get raw action logits (useful for some loss functions)"""
        policy_features = self.mlp_extractor.policy_net(x)
        return self.action_net(policy_features)

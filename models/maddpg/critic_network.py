from torch import nn, Tensor
import torch


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.nonlin = nn.ReLU()

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        X = torch.cat([states, actions], dim=1)
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out

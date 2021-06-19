import torch
from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, norm_in: bool = True):
        super().__init__()
        if norm_in:
            self.in_fn = nn.BatchNorm1d(state_dim + action_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.nonlin = nn.ReLU()

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        h1 = self.nonlin(self.fc1(torch.cat([states, actions], dim=1)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out

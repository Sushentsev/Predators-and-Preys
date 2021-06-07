import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 64,
                 constrain_out: bool = False, norm_in: bool = True):
        super().__init__()
        self.input_dim = input_dim

        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nn.ReLU()

        if constrain_out:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = nn.Tanh()
        else:
            self.out_fn = lambda x: x

    def forward(self, X: Tensor) -> Tensor:
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

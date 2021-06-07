from torch import nn, Tensor


class ActorNetwork(nn.Module):
    """
    Actor which defines policy. Policy is a function from state to action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # self.in_fn = nn.BatchNorm1d(state_dim)
        # self.in_fn.weight.data.fill_(1)
        # self.in_fn.bias.data.fill_(0)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.nonlin = nn.ReLU()

        self.fc3.weight.data.uniform_(-3e-3, 3e-3)  # Prevent saturation
        self.out_fn = nn.Tanh()
        # self.out_fn = lambda x: x

    def forward(self, states: Tensor) -> Tensor:
        """
        :param states: shape (batch_size, state_dim)
        :return: shape (batch_size, action_dim)
        """
        h1 = self.nonlin(self.fc1(states))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

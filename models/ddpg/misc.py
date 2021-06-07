from torch.nn import Module
import numpy as np

class RandomNoise:
    def __init__(self, mu: float = 0.0, theta: float = 0.1, sigma: float = 0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = 1
        self.low = -1.0
        self.high = 1.0
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


def soft_update(target: Module, source: Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: Module, source: Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

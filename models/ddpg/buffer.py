from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class Buffer(object):
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        self.max_size = max_size

        self.state_buff = torch.zeros(max_size, state_dim)
        self.action_buff = torch.zeros(max_size, action_dim)
        self.reward_buff = torch.zeros(max_size)
        self.next_state_buff = torch.zeros(max_size, state_dim)
        self.done_buff = torch.zeros(max_size)

        self.filled_i = 0
        self.curr_size = 0

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        self.state_buff[self.filled_i] = state
        self.action_buff[self.filled_i] = action
        self.reward_buff[self.filled_i] = reward
        self.next_state_buff[self.filled_i] = next_state
        self.done_buff[self.filled_i] = done

        self.curr_size = min(self.max_size, self.curr_size + 1)
        self.filled_i = (self.filled_i + 1) % self.max_size

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = np.random.choice(self.curr_size, batch_size, replace=False)
        indices = torch.Tensor(indices)

        return self.state_buff[indices], self.action_buff[indices], \
               self.reward_buff[indices], self.next_state_buff[indices], self.done_buff[indices]

    def __len__(self):
        return self.curr_size

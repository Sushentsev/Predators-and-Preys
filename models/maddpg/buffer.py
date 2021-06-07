from collections import deque
from random import randint
import numpy as np

from models.maddpg.utils import convert_state


class ReplayBuffer:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)

    def add(self, state_dict, action, reward, next_state_dict, done):
        transition = (convert_state(state_dict), action, reward, convert_state(next_state_dict), done)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for _ in range(batch_size):
            index = randint(0, len(self.buffer) - 1)
            state, action, reward, next_state, done = self.buffer[index]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

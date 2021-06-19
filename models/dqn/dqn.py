import copy
import random
import numpy as np
from collections import deque, namedtuple
from torch.nn import functional as F

import torch
from torch import nn
from torch.optim import Adam

from models.ddpg.buffer import Buffer

BUFFER_SIZE = 100_000
GAMMA = 0.99
TRANSITIONS = 1_000_000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 512
LEARNING_RATE = 5e-4


class DQN:
    def __init__(self, state_dim: int, action_dim: int):
        self.steps = 0  # Do not change
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.replay_buffer = Buffer(state_dim, action_dim, BUFFER_SIZE)
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=action_dim)
        )

        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    def consume_transition(self, transition):
        state, action, next_state, reward, done = transition
        self.replay_buffer.push(state, [action], reward, next_state, done)


    def sample_batch(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        return states, actions, next_states, rewards, dones

    def train_step(self, batch):
        states, actions, next_states, rewards, dones = batch
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_actions = self.policy_net(next_states).argmax(1, keepdim=True).long().detach()
        next_state_values = self.target_net(next_states).gather(1, next_state_actions).detach()
        expected_state_action_values = (GAMMA * next_state_values * (1 - dones)) + rewards
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.from_numpy(state).float()
        model = self.target_net if target else self.policy_net

        model.eval()
        with torch.no_grad():
            Q_value_actions = model(state)
        model.train()

        action = Q_value_actions.cpu().data.numpy().argmax()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.policy_net, "./submission/prey.pkl")

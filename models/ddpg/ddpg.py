from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from torch.optim import Adam

from models.ddpg.actor import Actor
from models.ddpg.buffer import Buffer
from models.ddpg.critic import Critic
from models.ddpg.misc import hard_update, soft_update


class DDPG:
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, buffer_size: int = 500_000, batch_size: int = 512,
                 actor_lr: float = 2e-4, critic_lr: float = 5e-4,
                 gamma: float = 0.99, tau: float = 0.002):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.buffer = Buffer(state_dim, action_dim, buffer_size)

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)

        self.actor_target = Actor(state_dim, action_dim)
        hard_update(self.actor_target, self.actor)

        self.critic_target = Critic(state_dim, action_dim)
        hard_update(self.critic_target, self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_criterion = torch.nn.MSELoss()

    def critic_update(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            next_actions = self.actor(next_states)

        critic_target_out = self.critic_target(next_states, next_actions)
        target_reward = rewards.view(-1, 1) + self.gamma * (1 - dones.view(-1, 1)) * critic_target_out
        critic_loss = self.critic_criterion(self.critic(states, actions), target_reward.detach())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.7)
        self.critic_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

    def actor_update(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        states, actions, rewards, next_states, dones = batch

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.7)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self.critic_update(batch)
        self.actor_update(batch)

    def act(self, state):
        self.actor.eval()
        with torch.no_grad():
            action = np.clip(self.actor(state).squeeze(), -1, 1)
        self.actor.train()
        return action.numpy()

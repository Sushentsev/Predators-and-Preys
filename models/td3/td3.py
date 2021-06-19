from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from models.td3.actor import Actor
from models.td3.buffer import Buffer
from models.td3.critic import Critic
from models.td3.misc import hard_update, soft_update


class TD3:
    def __init__(self, state_dim: int, action_dim: int, device, hidden_dim: int = 64,
                 replay_size: int = 1_000_000, gamma: float = 0.99, polyak: float = 0.995,
                 pi_lr: float = 0.001, q_lr: float = 0.001, batch_size: int = 1024,
                 update_after: int = 10_000, update_every: int = 50,
                 act_noise: float = 0.1, target_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2):
        # Params inits.
        self.gamma = gamma
        self.polyak = polyak

        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.ints_last = 0
        self.num_updates = 50

        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        self.low = -1.0
        self.high = 1.0

        self.device = device

        # NN inits.
        self.replay_buffer = Buffer(state_dim, action_dim, replay_size)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.actor_target, self.actor)

        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.actor_optim = Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=q_lr)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=q_lr)
        self.critic_loss_fn = torch.nn.MSELoss()

    def _critic_update(self, batch: Sequence[Tensor]):
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_actions = self.actor_target(next_states)

        noise = torch.clamp(self.target_noise * torch.randn(next_actions.shape).to(self.device),
                            min=-self.noise_clip, max=self.noise_clip)

        next_actions = torch.clamp(next_actions + noise, min=self.low, max=self.high)
        y = rewards + self.gamma * (1 - dones) * torch.min(self.critic_1_target(next_states, next_actions),
                                                           self.critic_2_target(next_states, next_actions))

        critic_1_loss = self.critic_loss_fn(self.critic_1(states, actions), y.detach())
        critic_2_loss = self.critic_loss_fn(self.critic_2(states, actions), y.detach())

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()

        self.critic_1_optim.step()
        self.critic_2_optim.step()

    def _actor_update(self, batch: Sequence[Tensor]):
        states, actions, rewards, next_states, dones = batch

        actor_loss = -self.critic_1(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def _batch_tensors(self, batch: Sequence[np.ndarray]) -> Sequence[Tensor]:
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).view(self.batch_size, -1).to(self.device)
        return states, actions, rewards, next_states, dones

    def update(self):
        if len(self.replay_buffer) <= self.update_after:
            return

        if self.ints_last >= self.update_every:
            self.ints_last = 0
            for j in range(self.num_updates):
                batch = self.replay_buffer.sample(self.batch_size)
                batch = self._batch_tensors(batch)

                self._critic_update(batch)

                if j % self.policy_delay == 0:
                    self._actor_update(batch)

                    soft_update(self.actor_target, self.actor, 1 - self.polyak)
                    soft_update(self.critic_1_target, self.critic_1, 1 - self.polyak)
                    soft_update(self.critic_2_target, self.critic_2, 1 - self.polyak)

    def act(self, state: np.ndarray, train: bool = True) -> np.ndarray:
        self.actor.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).view(1, -1).to(self.device)
            action = self.actor(state_tensor)

        self.actor.train()

        if train:
            noise = self.act_noise * torch.randn(action.shape).to(self.device)
            action = torch.clamp(action + noise, min=self.low, max=self.high)

        return action.squeeze().cpu().detach().numpy()

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.ints_last += 1
        self.replay_buffer.push(state, action, reward, next_state, done)

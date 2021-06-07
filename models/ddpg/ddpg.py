import torch
import numpy as np
from torch.optim import Adam

from models.ddpg.actor import Actor
from models.ddpg.buffer import Buffer
from models.ddpg.critic import Critic
from models.ddpg.misc import RandomNoise, hard_update, soft_update

BUFFER_SIZE = 100_000
ACTOR_LR = 2e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.002
BATCH_SIZE = 256


class DDPG:
    def __init__(self, state_dim: int, action_dim: int, eps: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.buffer = Buffer(BUFFER_SIZE, self.device)
        self.noise = RandomNoise()

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        hard_update(self.actor_target, self.actor)

        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.critic_criterion = torch.nn.MSELoss()

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        st, acts, rews, n_sts, dones = self.buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            na = self.actor(n_sts)

        temp_1 = self.critic(st, acts.view(-1, 1))
        temp_2 = self.critic_target(n_sts, na)
        temp_3 = rews.view(-1, 1) + GAMMA * (1 - dones.view(-1, 1)) * temp_2.detach()

        critic_loss = self.critic_criterion(temp_1, temp_3)
        actor_loss = -self.critic(st, self.actor(st)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()

        self.critic_optim.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.7)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.7)

        self.actor_optim.step()
        self.critic_optim.step()

        soft_update(self.actor_target, self.actor, TAU)
        soft_update(self.critic_target, self.critic, TAU)

    def act(self, state):
        with torch.no_grad():
            action = np.clip(self.actor(state).view(-1) + self.eps * self.noise.noise(), -1, 1)
        return action.numpy()

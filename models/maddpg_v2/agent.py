import torch
from torch import Tensor
from torch.optim import Adam

from models.maddpg_v2.misc import hard_update
from models.maddpg_v2.network import MLPNetwork
from models.maddpg_v2.noise import OUNoise


class DDPGAgent(object):
    def __init__(self, num_in_actor: int, num_out_actor: int, num_in_critic: int,
                 hidden_dim: int = 64, actor_lr: float = 0.01, critic_lr: float = 0.01,
                 device=torch.device("cpu")):
        self.device = device
        self.actor = MLPNetwork(num_in_actor, num_out_actor, hidden_dim=hidden_dim, constrain_out=True).to(device)
        self.critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, constrain_out=False).to(device)
        self.target_actor = MLPNetwork(num_in_actor, num_out_actor, hidden_dim=hidden_dim, constrain_out=True).to(device)
        self.target_critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, constrain_out=False).to(device)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.exploration = OUNoise(num_out_actor)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale: float):
        self.exploration.scale = scale

    def step(self, obs: Tensor, explore: bool = False):
        if obs.ndim == 1:
            obs = obs.view(1, -1)

        action = self.actor(obs)
        if explore:
            action += Tensor(self.exploration.noise()).to(self.device)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        self.target_actor.load_state_dict(params["target_actor"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.actor_optimizer.load_state_dict(params["actor_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
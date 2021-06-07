from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor


class ReplayBuffer(object):
    def __init__(self, max_size: int, num_agents: int, obs_dims: List[int], ac_dims: List[int]):
        self.max_size = max_size
        self.num_agents = num_agents

        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []

        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_size, odim)))
            self.ac_buffs.append(np.zeros((max_size, adim)))
            self.rew_buffs.append(np.zeros(max_size))
            self.next_obs_buffs.append(np.zeros((max_size, odim)))
            self.done_buffs.append(np.zeros(max_size))

        self.current_size = 0
        self.filled_i = 0

    def push(self, obs: List[np.ndarray], acs: List[np.ndarray], rews: np.ndarray,
             next_obs: List[np.ndarray], dones: List[np.ndarray]):
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.filled_i] = obs[agent_i]
            self.ac_buffs[agent_i][self.filled_i] = acs[agent_i]
            self.rew_buffs[agent_i][self.filled_i] = rews[agent_i]
            self.next_obs_buffs[agent_i][self.filled_i] = next_obs[agent_i]
            self.done_buffs[agent_i][self.filled_i] = dones[agent_i]

        self.filled_i = (self.filled_i + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int, device, norm_rews: bool = True) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor]]:
        indices = np.random.choice(np.arange(max(self.current_size, self.filled_i)),
                                   size=batch_size, replace=False)

        cast = lambda x: torch.Tensor(x).to(device)

        ret_obs = [cast(self.obs_buffs[agent_i][indices]) for agent_i in range(self.num_agents)]
        ret_acs = [cast(self.ac_buffs[agent_i][indices]) for agent_i in range(self.num_agents)]

        if norm_rews:
            last_index = max(self.filled_i, self.current_size)
            ret_rews = [cast((self.rew_buffs[agent_i][indices] - self.rew_buffs[agent_i][:last_index].mean())
                             / self.rew_buffs[agent_i][:last_index].std()
                             ) for agent_i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[agent_i][indices]) for agent_i in range(self.num_agents)]

        ret_n_obs = [cast(self.next_obs_buffs[agent_i][indices]) for agent_i in range(self.num_agents)]
        ret_dns = [cast(self.done_buffs[agent_i][indices]) for agent_i in range(self.num_agents)]

        return ret_obs, ret_acs, ret_rews, ret_n_obs, ret_dns

    def __len__(self):
        return self.current_size

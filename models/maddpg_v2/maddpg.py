from typing import Dict, List

import torch
from gym.spaces import Box, Discrete
from torch import Tensor
from torch.nn import Module

from .misc import soft_update
from .agent import DDPGAgent

MSELoss = torch.nn.MSELoss()


class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params: List[Dict[str, int]],
                 gamma: float = 0.95, tau: float = 0.01,
                 actor_lr: float = 0.01, critic_lr: float = 0.01,
                 hidden_dim: int = 64, device=torch.device("cpu")):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_actor (int): Input dimensions to policy
                num_out_actor (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.init_dict = dict()
        self.num_agents = len(agent_init_params)
        self.agents = [DDPGAgent(actor_lr=actor_lr,
                                 critic_lr=critic_lr,
                                 hidden_dim=hidden_dim, **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.niter = 0

    @property
    def actors(self) -> List[Module]:
        return [a.actor for a in self.agents]

    @property
    def target_actors(self) -> List[Module]:
        return [a.target_actor for a in self.agents]

    def scale_noise(self, scale: float):
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations: List[Tensor], explore: bool = False):
        return [agent.step(obs, explore=explore) for agent, obs in zip(self.agents, observations)]

    def update(self, sample, agent_i):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # Critic update
        curr_agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            all_target_actions = [target_actor(nobs)
                                  for target_actor, nobs in zip(self.target_actors, next_obs)]
        target_vf_in = torch.cat((*next_obs, *all_target_actions), dim=1)
        target_value = rews[agent_i].view(-1, 1) + self.gamma * curr_agent.target_critic(target_vf_in) * (
                1 - dones[agent_i].view(-1, 1))
        vf = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # Actor update
        curr_agent.actor_optimizer.zero_grad()
        curr_actor_out = curr_agent.actor(obs[agent_i])
        curr_pol_vf_in = curr_actor_out
        all_actor_actions = []
        for i, actor, ob in zip(range(self.num_agents), self.actors, obs):
            if i == agent_i:
                all_actor_actions.append(curr_pol_vf_in)
            else:
                all_actor_actions.append(acs[i])
        vf = torch.cat((*obs, *all_actor_actions), dim=1)
        actor_loss = -curr_agent.critic(vf).mean()
        # actor_loss += (curr_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.actor_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_actor, a.actor, self.tau)
        self.niter += 1

    def prep_training(self):
        for a in self.agents:
            a.actor.train()
            a.critic.train()
            a.target_actor.train()
            a.target_critic.train()

    def prep_rollouts(self):
        for a in self.agents:
            a.actor.eval()

    def save(self, filename: str):
        self.prep_training()
        save_dict = {"init_dict": self.init_dict,
                     "agent_params": [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, gamma: float = 0.95, tau: float = 0.01,
                      actor_lr: float = 0.01, critic_lr: float = 0.01,
                      hidden_dim: int = 64):
        num_preds = env.game.num_preds
        num_preys = env.game.num_preys
        num_obsts = env.game.num_obsts
        num_agents = num_preds + num_preys

        num_in_actor = (num_agents - 1) * 3 + num_obsts * 3 + 1
        num_out_actor = 1
        num_in_critic = num_in_actor * num_agents + num_agents * 1

        agent_init_params = [
            {"num_in_actor": num_in_actor, "num_out_actor": num_out_actor, "num_in_critic": num_in_critic}
            for _ in range(num_agents)]

        init_dict = {
            "gamma": gamma, "tau": tau, "actor_lr": actor_lr, "critic_lr": critic_lr,
            "hidden_dim": hidden_dim, "device": device, "agent_init_params": agent_init_params
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename: str):
        save_dict = torch.load(filename)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        return instance

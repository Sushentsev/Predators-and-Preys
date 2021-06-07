import numpy as np
import random
import torch
import json
from torch.optim import Adam
from torch import nn

from models.maddpg.actor_network import ActorNetwork
from models.maddpg.critic_network import CriticNetwork
from models.maddpg.buffer import ReplayBuffer
from models.maddpg.rewards import predator_reward, prey_reward
from models.maddpg.utils import convert_state
from predators_and_preys_env.env import PredatorsAndPreysEnv

SEED = 42
ACTOR_LR = 0.01
CRITIC_LR = 0.01
TAU = 0.01
GAMMA = 0.95
BUFFER_SIZE = 100_000
UPDATE_EVERY = 100
BATCH_SIZE = 256
EPS = 1.0
TRANSITIONS = 1_000_000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG = json.load(open("../maddpg_v2/config.json"))


def set_seed(env):
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


def soft_update(tau, target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


class MADDPG:
    def __init__(self, num_preds: int, num_preys: int, actor_state_dim: int,
                 critic_state_dim: int, critic_action_dim: int):
        self.num_preds = num_preds
        self.num_preys = num_preys
        self.num_agents = self.num_preds + self.num_preys
        self.low, self.high = -1.0, 1.0
        self.buffer = ReplayBuffer(maxsize=BUFFER_SIZE)

        self.actors = [ActorNetwork(actor_state_dim, 1).to(DEVICE) for _ in range(self.num_agents)]
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim).to(DEVICE) for _ in range(self.num_agents)]

        self.actors_target = []
        for actor in self.actors:
            actor_target = ActorNetwork(actor_state_dim, 1).to(DEVICE)
            actor_target.load_state_dict(actor.state_dict())
            self.actors_target.append(actor_target)

        self.critics_target = []
        for critic in self.critics:
            critic_target = CriticNetwork(critic_state_dim, critic_action_dim).to(DEVICE)
            critic_target.load_state_dict(critic.state_dict())
            self.critics_target.append(critic_target)

        self.actor_optims = [Adam(actor.parameters(), lr=ACTOR_LR) for actor in self.actors]
        self.critic_optims = [Adam(critic.parameters(), lr=CRITIC_LR) for critic in self.critics]
        self.critic_losses = [nn.MSELoss() for _ in range(self.num_agents)]

    def update_critic(self, current_agent_id, states, actions, rewards, next_states, dones):
        # Evaluating critic loss
        next_actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                next_actions.append(self.actors_target[agent_id](next_states))

        next_actions = torch.cat(next_actions, dim=1)
        y = rewards + GAMMA * (1 - dones) * self.critics_target[current_agent_id](next_states, next_actions).detach()
        q_estimation = self.critics[current_agent_id](states, actions)
        critic_loss = self.critic_losses[current_agent_id](y, q_estimation)

        # Update critic
        self.critic_optims[current_agent_id].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critics[current_agent_id].parameters(), 0.7)
        self.critic_optims[current_agent_id].step()

    def update_actor(self, current_agent_id, states, actions, rewards, next_states, dones):
        # curr_actions = [self.actors[agent_id](states) for agent_id in range(self.num_agents)]
        curr_actions = []

        for i in range(self.num_agents):
            if i == current_agent_id:
                curr_actions.append(self.actors[current_agent_id](states))
            else:
                curr_actions.append(actions[:, i].view(-1, 1))

        curr_actions = torch.cat(curr_actions, dim=1)

        actor_loss = -self.critics[current_agent_id](states, curr_actions).mean()

        self.actor_optims[current_agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actors[current_agent_id].parameters(), 0.7)
        self.actor_optims[current_agent_id].step()

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        for i in range(self.num_agents):
            states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
            states = torch.FloatTensor(states).to(DEVICE)
            actions = torch.FloatTensor(actions).to(DEVICE)
            rewards = torch.FloatTensor(rewards[:, i]).view(BATCH_SIZE, 1).to(DEVICE)
            next_states = torch.FloatTensor(next_states).to(DEVICE)
            dones = torch.FloatTensor(dones).view(BATCH_SIZE, 1).to(DEVICE)

            self.update_critic(i, states, actions, rewards, next_states, dones)
            self.update_actor(i, states, actions, rewards, next_states, dones)

        for i in range(self.num_agents):
            soft_update(TAU, self.critics_target[i], self.critics[i])
            soft_update(TAU, self.actors_target[i], self.actors[i])


    def act(self, state):
        actions = []
        state = torch.FloatTensor(state).view(1, -1)

        with torch.no_grad():
            for agent_id in range(self.num_agents):
                actor = self.actors[agent_id]
                actor.eval()
                action = actor(state).squeeze(0).detach().cpu().numpy()
                action = action + EPS * np.random.normal(0, 0.01, len(action))
                action = np.clip(action, -1, 1)
                actions.append(action[0])
                actor.train()

        return np.array(actions)

    def save(self):
        for i, pred_actor in enumerate(self.actors[:self.num_preds]):
            torch.save(pred_actor.state_dict(), f"./solutions/pred{i}.pkl")

        for i, prey_actor in enumerate(self.actors[self.num_preds:]):
            torch.save(prey_actor.state_dict(), f"./solutions/prey{i}.pkl")


def main():
    env = PredatorsAndPreysEnv(CONFIG, render=False)
    set_seed(env)

    game = CONFIG["game"]
    num_preds = game["num_preds"]
    num_preys = game["num_preys"]
    num_obsts = game["num_obsts"]
    num_agents = num_preds + num_preys
    actor_state_dim = critic_state_dim = num_preds * 4 + num_preys * 5 + num_obsts * 3
    critic_action_dim = num_agents * 1
    maddpg = MADDPG(num_preds, num_preys, actor_state_dim, critic_state_dim, critic_action_dim)

    state, done = env.reset(), False
    current_preds_rewards, current_preys_rewards = [], []

    for i in range(TRANSITIONS):
        action = maddpg.act(convert_state(state))
        next_state, rewards, done = env.step(action[:num_preds], action[num_preds:])
        rewards = np.concatenate((rewards["predators"], rewards["preys"]))

        ##
        current_preds_rewards.append(np.mean(rewards[:num_preds]))
        current_preys_rewards.append(np.mean(rewards[num_preds:]))
        ##

        maddpg.buffer.add(state, action, rewards, next_state, done)
        maddpg.update()

        state = next_state if not done else env.reset()
        if i % 500 == 0:
            print(f"Current transition: {i}")
            print(f"Predators mean reward: {np.mean(current_preds_rewards)}")
            print(f"Preys mean reward: {np.mean(current_preys_rewards)}")
            print(f"Current action: {action}")
            print()

            current_preds_rewards = []
            current_preys_rewards = []

            maddpg.save()


if __name__ == '__main__':
    main()

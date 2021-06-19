import argparse
import json
import numpy as np
import torch
import random

import sys

sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")

from models.ddpg.converts import state_to_prey_obs, state_to_pred_obs
from models.ddpg.rewards import pred_reward
from models.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent
from models.ddpg.ddpg import DDPG
from models.imitation_learning.predator import PredatorAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


def set_seed(env, seed: int):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train(train_config):
    print(f"Running train with params: {train_config}")
    import time
    start_time = time.time()

    env = PredatorsAndPreysEnv(json.load(open("config_predator.json")), render=False)
    prey = FleeingPreyAgent()
    set_seed(env, train_config.seed)

    ddpg = DDPG(state_dim=3 + 5 * 3 + 5 * 3, action_dim=1)

    done = True
    total_rewards = 0.
    for step in range(train_config.transitions):
        if done:
            state_dict = env.reset()
            done = False

        sigma = train_config.max_sigma - \
                (train_config.max_sigma - train_config.min_sigma) * step / train_config.transitions

        action = ddpg.act(torch.FloatTensor(state_to_pred_obs(state_dict)).view(1, -1))
        action += np.random.normal(scale=sigma)
        action = np.array(np.clip(action, -1, 1))
        next_state_dict, _, done = env.step(action, prey.act(state_dict))

        reward = pred_reward(next_state_dict["predators"][0], next_state_dict)
        total_rewards += reward

        ddpg.buffer.push(state_to_pred_obs(state_dict), action, reward, state_to_pred_obs(next_state_dict), done)
        state_dict = next_state_dict
        ddpg.update()

        if step % train_config.save_every == 0:
            print(f"Transitions: {step}")
            print(f"Mean reward: {round(total_rewards / train_config.save_every, 4)}")
            print(f"Elapsed time: {round((time.time() - start_time) / 60, 1)} minutes")
            print()

            total_rewards = 0.
            torch.save(ddpg.actor.state_dict(), "./solutions/predator.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--transitions", default=2_000_000, type=int)
    parser.add_argument("--max_sigma", default=0.3, type=float)
    parser.add_argument("--min_sigma", default=0.05, type=float)
    parser.add_argument("--save_every", default=10_000, type=int)
    config = parser.parse_args()

    train(config)

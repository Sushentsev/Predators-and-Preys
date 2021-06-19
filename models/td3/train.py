import argparse
import json
import numpy as np
import torch
import random
import time

import sys
# sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")


from models.imitation_learning.predator import PredatorAgent
from models.imitation_learning.prey import PreyAgent
from models.td3.converts import state_to_prey_obs
from models.td3.rewards import prey_reward
from models.td3.td3 import TD3
from predators_and_preys_env.env import PredatorsAndPreysEnv


def set_seed(env, seed: int):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train(train_config):
    print(f"Running train TD3 with params: {train_config}")
    start_time = time.time()

    env = PredatorsAndPreysEnv(json.load(open("config.json")), render=False)
    predator = PredatorAgent()
    set_seed(env, train_config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td3 = TD3(state_dim=3 + 2 * 2 + 5 * 3, action_dim=1, device=device)


    done = True
    total_rewards = 0.
    for step in range(train_config.transitions):
        if done:
            state_dict = env.reset()
            done = False

        prey_action = td3.act(state_to_prey_obs(state_dict), train=True)
        next_state_dict, _, done = env.step(predator.act(state_dict), prey_action)
        reward = prey_reward(next_state_dict["preys"][0], next_state_dict)
        total_rewards += reward

        td3.push(state_to_prey_obs(state_dict), prey_action, reward, state_to_prey_obs(next_state_dict), done)
        state_dict = next_state_dict
        td3.update()

        if step % train_config.save_every == 0:
            print(f"Transitions: {step + 1}")
            print(f"Mean reward: {round(total_rewards / train_config.save_every, 4)}")
            print(f"Elapsed time: {round((time.time() - start_time) / 60, 1)} minutes")
            print(f"Buffer size: {len(td3.replay_buffer)}")
            print()

            total_rewards = 0.
            torch.save(td3.actor.cpu().state_dict(), "./solutions/prey.pkl")
            td3.actor.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--transitions", default=10_000_000, type=int)
    parser.add_argument("--save_every", default=10_000, type=int)
    config = parser.parse_args()

    train(config)

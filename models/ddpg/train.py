import argparse
import json
import numpy as np
import torch
import random

import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")

from models.ddpg.rewards import prey_reward



from models.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent
from models.ddpg.ddpg import DDPG
from predators_and_preys_env.env import PredatorsAndPreysEnv


EPISODES = 20_000


def set_seed(env, seed: int = 42):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def observations(state_dict):
    agent = state_dict["preys"][0]
    obs = [agent["speed"], agent["radius"]]
    enemy = state_dict["predators"][0]

    dx = agent["x_pos"] - enemy["x_pos"]
    dy = agent["y_pos"] - enemy["y_pos"]
    obs.extend([dx, dy, enemy["speed"], enemy["radius"]])

    for obstacle in state_dict["obstacles"]:
        obs.extend([agent["x_pos"] - obstacle["x_pos"],
                    agent["y_pos"] - obstacle["y_pos"],
                    obstacle["radius"]])

    return obs


def train(config):
    game_config = json.load(open("config.json"))
    env = PredatorsAndPreysEnv(game_config, render=False)
    predator = ChasingPredatorAgent()
    set_seed(env, 42)

    ddpg = DDPG(2 + 4 + game_config["game"]["num_obsts"] * 3, 1, 1.0)
    total_trans = 0

    for i in range(EPISODES):
        state = env.reset()
        current_rews, done = [], False

        while not done:
            action = ddpg.act(torch.tensor(observations(state)).float().view(1, -1))
            next_state, reward, done = env.step(predator.act(state), action)
            reward = prey_reward(next_state["preys"][0], next_state)
            current_rews.append(reward)
            ddpg.buffer.push(observations(state), action, reward, observations(next_state), done)
            state = next_state
            ddpg.update()
            total_trans += 1

        ddpg.eps *= 0.98

        if i % 10 == 0:
            print(f"Episode: {i}")
            print(f"Mean reward: {np.mean(current_rews)}")
            print(f"Total transitions: {total_trans}")
            print()

            torch.save(ddpg.actor.state_dict(), "./solutions/prey.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    train(config)


import argparse
import json

import torch
import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")


from models.td3.converts import state_to_prey_obs
from models.td3.actor import Actor
from models.td3.rewards import prey_reward
from models.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent
from models.imitation_learning.predator import PredatorAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


class Prey:
    def __init__(self, state_dim: int, action_dim: int):
        self.model = Actor(state_dim, action_dim)
        self.model.load_state_dict(torch.load("./solutions/prey.pkl"))
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            action = self.model(torch.FloatTensor(state_to_prey_obs(state)).view(1, -1))

        return action.numpy()


def evaluate(eval_config):
    game_config = json.load(open("config.json"))
    game_config["environment"]["time_limit"] = eval_config.time_limit

    env = PredatorsAndPreysEnv(game_config, render=True)
    env.seed(eval_config.seed)
    
    predator = PredatorAgent()
    prey = Prey(state_dim=3 + 2 * 2 + 5 * 3, action_dim=1)
    # prey = FleeingPreyAgent()

    total_reward = 0.
    for ep_i in range(eval_config.episodes):
        state = env.reset()
        done = False

        while not done: 
            next_state, _, done = env.step(predator.act(state), prey.act(state))
            state = next_state

            total_reward += prey_reward(state["preys"][0], state)

    print(f"Mean reward for {eval_config.episodes} episodes: {round(total_reward / eval_config.episodes, 4)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--time_limit", default=200, type=int)
    parser.add_argument("--episodes", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    config = parser.parse_args()
    evaluate(config)

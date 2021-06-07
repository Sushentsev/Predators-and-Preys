import json

import torch
import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")

from models.ddpg.actor import Actor
from models.ddpg.train import observations
from models.simple_chasing_agents.agents import ChasingPredatorAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


class Prey:
    def __init__(self, state_dim: int):
        self.model = Actor(state_dim, 1)
        self.model.load_state_dict(torch.load("./solutions/prey.pkl"))
        self.model.eval()

    def act(self, obs):
        with torch.no_grad():
            action = self.model(torch.Tensor(obs).view(1, -1))

        return action.numpy()


def evaluate():
    game_config = json.load(open("config.json"))
    game_config["environment"]["time_limit"] = 50

    env = PredatorsAndPreysEnv(game_config, render=True)
    predator = ChasingPredatorAgent()
    prey = Prey(2 + 4 + game_config["game"]["num_obsts"] * 3)

    for ep_i in range(50):
        state = env.reset()
        done = False

        while not done:
            next_state, _, done = env.step(predator.act(state), prey.act(observations(state)))
            state = next_state


if __name__ == '__main__':
    evaluate()

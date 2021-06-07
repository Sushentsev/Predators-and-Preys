from typing import List

from models.maddpg.actor_network import ActorNetwork
from models.maddpg.utils import convert_state
from predators_and_preys_env.env import PredatorsAndPreysEnv

import torch
import json

CONFIG = json.load(open("../maddpg_v2/config.json"))


class PredatorAgent:
    def __init__(self, num_preds: int, state_dim: int, action_dim: int):
        self.num_preds = num_preds
        self.preds = [ActorNetwork(state_dim, action_dim) for _ in range(self.num_preds)]

    def act(self, state_dict) -> List[float]:
        actions = []
        state = torch.tensor(convert_state(state_dict)).view(1, -1).float()

        with torch.no_grad():
            for pred in self.preds:
                action = pred(state)
                actions.append(action.item())

        return actions

    def load(self, path: str):
        for i, pred in enumerate(self.preds):
            pred.load_state_dict(torch.load(path + f"pred{i}.pkl"))
            pred.eval()


class PreyAgent:
    def __init__(self, num_preys: int, state_dim: int, action_dim: int):
        self.num_preys = num_preys
        self.preys = [ActorNetwork(state_dim, action_dim) for _ in range(self.num_preys)]

    def act(self, state_dict) -> List[float]:
        actions = []
        state = torch.tensor(convert_state(state_dict)).view(1, -1).float()

        with torch.no_grad():
            for prey in self.preys:
                action = prey(state)
                actions.append(action.item())

        return actions

    def load(self, path: str):
        for i, prey in enumerate(self.preys):
            prey.load_state_dict(torch.load(path + f"prey{i}.pkl"))
            prey.eval()


def main():
    env = PredatorsAndPreysEnv(CONFIG, render=True)
    num_preds = CONFIG["game"]["num_preds"]
    num_preys = CONFIG["game"]["num_preys"]
    num_obsts = CONFIG["game"]["num_obsts"]
    num_agents = num_preds + num_preys
    state_dim = num_preds * 4 + num_preys * 5 + num_obsts * 3

    predator_agent = PredatorAgent(num_preds, state_dim, 1)
    prey_agent = PreyAgent(num_preys, state_dim, 1)
    predator_agent.load("./solutions/")
    prey_agent.load("./solutions/")

    done = True
    step_count = 0
    state_dict = None
    while True:
        if done:
            state_dict = env.reset()
            step_count = 0

        state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
        step_count += 1


if __name__ == '__main__':
    # main()

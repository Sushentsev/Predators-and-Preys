import sys

sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")

from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
import torch
from models.simple_chasing_agents.agents import ChasingPredatorAgent
from models.simple_chasing_agents.agents import FleeingPreyAgent
from models.imitation_learning.train import ActorNetwork
from models.imitation_learning.misc import vectorize_state


class ILPreyAgent:
    def __init__(self):
        self.model = ActorNetwork(5 + 2 * 4 + 10 * 3, 64)
        self.model.load_state_dict(torch.load("./prey.pkl"))
        self.model.eval()

    def act(self, state_dict):
        actions = []

        with torch.no_grad():
            for i in range(len(state_dict["preys"])):
                state = vectorize_state(state_dict, i, "prey")
                action = self.model(torch.FloatTensor(state).view(1, -1)).item()
                actions.append(action)

        return actions


class ILPredatorAgent:
    def __init__(self):
        self.model = ActorNetwork(4 + 5 * 5 + 10 * 3, 64)
        self.model.load_state_dict(torch.load("./predator.pkl"))
        self.model.eval()

    def act(self, state_dict):
        actions = []

        with torch.no_grad():
            for i in range(len(state_dict["predators"])):
                state = vectorize_state(state_dict, i, "predator")
                action = self.model(torch.FloatTensor(state).view(1, -1)).item()
                actions.append(action)

        return actions


env = PredatorsAndPreysEnv(render=True)
env.time_limit = 200
predator_agent = ILPreyAgent()
prey_agent = ILPreyAgent()

done = True
state_dict = None
while True:
    if done:
        state_dict = env.reset()
        step_count = 0

    state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
    step_count += 1

import argparse
import json

import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")

import numpy as np

from models.imitation_learning.misc import vectorize_state
from models.imitation_learning.predator import PredatorAgent
from models.imitation_learning.prey import PreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


class Buffer:
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))

        self.filled_i = 0

    def clear(self):
        self.states = np.zeros((self.buffer_size, self.state_dim))
        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.filled_i = 0

    def push(self, state: np.ndarray, action: np.ndarray):
        if self.filled_i >= self.buffer_size:
            return

        self.states[self.filled_i] = state
        self.actions[self.filled_i] = action
        self.filled_i += 1

    def full(self) -> bool:
        return self.filled_i == self.buffer_size

    def save(self, name: str, dir: str):
        np.save(f"{dir}/{name}:actions.npy", self.actions)
        np.save(f"{dir}/{name}:states.npy", self.states)


def parse(params):
    env = PredatorsAndPreysEnv(json.load(open("config.json")), render=False)
    prey_buffer = Buffer(params.save_size, 5 + env.game.num_preds * 4 + env.game.num_obsts * 3, 1)
    predator_buffer = Buffer(params.save_size, 4 + env.game.num_preys * 5 + env.game.num_obsts * 3, 1)

    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()

    done = True
    transitions = 0
    while True:
        if done:
            state_dict = env.reset()        

        predator_action = predator_agent.act(state_dict)
        prey_action = prey_agent.act(state_dict)

        if transitions % 5 == 0:
            for i, prey in enumerate(state_dict["preys"]):
                prey_buffer.push(vectorize_state(state_dict, i, "prey"), prey_action[i])

            for j, predator in enumerate(state_dict["predators"]):
                predator_buffer.push(vectorize_state(state_dict, j, "predator"), predator_action[j])

        state_dict, reward, done = env.step(predator_action, prey_action)
        transitions += 1

        if predator_buffer.full():
            predator_buffer.save(f"{transitions}", "./predator")
            predator_buffer.clear()
            print(f"Save predator buffer at transition {transitions}")

        if prey_buffer.full():
            prey_buffer.save(f"{transitions}", "./prey")
            prey_buffer.clear()
            print(f"Save prey buffer at transition {transitions}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_size", default=100_000, type=int)
    params = parser.parse_args()

    parse(params)

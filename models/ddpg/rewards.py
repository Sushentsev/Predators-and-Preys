from typing import Tuple
import numpy as np
import math


def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2)


def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) + 1e-4 < agent1["radius"] + agent2["radius"]


def prey_reward(prey, state_dict) -> float:
    rew = 0.
    shape = True

    if shape:
        for pred in state_dict["predators"]:
            rew += 0.1 * distance(pred, prey)

    for pred in state_dict["predators"]:
        if is_collision(prey, pred):
            rew -= 10.

    return rew


def pred_reward(pred, state_dict) -> float:
    rew = 0.
    shape = True

    if shape:
        for prey in state_dict["preys"]:
            rew -= 0.1 * distance(pred, prey)

    for predator in state_dict["predators"]:
        for prey in state_dict["preys"]:
            if is_collision(predator, prey):
                rew += 10.

    return rew

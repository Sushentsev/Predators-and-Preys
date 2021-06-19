from typing import Tuple
import numpy as np
import math


def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2)


def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) + 1e-3 < agent1["radius"] + agent2["radius"]


def prey_reward(prey, state_dict) -> float:
    rew = 0.
    shape = False

    if shape:
        for pred in state_dict["predators"]:
            rew += 0.1 * distance(pred, prey)

    for pred in state_dict["predators"]:
        if is_collision(prey, pred):
            rew -= 10.

    # def bound(x: float) -> float:
    #     if x >= 0.3:
    #         return 0
    #     else:
    #         return -50 * x + 5

    # for obst in state_dict["obstacles"]:
    #     rew -= bound(distance(obst, prey) - prey["radius"] - obst["radius"])

    return rew


def pred_reward(pred, state_dict) -> float:
    rew = 0.
    shape = False

    if shape:
        for prey in state_dict["prey"]:
            rew -= 0.1 * distance(pred, prey)

    for predator in state_dict["predators"]:
        for prey in state_dict["prey"]:
            if is_collision(predator, prey):
                rew += 10.

    return rew

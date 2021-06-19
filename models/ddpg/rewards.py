from typing import Tuple
import numpy as np
import math

def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2) - agent1["radius"] - agent2["radius"]


def top_k(agent, entities, k: int = None):
    distances = [distance(agent, entity) for entity in entities]
    return [entities[index] for index in np.argsort(distances)][:k]


def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) < 1e-3


def prey_reward(prey, state_dict) -> float:
    rew = 0.
    shape = False

    if shape:
        for pred in top_k(prey, state_dict["predators"], k=2):
            rew += 0.1 * distance(pred, prey)

    for pred in state_dict["predators"]:
        if is_collision(prey, pred):
            rew -= 15.

    def bound(x: float) -> float:
        if x <= 0.2:
            return -2.
        else:
            return 0.

    rew += bound(9 - np.abs(prey["x_pos"]) - prey["radius"])
    rew += bound(9 - np.abs(prey["y_pos"]) - prey["radius"])

    return rew


def pred_reward(pred, state_dict) -> float:
    rew = 0.
    shape = True

    if shape:
        alive_preys_distances = [distance(pred, prey) for prey in state_dict["preys"] if prey["is_alive"]]
        if len(alive_preys_distances) > 0:
            rew -= 0.1 * min(alive_preys_distances)

    for prey in state_dict["preys"]:
        if prey["is_alive"] and is_collision(pred, prey):
            rew += 15.

    return rew

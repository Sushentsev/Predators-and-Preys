import math

import numpy as np


def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2) - agent1[
        "radius"] - agent2["radius"]


def top_k(agent, entities, k: int = None):
    distances = [distance(agent, entity) for entity in entities]
    return [entities[index] for index in np.argsort(distances)][:k]


def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) < 1e-3


def prey_reward(prey, state_dict) -> float:
    rew = 0.
    shape = True

    if shape:
        for pred in state_dict["predators"]:
            rew += 0.1 * distance(pred, prey)

    for pred in state_dict["predators"]:
        if is_collision(prey, pred):
            rew -= 15.

    def bound(x: float) -> float:
        if x <= 0.2:
            return -5.
        else:
            return 0.

    rew += bound(9 - np.abs(prey["x_pos"]) - prey["radius"])
    rew += bound(9 - np.abs(prey["y_pos"]) - prey["radius"])

    return rew


def pred_reward(pred, state_dict) -> float:
    rew = 0.
    shape = True

    if shape:
        closest_prey = top_k(pred, state_dict["preys"])
        rew -= 0.1 * distance(pred, closest_prey)

    for prey in state_dict["prey"]:
        if prey["is_alive"] and is_collision(pred, prey):
                rew += 15.

    return rew

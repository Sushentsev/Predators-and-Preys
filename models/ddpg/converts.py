import numpy as np
import math


def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2) - agent1[
        "radius"] - agent2["radius"]


def top_k(agent, entities, k: int = None):
    distances = [distance(agent, entity) for entity in entities]
    return [entities[index] for index in np.argsort(distances)][:k]


def angle(agent1, agent2) -> float:
    return np.arctan2(agent1["y_pos"] - agent2["y_pos"], agent1["x_pos"] - agent2["x_pos"]) / np.pi


def state_to_prey_obs(state_dict) -> np.ndarray:
    prey = state_dict["preys"][0]
    obs = [prey["x_pos"], prey["y_pos"], prey["radius"]]

    for predator in top_k(prey, state_dict["predators"]):
        # obs.extend([dist, predator["x_pos"] - prey["x_pos"], predator["y_pos"] - prey["y_pos"]])
        obs.extend([distance(prey, predator), angle(prey, predator)])

    for obstacle in top_k(prey, state_dict["obstacles"], k=5):
        # obs.extend([dist, obstacle["x_pos"] - prey["x_pos"], obstacle["y_pos"] - prey["y_pos"]])
        obs.extend([distance(prey, obstacle), angle(prey, obstacle), obstacle["radius"]])

    return np.array([obs])


def state_to_pred_obs(state_dict) -> np.ndarray:
    predator = state_dict["predators"][0]
    obs = [predator["x_pos"], predator["y_pos"], predator["radius"]]

    for prey in top_k(predator, [prey for prey in state_dict["preys"] if prey["is_alive"]]):
        obs.extend([distance(prey, predator), angle(prey, predator), 100])

    for _ in top_k(predator, [prey for prey in state_dict["preys"] if not prey["is_alive"]]):
        obs.extend([0.] * 3)

    for obstacle in top_k(predator, state_dict["obstacles"], k=5):
        # obs.extend([dist, obstacle["x_pos"] - prey["x_pos"], obstacle["y_pos"] - prey["y_pos"]])
        obs.extend([distance(predator, obstacle), angle(predator, obstacle), obstacle["radius"]])

    return np.array([obs])


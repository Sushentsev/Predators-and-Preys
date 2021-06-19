from typing import Dict, Any

import numpy as np


def distance(agent1: Dict[Any, Any], agent2: Dict[Any, Any]) -> float:
    pos1 = np.array([agent1["x_pos"], agent1["y_pos"]])
    pos2 = np.array([agent2["x_pos"], agent2["y_pos"]])

    return np.sqrt(np.sum(np.square(pos1 - pos2))) - agent1["radius"] - agent2["radius"]


def sort(agent, entities):
    distances = [distance(agent, entity) for entity in entities]
    return [entities[index] for index in np.argsort(distances)]


def vectorize_state(state_dict, agent_id: int, agent_type: str) -> np.ndarray:
    observation = []
    if agent_type == "prey":
        agent = state_dict["preys"][agent_id]
        observation.extend([agent["x_pos"], agent["y_pos"], agent["speed"], agent["radius"], agent["is_alive"]])

        for predator in sort(agent, state_dict["predators"]):
            observation.extend([predator["x_pos"] - agent["x_pos"],
                                predator["y_pos"] - agent["y_pos"],
                                predator["speed"], predator["radius"]])

        for obstacle in sort(agent, state_dict["obstacles"]):
            observation.extend([obstacle["x_pos"] - agent["x_pos"],
                                obstacle["y_pos"] - agent["y_pos"],
                                obstacle["radius"]])

        assert len(observation) == 5 + 2 * 4 + 10 * 3

    elif agent_type == "predator":
        agent = state_dict["predators"][agent_id]
        observation.extend([agent["x_pos"], agent["y_pos"], agent["speed"], agent["radius"]])

        for prey in sort(agent, state_dict["preys"]):
            observation.extend([prey["x_pos"] - agent["x_pos"],
                                prey["y_pos"] - agent["y_pos"],
                                prey["speed"], prey["radius"], prey["is_alive"]])

        for obstacle in sort(agent, state_dict["obstacles"]):
            observation.extend([obstacle["x_pos"] - agent["x_pos"],
                                obstacle["y_pos"] - agent["y_pos"],
                                obstacle["radius"]])
        assert len(observation) == 4 + 5 * 5 + 10 * 3

    else:
        raise ValueError("Incorrect agent type.")

    return np.array(observation)

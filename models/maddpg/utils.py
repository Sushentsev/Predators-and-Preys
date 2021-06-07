import numpy as np


def convert_state(state_dict):
    entities_types = ["predators", "preys", "obstacles"]
    keys = {
        "predators": ["x_pos", "y_pos", "radius", "speed"],
        "preys": ["x_pos", "y_pos", "radius", "speed", "is_alive"],
        "obstacles": ["x_pos", "y_pos", "radius"]
    }

    out = []

    for entity_type in entities_types:
        entities_list = state_dict[entity_type]
        entity_keys = keys[entity_type]
        for entity in entities_list:
            for key in entity_keys:
                out.append(entity[key])

    return np.array(out)

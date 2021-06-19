import numpy as np

    
    
def state_to_obs(state_dict):
    agent = state_dict["prey"][0]
    obs = [agent["x_pos"], agent["y_pos"],
           agent["speed"], agent["radius"]]
    enemy = state_dict["predators"][0]

    obs.extend([agent["x_pos"] - enemy["x_pos"],
                agent["y_pos"] - enemy["y_pos"],
                enemy["speed"], enemy["radius"]])

    for obstacle in state_dict["obstacles"]:
        obs.extend([agent["x_pos"] - obstacle["x_pos"],
                    agent["y_pos"] - obstacle["y_pos"],
                    obstacle["radius"]])

    return np.array(obs)
    


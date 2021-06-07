from typing import List


def observation(agent_i, agents, obstacles) -> List[float]:
    obs = [agents[agent_i]["radius"]]

    for i, agent in enumerate(agents):
        if i != agent_i:
            dx = agent["x_pos"] - agents[agent_i]["x_pos"]
            dy = agent["y_pos"] - agents[agent_i]["y_pos"]
            speed = agent["speed"]
            obs.extend([dx, dy, speed])

    for obst in obstacles:
        dx = obst["x_pos"] - agents[agent_i]["x_pos"]
        dy = obst["y_pos"] - agents[agent_i]["y_pos"]
        radius = obst["radius"]
        obs.extend([dx, dy, radius])

    return obs


def observations(state_dict) -> List[List[float]]:
    agents = state_dict["predators"] + state_dict["preys"]
    obstacles = state_dict["obstacles"]
    return [observation(agent_i, agents, obstacles) for agent_i in range(len(agents))]

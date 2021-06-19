import numpy as np
import torch
from torch import nn
from typing import Any, Dict

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

    else:
        raise ValueError("Incorrect agent type.")

    return np.array(observation)
    
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class PreyAgent:
    def __init__(self):
        self.model = ActorNetwork(5 + 2 * 4 + 10 * 3, 64)
        # self.model.load_state_dict(torch.load("./prey.pkl"))    
        self.model.load_state_dict(torch.load(__file__[:-13] + "/prey.pkl"))    
        self.model.eval()

    def act(self, state_dict):
        actions = []

        with torch.no_grad():
            for i in range(len(state_dict["preys"])):
                state = vectorize_state(state_dict, i, "prey")
                action = self.model(torch.FloatTensor(state).view(1, -1)).item()
                actions.append(action)

        return actions




class PredatorAgent:
    def __init__(self):
        self.model = ActorNetwork(4 + 5 * 5 + 10 * 3, 64)
        # self.model.load_state_dict(torch.load("./predator.pkl")) 
        self.model.load_state_dict(torch.load(__file__[:-13] + "/predator.pkl"))           
        self.model.eval()

    def act(self, state_dict):
        actions = []

        with torch.no_grad():
            for i in range(len(state_dict["predators"])):
                state = vectorize_state(state_dict, i, "predator")
                action = self.model(torch.FloatTensor(state).view(1, -1)).item()
                actions.append(action)

        return actions


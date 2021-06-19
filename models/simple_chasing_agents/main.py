from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
from models.simple_chasing_agents.agents import ChasingPredatorAgent
from models.simple_chasing_agents.agents import FleeingPreyAgent
import time

env = PredatorsAndPreysEnv(render=True)
predator_agent = ChasingPredatorAgent()
prey_agent = FleeingPreyAgent()

done = True
step_count = 0
state_dict = None
while True:
    if done:
        state_dict = env.reset()
        step_count = 0

    prey_actions = prey_agent.act(state_dict)

    for action in prey_actions:
        if np.abs(action) > 1.:
            print(action)

    state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
    step_count += 1

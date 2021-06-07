import argparse
import json
import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys/")
import torch

from models.maddpg_v2.maddpg import MADDPG
from models.maddpg_v2.state_converts import observations
from predators_and_preys_env.env import PredatorsAndPreysEnv


def evaluate(game_config, n_episodes: int = 50):
    maddpg = MADDPG.init_from_save("./solutions/state_dict")
    env = PredatorsAndPreysEnv(game_config, render=True)
    maddpg.prep_rollouts()
    
    from models.simple_chasing_agents.agents import ChasingPredatorAgent
    predator = ChasingPredatorAgent()

    for ep_i in range(n_episodes):
        obs = env.reset()
        obs_tr = observations(obs)
        done = False

        while not done:
            torch_obs = [torch.Tensor(obs_tr[i]).view(1, -1) for i in range(maddpg.num_agents)]
            torch_actions = maddpg.step(torch_obs, explore=False)
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, _, done = env.step(actions[:env.game.num_preds], actions[env.game.num_preds:])
            # obs, _, done = env.step(predator.act(obs), actions[env.game.num_preds:])
            obs_tr = observations(obs)


if __name__ == '__main__':
    game_config = json.load(open("config.json"))
    game_config["environment"]["time_limit"] = 50
    evaluate(game_config)

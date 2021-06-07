import argparse
import json
import random
import numpy as np
import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")
import torch

from models.maddpg_v2.buffer import ReplayBuffer
from models.maddpg_v2.maddpg import MADDPG
from models.maddpg_v2.rewards import pred_reward, prey_reward
from models.maddpg_v2.state_converts import observations
from predators_and_preys_env.env import PredatorsAndPreysEnv


def set_seed(env, seed):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train(game_config, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = PredatorsAndPreysEnv(game_config, render=False)
    set_seed(env, params.seed)

    maddpg = MADDPG.init_from_env(env, device)
    num_in_actor = 3 * (env.game.num_preys + env.game.num_preds - 1) + 3 * env.game.num_obsts + 1
    replay_buffer = ReplayBuffer(params.buffer_length, maddpg.num_agents,
                                 [num_in_actor] * maddpg.num_agents, [1] * maddpg.num_agents)

    current_rewards = np.zeros((0, maddpg.num_agents))
    total_trns = 0
    for ep_i in range(params.n_episodes):
        obs = env.reset()
        obs = observations(obs)

        maddpg.prep_rollouts()
        explr_pct_remaining = max(0, params.n_exploration_eps - ep_i) / params.n_exploration_eps
        maddpg.scale_noise(
            params.final_noise_scale + (params.init_noise_scale - params.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        done = False
        while not done:
            torch_obs = [torch.Tensor(obs[agent_i]).view(1, -1).to(device) for agent_i in range(maddpg.num_agents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            next_obs, rewards, done = env.step(agent_actions[:env.game.num_preds], agent_actions[env.game.num_preds:])
            next_obs_tr = observations(next_obs)

            # rewards["predators"] = np.array([np.sum(rewards["predators"])] * env.game.num_preds)
            predator_state = next_obs["predators"][0]
            prey_state = next_obs["preys"][0]
            reward_concat = np.array([pred_reward(predator_state, next_obs),prey_reward(prey_state, next_obs)])
            replay_buffer.push(obs, agent_actions, reward_concat, next_obs_tr, [done] * maddpg.num_agents)
            obs = next_obs_tr

            if len(replay_buffer) >= params.batch_size:
                maddpg.prep_training()
                for agent_i in range(maddpg.num_agents):
                    sample = replay_buffer.sample(params.batch_size, device=device)
                    maddpg.update(sample, agent_i)
                    maddpg.update_all_targets()

                maddpg.prep_rollouts()

            current_rewards = np.vstack((current_rewards, reward_concat))
            total_trns += 1

        if ep_i % params.update_every == 0:
            print(f"Current episode: {ep_i}")
            print(f"Mean {params.update_every} episodes rewards: {current_rewards.mean(axis=0)}")
            print(f"Total transitions: {total_trns}")
            print()

            current_rewards = np.zeros((0, maddpg.num_agents))
            maddpg.save("./solutions/state_dict")


if __name__ == '__main__':
    game_config = json.load(open("config.json"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=25_000, type=int)
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25_000, type=int)
    parser.add_argument("--update_every", default=10, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--actor_lr", default=0.01, type=float)
    parser.add_argument("--critic_lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    params = parser.parse_args()

    train(game_config, params)

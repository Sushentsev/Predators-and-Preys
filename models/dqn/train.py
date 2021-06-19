import argparse
import json

from gym import make
import numpy as np
import torch
import random

from tqdm import tqdm

from models.dqn.converts import state_to_obs
from models.dqn.dqn import DQN
from models.dqn.rewards import prey_reward
from models.simple_chasing_agents.agents import ChasingPredatorAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


def set_seed(env, seed):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def take_initial_steps(env, dqn):
    state = env.reset()
    for _ in tqdm(range(1024)):
        action = random.randint(0, 7)

        next_state, _, done = env.step(random.random() * 2 - 1, -1 + action / 4)
        dqn.consume_transition((state_to_obs(state),
                                action,
                                state_to_obs(next_state),
                                prey_reward(state["prey"][0], state),
                                done))

        state = next_state if not done else env.reset()

    return state


def train(config):
    env = PredatorsAndPreysEnv(json.load(open("config.json")), render=True)
    predator = ChasingPredatorAgent()
    set_seed(env, config.seed)

    dqn = DQN(state_dim=4 + 4 + env.game.num_obsts * 3, action_dim=8)

    total_reward = 0.
    state = take_initial_steps(env, dqn)
    decay = 0.99
    eps = 0.5

    for transition in range(config.transitions):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = random.randint(0, 7)
        else:
            action = dqn.act(state_to_obs(state))

        prey_action = dqn.act(state_to_obs(state))
        next_state, _, done = env.step(predator.act(state), -1 + prey_action / 4)
        dqn.update((state_to_obs(state),
                    action,
                    state_to_obs(next_state),
                    prey_reward(state["prey"][0], state),
                    done))
        total_reward += prey_reward(state["prey"][0], state)

        state = next_state if not done else env.reset()
        eps *= decay if done else 1

        if transition % config.save_every == 0:
            print(f"Current transition: {transition}")
            print(f"Mean prey reward: {round(total_reward / config.save_every, 4)}")
            total_reward = 0.
            dqn.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--transitions", default=1_000_000, type=int)
    parser.add_argument("--initial_steps", default=1_000, type=int)
    parser.add_argument("--save_every", default=10_000, type=int)
    config = parser.parse_args()

    train(config)


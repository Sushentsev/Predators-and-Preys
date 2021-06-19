import numpy as np



def prey_reward(prey, predators):
    reward = 0

    if prey["x_pos"] < 0:
        return 0
    else:
        return 10

    for predator in predators:
        if distance(prey, predator) < 2:
            reward -= 10

    # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    # def bound(x):
    #     if x < 0.9:
    #         return 0
    #     if x < 1.0:
    #         return (x - 0.9) * 10
    #     return min(np.exp(2 * x - 2), 10)
    #
    # for p in range(world.dim_p):
    #     x = abs(agent.state.p_pos[p])
    #     rew -= bound(x)

    return reward


def predator_reward(predator, preys):
    # Adversaries are rewarded for collisions with agents
    reward = 0
    #
    # for prey in prey:
    #     dx = prey["x_pos"] - predator["x_pos"]
    #     dy = prey["y_pos"] - predator["y_pos"]
    #     length = np.sqrt(np.sum(np.square([dx, dy])))
    #
    #     reward -= 0.1 * length

    if (predator["x_pos"] < 6) and (predator["x_pos"] > -6) and (predator["y_pos"] < 6) and (predator["y_pos"] > -6):
        return 10
    else:
        return 0

import gym
from gym import wrappers
import random
import numpy as np
from value_function import ValueFunction
from collections import defaultdict
from collections import namedtuple
import itertools
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_average_performance(episode_stats, algor_name, num_runs):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot()
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    ax.set_title(f"Episode Length over Time (averaged over {num_runs} runs)")

    average = episode_stats.mean()
    average.plot(ax=ax)

    fig.savefig(algor_name + "_normal_y_scale.png")
    ax.set_yscale('log')
    fig.savefig(algor_name + "_log_y_scale.png")
    plt.show(block=False)


def SARSA_semi_gradient(env, value_function, num_episodes, discount_factor=1.0, epsilon=0.1, alpha=0.1, print_=False, record_animation=False):
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    q = value_function.__call__

    for i_episode in range(num_episodes):
        if(print_ and ((i_episode + 1) % 100 == 0)):
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")

        # Exploit and record animation once we have learned q value
        if(i_episode == num_episodes-1 and record_animation):
            epsilon = 0

        state = env.reset()
        action = value_function.act(state, epsilon)

        for j in itertools.count():
            next_state, reward, done, info = env.step(action)
            # Update stats per episode
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = j

            if(done):
                target = reward
                value_function.update(target, state, action)
                break
            else:
                # Estimate q-value at next state-action
                next_action = value_function.act(next_state, epsilon)
                q_new = q(next_state, next_action)
                target = reward + discount_factor * q_new

                value_function.update(target, state, action)
            state = next_state
            action = next_action

    return stats


def run_sarsa_semi_gradient(num_episodes=500, num_runs=1):
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 10000
    episode_lengths_total = pd.DataFrame([])
    for i in range(num_runs):
        print(f"\n Run {i+1} of {num_runs} \n")
        value_function = ValueFunction(alpha=0.1, n_actions=env.action_space.n)
        stats = SARSA_semi_gradient(
            env, value_function, num_episodes, print_=True)
        episode_lengths_total = episode_lengths_total.append(
            pd.DataFrame(stats.episode_lengths).T)
    env.close()

    plot_average_performance(
        episode_lengths_total, "SARSA_Semi-Gradient_for_Mountain_Car_Environment", num_runs)


def animate_environment(num_episodes=500):
    print(" \n Animating the last episode")
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 10000
    env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id==num_episodes-1,force = True)
    value_function = ValueFunction(alpha=0.1, n_actions=env.action_space.n)
    stats = SARSA_semi_gradient(
        env, value_function, num_episodes,  print_=True, record_animation=True)
    env.close()

def main():
    run_sarsa_semi_gradient(num_episodes=500,num_runs=1)
    animate_environment(num_episodes=500)


if __name__ == "__main__":
    main()

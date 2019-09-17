import gym
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


def td_plot_episode_stats(stats, algor_name="", smoothing_window=10):
    # Plot the episode length over time

    fig = plt.figure(figsize=(10, 10))
    st = fig.suptitle(algor_name, fontsize="large")

    # fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(311)
    ax1.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    ax1.set_title("Episode Length over Time")

    # Plot the episode reward over time
    # fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig.add_subplot(312)
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    ax2.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    ax2.set_title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))

    # Plot time steps and episode number
    # fig3 = plt.figure(figsize=(10, 5))
    ax3 = fig.add_subplot(313)
    ax3.plot(np.cumsum(stats.episode_lengths),
             np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    ax3.set_title("Episode per time step")

    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.99)
    fig.subplots_adjust(top=0.93)

    fig.savefig(algor_name + "_td_plot_episode_stats.png")
    plt.show(block=False)


def SARSA_semi_gradient(env, value_function, num_episodes, discount_factor=1.0, epsilon=0.1, alpha=0.1, print_=False):
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    q = value_function.__call__

    for i_episode in range(num_episodes):
        if(print_ and ((i_episode + 1) % 100 == 0)):
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")

        state = env.reset()
        action = value_function.act(state, epsilon)

        for j in itertools.count():
            next_state, reward, done, info = env.step(action)
            # Update stats per episode
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = j

            if(done):
                value_function.update(reward+q(state, action), state, action)
                break
            next_action = value_function.act(next_state, epsilon)
            value_function.update(reward+discount_factor*q(
                next_state, next_action)-q(state, action), next_state, next_action)
            state = next_state
            action = next_action
            # env.render()

    return stats


def main():
    env = gym.make('MountainCar-v0')
    value_function = ValueFunction(alpha=0.1, n_actions=env.action_space.n)
    num_runs = 1
    episode_lengths_total = pd.DataFrame([])
    for i in range(num_runs):
        stats = SARSA_semi_gradient(env, value_function, 500, print_=True)
        episode_lengths_total = episode_lengths_total.append(
            pd.DataFrame(stats.episode_lengths).T)
    print(episode_lengths_total)
    # td_plot_episode_stats(stats, "SARSA Semi Gradient")


if __name__ == "__main__":
    main()

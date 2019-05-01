import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, alpha, discount, e, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Rewards")
    plt.title("Episode Rewards over Time; alpha = " + str(alpha) + ", discount factor = " + str(discount) + ", epsilon = " + str(e))
    if noshow:
        plt.close(fig1)
        plt.savefig("Reward_vs_time.png")
    else:
        plt.show(fig1)
        plt.savefig("Reward_vs_time.png")

    return fig1

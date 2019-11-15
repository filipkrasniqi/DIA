import functools
import math
import os, pickle
import pathlib

import numpy as np

from Code.Part_1.Learner import Learner
import seaborn as sns
import matplotlib.pyplot as plt

from Code.Part_1.ProjectEnvironment import User, ProjectEnvironment

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
output_plots_dir = "v1"
env_dir = outputs_dir+"v01_with_context/"
output_dir = env_dir+"{}/".format(output_plots_dir)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

results = pickle.load(open("{}/results.pickle".format(env_dir), 'rb'))
sigmas = results["sigma"]
hue_val = 0
total_length = 0
colors_palette = {}
season_length = 91

def num_users(idx, t):
    t_in_season, season = (t % season_length) + 1, int(t / season_length)
    q = (1+idx) * 1000
    coeffs_user = [0.5, 8, 10]
    coeffs_season = [1, -2, 0.5, 0.3]
    m = (1+idx) * coeffs_user[idx] * coeffs_season[season] * (-1)
    return q + m * t_in_season

def gauss(coeff, sigma, t, mu, price):
    x = price - mu
    t_in_season = t % season_length + 1
    max_coeff, min_coeff = 1, 0
    m_t = (max_coeff - min_coeff) / (season_length - 1)
    q_t = min_coeff - m_t
    coeff_t = t_in_season * m_t + q_t
    coeff_price = 1
    exp_ = math.exp(-1 * (x ** 2) / (2 * sigma **2))
    fraction_ =  1 / (sigma * math.pow(np.pi * 2, 0.5))
    ret_val = coeff_price * coeff_t * coeff * fraction_ * exp_
    return ret_val

n_users = 3
single_context_alternatives = [[[0, 1, 2]]]
batch_size = 16
matrix_parameters_aggregate = [
    [[0.01, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
     [0.03, 400, [20, 40, 60, 80], functools.partial(gauss, 160, 5)],
     [0.03, 300, [20, 40, 60, 80], functools.partial(gauss, 150, 5)],
     [0.02, 300, [20, 40, 60, 80], functools.partial(gauss, 140, 5)]]
]
T = 363
min_price = 10
max_price = 150
n_arms = math.ceil(math.pow(T * math.log(T, 10), 0.25))
arms = np.linspace(min_price, max_price, num=n_arms)
num_users_functions = [functools.partial(num_users, i) for i in range(n_users)]
env = ProjectEnvironment(arms, num_users_functions, 1, [matrix_parameters_aggregate], single_context_alternatives, batch_size=batch_size)

for learner_name in [key for key in results.keys() if "sigma" not in key]:
    sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
    for sigma, _ in zip(sigmas, output_per_sigma):
        total_length += 10

for learner_name in [key for key in results.keys() if "sigma" not in key]:
    sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
    for sigma, _ in zip(sigmas, output_per_sigma):
        colors_palette[learner_name+" - "+str(sigma)] = User.hue_map(hue_val / total_length)
        hue_val += 10

def plot(group_history, learner_name, sigma = None):
    if sigma is None:
        sigma = ""
        history = group_history
    else:
        history = group_history[sigma]
    x = list(range(len(history)))
    ax = sns.lineplot(x=x, y=history, markers=True, label=learner_name + " - " + str(sigma))
    ax.legend()
    return ax, history

for learner_name in [key for key in results.keys() if "sigma" not in key]:
    sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
    cumulative_regrets_groupped, regrets_groupped, rewards_groupped = {},{},{}
    # plot cumulative regret
    for sigma, output_per_sigma in zip(sigmas, output_per_sigma):
        (_, real_rewards, regret_history, cumulative_regret_history, (idx_c, idx_s, demand_mapping)) = output_per_sigma
        cumulative_regrets_groupped[sigma] = cumulative_regret_history
        regrets_groupped[sigma] = regret_history
        rewards_groupped[sigma] = real_rewards

    all_y_vals = []
    min_x, max_x = 0, 0
    for sigma in cumulative_regrets_groupped.keys():
        ax, y_vals = plot(cumulative_regrets_groupped, learner_name, sigma)
        all_y_vals += y_vals
        max_x = len(y_vals)

    ax.set_title("Cumulative regret")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Value [$]")
    max_val = np.max(all_y_vals)
    delta = max_val / 10
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(-delta, max_val + delta)
    for s_idx in [i * 91 for i in range(1, 4)]:
        plt.axvline(s_idx, 0, max(all_y_vals), color="#FF0000A0", linestyle='dashed')
    plt.show()

    fig = ax.get_figure()
    name = "{}cum_regrets_{}.png".format(output_dir, learner_name)
    fig.savefig(name)

    all_y_vals = []
    for sigma in regrets_groupped.keys():
        ax, y_vals = plot(regrets_groupped, learner_name, sigma)
        all_y_vals += y_vals

    ax.set_title("Regret")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Value [$]")
    max_val = np.max(all_y_vals)
    delta = max_val / 10
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(-delta, max_val + delta)
    for s_idx in [i * 91 for i in range(1, 4)]:
        plt.axvline(s_idx, 0, max(all_y_vals), color="#FF0000A0", linestyle='dashed')
    plt.show()

    fig = ax.get_figure()
    name = "{}regret_{}.png".format(output_dir, learner_name)
    fig.savefig(name)

    all_y_vals = []
    for sigma in rewards_groupped.keys():
        ax, y_vals = plot(rewards_groupped, learner_name, sigma)
        all_y_vals += y_vals

    ax.set_title("Rewards")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Value [$]")
    max_val = np.max(all_y_vals)
    delta = max_val / 10
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(-delta, max_val + delta)
    for s_idx in [i * 91 for i in range(1, 4)]:
        plt.axvline(s_idx, 0, max(all_y_vals), color="#FF0000A0", linestyle='dashed')
    plt.show()

    fig = ax.get_figure()
    name = "{}rewards_{}.png".format(output_dir, learner_name)
    fig.savefig(name)
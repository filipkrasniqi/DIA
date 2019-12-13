import functools
import math
import os, pickle
import pathlib

import numpy as np
import pandas

import seaborn as sns
import matplotlib.pyplot as plt

from Code.Part_1.ProjectEnvironment import User, ProjectEnvironment

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
output_plots_dir = "v1"
env_dir = outputs_dir+"seqab_v2/"
output_dir = env_dir+"{}/".format(output_plots_dir)
output_dir_with_context = env_dir+"{}_ctx/".format(output_plots_dir)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(output_dir_with_context).mkdir(parents=True, exist_ok=True)

plot_context = False
plot_without_context = True

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

n_users = 3
single_context_alternatives = [[[0, 1, 2]]]
batch_size = 16
T = 363
min_price = 25
max_price = 750
n_arms = math.ceil(math.pow(T * math.log(T, 10), 0.25))
arms = np.linspace(min_price, max_price, num=n_arms)
num_users_functions = [functools.partial(num_users, i) for i in range(n_users)]

def plot_with_season(to_plot, learner_name, filename, title, x_label, y_label, scatters = None):
    all_y_vals = []
    min_x, max_x = 0, 0
    for sigma in to_plot.keys():
        ax, y_vals = plot(to_plot, learner_name, sigma)
        all_y_vals += y_vals
        max_x = len(y_vals)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    max_val = np.max(all_y_vals)
    delta = max_val / 10
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(-delta, max_val + delta)
    for s_idx in [i * 91 for i in range(1, 4)]:
        plt.axvline(s_idx, 0, max(all_y_vals), color="#FF0000A0", linestyle='dashed')

    if scatters is not None:
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 14,
                }
        for scatter in scatters:
            ax.text(max_x - 25, scatter, str(round(scatter, 2)), fontdict=font)
    plt.show()

    fig = ax.get_figure()
    name = "{}{}_{}.png".format(output_dir, filename, learner_name)
    fig.savefig(name)


if plot_without_context:
    total_length = 0
    hue_val = 0
    colors_palette = {}
    results = pickle.load(open("{}/results.pickle".format(env_dir), 'rb'))
    sigmas = results["sigma"]

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        for sigma, _ in zip(sigmas, output_per_sigma):
            total_length += 10

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        for sigma, _ in zip(sigmas, output_per_sigma):
            colors_palette[learner_name+" - "+str(sigma)] = User.hue_map(hue_val / total_length)
            hue_val += 10

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        cumulative_regrets_groupped, regrets_groupped, rewards_groupped = {},{},{}
        # plot cumulative regret
        max_y_per_sigma = []
        for sigma, output_per_sigma in zip(sigmas, output_per_sigma):
            (results_c_learner, idx_c, history_best_contexts_selected, history_best_contexts, history_results_each_context) = output_per_sigma["results_c_learner"], output_per_sigma["idx_c"], output_per_sigma[
                    "history_best_contexts_selected"], output_per_sigma["history_best_contexts"], \
                output_per_sigma["history_results_each_contexts"]

            for results_learner in results_c_learner:
                (_, real_rewards, regret_history, cumulative_regret_history, _, _) = results_learner
                print()

            cumulative_regrets_groupped[sigma] = cumulative_regret_history
            regrets_groupped[sigma] = regret_history
            rewards_groupped[sigma] = real_rewards
            max_y_per_sigma.append(max(cumulative_regret_history))

        plot_with_season(cumulative_regrets_groupped, learner_name, "cum_regret", "Cumulative regret", "Time [day]", "Value [$]", scatters=max_y_per_sigma)
        plot_with_season(regrets_groupped, learner_name,
                         "regret", "Regret", "Time [day]", "Value [$]")
        plot_with_season(rewards_groupped, learner_name,
                         "rewards", "Rewards", "Time [day]", "Value [$]")

histories_best_contexts, histories_best_contexts_selected_to_show = {}, {}
if plot_context:
    total_length = 0
    hue_val = 0
    colors_palette = {}
    results = pickle.load(open("{}/results_context.pickle".format(env_dir), 'rb'))
    sigmas = results["sigma"]

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        for sigma, _ in zip(sigmas, output_per_sigma):
            total_length += 10

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        for sigma, _ in zip(sigmas, output_per_sigma):
            colors_palette[learner_name + " - " + str(sigma)] = User.hue_map(hue_val / total_length)
            hue_val += 10

    for learner_name in [key for key in results.keys() if "sigma" not in key]:
        sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
        cumulative_regrets_groupped, regrets_groupped, rewards_groupped = {}, {}, {}
        # plot cumulative regret
        max_y_per_sigma = []
        for idx, (sigma, output_per_sigma) in enumerate(zip(sigmas, output_per_sigma)):
            (results_c_learner, idx_c, history_best_contexts_selected, history_best_contexts, history_results_each_context) = \
                output_per_sigma["results_c_learner"], output_per_sigma["idx_c"], output_per_sigma[
                    "history_best_contexts_selected"], output_per_sigma["history_best_contexts"], \
                output_per_sigma["history_results_each_contexts"]
            if idx == 0:
                # take only first to show how selection of contexts works
                history_best_contexts_to_show, history_best_contexts_selected_to_show, sigma_to_show = history_best_contexts, history_best_contexts_selected, sigma
            for results_learner in results_c_learner:
                (_, real_rewards, regret_history, cumulative_regret_history, _, _) = results_learner
                print()
            cumulative_regrets_groupped[sigma] = cumulative_regret_history
            regrets_groupped[sigma] = regret_history
            rewards_groupped[sigma] = real_rewards
            max_y_per_sigma.append(max(cumulative_regret_history))

            cumulative_regrets_groupped[sigma] = cumulative_regret_history
            regrets_groupped[sigma] = regret_history
            rewards_groupped[sigma] = real_rewards

        key = learner_name+" - "+str(sigma_to_show)
        histories_best_contexts[key] = history_best_contexts_to_show
        histories_best_contexts_selected_to_show[key] = history_best_contexts_selected_to_show
        plot_with_season(cumulative_regrets_groupped, learner_name, "cum_regret", "Cumulative regret", "Time [day]",
                         "Value [$]", scatters=max_y_per_sigma)
        plot_with_season(regrets_groupped, learner_name,
                         "regret", "Regret", "Time [day]", "Value [$]")
        plot_with_season(rewards_groupped, learner_name,
                     "rewards", "Rewards", "Time [day]", "Value [$]")

        # TODO scatterplot on ...
        # ax.text(point['x'] + .02, point['y'], str(point['val']))


    weeks, learner_names, histories, histories_best = np.array([]), np.array([]), np.array([]), np.array([])
    for learner_name in histories_best_contexts.keys():
        history_best_tmp, history = histories_best_contexts[learner_name], histories_best_contexts_selected_to_show[learner_name]
        histories = np.append(histories, history)
        histories_best = np.append(histories_best, [history_tmp[1] for history_tmp in history_best_tmp])
        learner_names = np.append(learner_names, np.repeat(learner_name, len(history)))
        weeks_list = list(range(len(history)))
        weeks = np.append(weeks, weeks_list)

    list_if_selected = np.repeat("Selected", len(histories))
    list_if_selected = np.append(list_if_selected, np.repeat("Best", len(histories_best)))
    weeks = np.append(weeks, weeks_list)
    learner_names = np.append(learner_names, np.repeat("Best", len(weeks_list)))

    df_contexts = pandas.DataFrame(data={"Week": weeks, "Learner": learner_names, "Selected Context": np.append(histories, histories_best), "Selected/Best": list_if_selected})
    # ax, y_vals = plot(history, learner_name)
    # markers = {"Selected": "x", "Best": "s"}
    ax = sns.scatterplot(x="Week", y="Selected Context", style="Selected/Best", data=df_contexts, s=80, hue="Selected/Best")
    # ax = sns.scatterplot(x="Week", y="Best Context", style="Learner", data=df_contexts, s=80, color=".2", marker="+")
    # g = sns.catplot(x="Week", y="pulse", hue="kind", data=exercise)
    ax.set_title("History selected contexts")
    ax.set_xlabel("Week")
    ax.set_ylabel("Context")
    plt.xlim(0, 52)
    plt.ylim(-0.5, 5)

    for s_idx in [13, 26, 39]:
        plt.axvline(s_idx, 0, 52, color="#FF0000A0", linestyle='dashed')
    plt.show()

    fig = ax.get_figure()
    name = "{}{}.png".format(output_dir, "history_selection")
    fig.savefig(name)
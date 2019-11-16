from Code.learners.GPTS_Learner import *
from Code.Part_3.environment import *
from Code.Part_3.dp_algorithm import *

import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import time

import warnings

warnings.filterwarnings("ignore")


def noisy_sampling(n_subcampaigns, gpts_learners):
    samples = list()
    for idx_subcampaign in range(0, n_subcampaigns):
        clicks = gpts_learners[idx_subcampaign].pull_arms()
        samples.append(clicks)
    return samples


def real_sampling(arms, n_sub_campaigns, env):
    samples = list()
    for idx_subcampaign in range(0, n_sub_campaigns):
        vet = list()
        for j in range(0, len(arms)):
            clicks = env.get_clicks_real(arms[j], idx_subcampaign)
            vet.append(clicks)
        samples.append(vet)
    return samples


def plot_regression(current_folder, arms, environment, idx_subcampaign, save_figure=False):
    """
    Plots the regression graph.
    :param current_folder: folder to put the plots in.
    :param arms: arms.
    :param environment: variable referencing the environment.
    :param idx_subcampaign: index of the subcampaign to plot.
    :param save_figure: indicates whether to save the figure or not.
    """

    # Plot every subcampaign.
    real_function_y = []
    for arm in arms:
        # For each arm (budget) of a subcampaign, get real number of clicks obtained.
        real_function_y.append(environment.get_clicks_real(arm, idx_subcampaign))

    x_pred = np.atleast_2d(arms).T

    y = np.array(gpts_learners[idx_subcampaign].means)
    sigma_gp = np.array(gpts_learners[idx_subcampaign].sigmas)

    plt.figure(t)
    plt.plot(x_pred, real_function_y, 'r:', label=r'$Real Function$')
    plt.plot(x_pred, y, 'b-', label=u'Predicted Rewards')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y - 1.96 * sigma_gp, (y + 1.96 * sigma_gp)[::-1]]),
             alpha=0.5, fc='b', ec='None', label='95% conf interval')

    plt.xlabel('$x$')
    plt.ylabel('$Real Function$')
    plt.legend(loc='lower right')
    plt.title("Subcampaign {}".format(idx_subcampaign))
    if save_figure:
        plt.savefig('{}/prediction_subcampaign_{}.png'.format(current_folder, idx_subcampaign))

    plt.show()


# Functions characteristics.
bids = np.array([
    [5, 2, 1],
    [2, 1, 4],
    [3, 4, 1],
    [3, 5, 1],
    [3, 4, 1]])
slopes = -1 * np.array([
    [0.5, 0.3, 0.1],
    [0.3, 0.1, 0.4],
    [0.4, 0.4, 0.1],
    [0.5, 0.5, 0.2],
    [0.3, 0.3, 0.1]])
max_clicks = np.array([
    [20, 10, 10],
    [15, 10, 25],
    [20, 25, 10],
    [20, 25, 10],
    [15, 20, 10]])
user_probabilities = [[
    [0.80, 0.10, 0.10],
    [0.20, 0.10, 0.70],
    [0.30, 0.60, 0.10],
    [0.20, 0.70, 0.10],
    [0.30, 0.65, 0.05]]]

# Prepare environment.
sigma_env_n = [0.01, 1, 2]
n_subcampaigns = 5
n_users_x_subcampaign = 3
n_arms_sub = 21
total_budget = 200
min_daily_budget = 0.0
max_daily_budget = total_budget
T = 200

# Folders to save images.
curr_dir = os.getcwd()
outputs_dir = curr_dir + "/outputs/"
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
env_dir = outputs_dir + "2vars/"
if not os.path.exists(env_dir):
    os.mkdir(env_dir)

# One iteration for each combination of sigma and user probabilities.
for k, s_p in enumerate(itertools.product(sigma_env_n, user_probabilities)):
    sigma_env = s_p[0]
    prob_users = s_p[1]

    cur_fold = env_dir + str(k)
    if not os.path.exists(cur_fold):
        os.mkdir(cur_fold)

    gpts_rewards_per_experiment_sub_1 = list()
    gaussian_error_per_experiment_1 = list()

    env = Environment(
        n_arms=n_arms_sub,
        n_users=n_users_x_subcampaign,
        n_subcampaigns=n_subcampaigns,
        max_budget=total_budget,
        prob_users=prob_users,
        sigma=sigma_env,
        bids=bids,
        slopes=slopes,
        max_clicks=max_clicks)
    arms = env.get_arms()
    # env.plot()

    # CLAIRVOYANT ALGORITHM.

    # Execute combinatorial algorithm to get optimal distribution of budgets to different subcampaigns.
    samples = real_sampling(arms, n_subcampaigns, env)
    combinatorial_alg = DPAlgorithm(arms, n_subcampaigns, samples, min_daily_budget, total_budget)
    combinatorial = combinatorial_alg.get_budgets()
    # Get optimal value of clicks for the campaign (clairvoyant).
    optimum = combinatorial[0]

    # REAL ALGORITHM.

    rewards_per_round = list()
    regression_error = list()
    arm_obs = np.array(list())

    gpts_learners = list()
    # For each subcampaign, define a GPTS learner.
    for idx_subcampaign in range(0, n_subcampaigns):
        gpts_learners.append(GPTSLearner(arms=arms, sigma=sigma_env))

    start_time = time.time()
    # Every round, pull arms and update rewards.
    for t in range(1, T + 1):
        # Sample all the learners.
        samples = noisy_sampling(n_subcampaigns, gpts_learners)
        # Run the DP algorithm in order to get optimal distribution of budgets between subcampaigns.
        combinatorial_alg = DPAlgorithm(arms, n_subcampaigns, samples, min_daily_budget, total_budget)
        combinatorial = combinatorial_alg.get_budgets()
        # Array containing optimal allocation of budgets.
        arms_to_pull = combinatorial[1]
        # Total budget instantiated for the campaign.
        instantiated_budget = np.sum(arms_to_pull)

        # Pull arms and get the rewards (number of clicks with noise).
        noisy_rewards = [env.get_clicks_noise(arm, idx_subcampaign) for idx_subcampaign, arm in enumerate(arms_to_pull)]
        # Get real number of clicks from a subcampaign for a certain budget.
        real_rewards = [env.get_clicks_real(arm, idx_subcampaign) for idx_subcampaign, arm in enumerate(arms_to_pull)]

        # Calculate regression error.
        current_regression_error = [abs(reward - real_value_for_arm) for (reward, real_value_for_arm) in
                                    zip(noisy_rewards, real_rewards)]
        # Regression error is avg(pulled_clicks - real_clicks).
        regression_error.append(np.max(np.array(current_regression_error)))

        # For each subcampaign, update respective learner.
        for idx_subcampaign in range(0, n_subcampaigns):
            # Get index of pulled arm.
            idx_pulled_arm = gpts_learners[idx_subcampaign].arms.tolist().index(arms_to_pull[idx_subcampaign])
            gpts_learners[idx_subcampaign].update(idx_pulled_arm, noisy_rewards[idx_subcampaign])

            if t % 50 == 0:
                # Plot every subcampaign every 50 rounds.
                plot_regression(cur_fold, arms, env, idx_subcampaign)

        rewards_per_round.append(np.sum(real_rewards))

        # Print time necessary for 10 epochs.
        if t % 10 == 0:
            end_time = time.time()
            t_time = end_time - start_time
            print(str(t) + ' - time: ' + str(round(t_time, 2)) + ' sec')
            start_time = time.time()

    # PLOT CUMULATIVE REGRET.

    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plot1 = np.cumsum(optimum - rewards_per_round)
    plt.plot(plot1, 'r')
    # save log in file
    plt.savefig(cur_fold + '/cumreg.png')
    plt.show()

    # PLOT AVERAGE REGRESSION ERROR.

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Avg Regression Error")
    plt.plot(regression_error, 'b')
    plt.savefig(cur_fold + '/avrregerr.png')
    plt.show()

    # PLOT REGRESSION ERROR FOR EACH SUBCAMPAIGN.

    for idx_subcampaign in range(n_subcampaigns):
        plot_regression(cur_fold, arms, env, idx_subcampaign, save_figure=True)

    log = str(sigma_env) + "\n" + str(prob_users)
    f = open(cur_fold + "/log.txt", "w")
    f.write(log)
    f.close()
    print("Completed {} out of {}".format(k + 1, len(sigma_env_n) * len(user_probabilities)))

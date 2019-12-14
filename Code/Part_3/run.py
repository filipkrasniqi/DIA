import math
import os
import time
import warnings

from Code.Part_3.dp_algorithm import *
from Code.Part_3.environment import *
from Code.learners.GPTS_Learner import *

warnings.filterwarnings("ignore")


def plot_regression(current_folder, arms, environment, idx_subcampaign, t, save_figure=False):
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

    plt.xlabel("Budget [€]")
    plt.ylabel("Reward")
    plt.ylim((0, max(real_function_y) * max(environment.click_values)))
    plt.legend(loc='lower right')
    plt.title("Subcampaign {}, t = {}".format(idx_subcampaign, t))
    plt.grid()
    if save_figure:
        plt.savefig('{}/prediction_subcampaign_{}.png'.format(current_folder, idx_subcampaign))
    plt.show()

def pull_gpts_arms(learners):
    # Get one sample from each GPTS learner.
    samples = list()
    for idx_subcampaign in range(0, n_subcampaigns):
        clicks = learners[idx_subcampaign].sample_arms()
        samples.append(clicks)
    return samples


def real_sampling(arms, env):
    # Get real values from all arms, from all subcampaigns.
    samples = list()
    for idx_subcampaign in range(0, n_subcampaigns):
        vet = list()
        for j in range(0, len(arms)):
            clicks = env.get_clicks_real(arms[j], idx_subcampaign)
            vet.append(clicks)
        samples.append(vet)
    return samples


# Functions characteristics.
bids = np.array([
    [15, 10, 20],
    [20, 30, 15],
    [15, 25, 30],
    [27, 22, 14],
    [13, 29, 33]])
slopes = np.array([
    [0.05, 0.03, 0.01],
    [0.03, 0.01, 0.04],
    [0.04, 0.04, 0.01],
    [0.05, 0.05, 0.02],
    [0.03, 0.03, 0.01]])
max_clicks = np.array([
    [2.0, 1.0, 1.0],
    [1.5, 1.0, 2.5],
    [2.0, 2.5, 1.0],
    [2.0, 2.5, 1.0],
    [1.5, 2.0, 1.0]])
user_probabilities = [
    [1, 0, 0],
    [0.20, 0.10, 0.70],
    [0.30, 0.60, 0.10],
    [0.20, 0.70, 0.10],
    [0.30, 0.65, 0.05]]

# Prepare environment.
T = 200
sigma = 0.1
n_subcampaigns = 5
n_users_x_subcampaign = 3
n_arms = math.ceil(math.pow(T * math.log(T, 10), 0.25))
total_budget = 200
min_budgets = [0, 0, 0, 0, 0]
max_budgets = [200, 200, 200, 200, 200]
click_values = [1, 1.3, 1.6, 1.9, 2.2]

# Folders to save images.
curr_dir = os.getcwd()
outputs_dir = curr_dir + "/outputs/"
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
env_dir = outputs_dir + "2vars/"
if not os.path.exists(env_dir):
    os.mkdir(env_dir)


def get_all_number_of_clicks(arms, env):
    """
        Given an environment and the values of the arms, returns all the real number of clicks.
    """
    # Get real values from all arms, from all subcampaigns.
    number_of_clicks = list()
    for idx_subcampaign in range(0, n_subcampaigns):
        vet = list()
        for j in range(0, len(arms)):
            clicks = env.get_clicks_real(arms[j], idx_subcampaign)
            vet.append(clicks)
        number_of_clicks.append(vet)
    return number_of_clicks


cur_fold = env_dir
if not os.path.exists(cur_fold):
    os.mkdir(cur_fold)

env = Environment(
    n_arms=n_arms,
    n_users=n_users_x_subcampaign,
    n_subcampaigns=n_subcampaigns,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    total_budget=total_budget,
    user_probabilities=user_probabilities,
    sigma=sigma,
    bids=bids,
    slopes=slopes,
    max_clicks=max_clicks,
    click_values=click_values)
arms = env.get_arms()
env.plot()

# CLAIRVOYANT ALGORITHM.

# Execute combinatorial algorithm to get optimal distribution of budgets to different subcampaigns.
number_of_clicks = get_all_number_of_clicks(arms, env)
perfect_combinatorial_result = DPAlgorithm(arms, n_subcampaigns, number_of_clicks,
                                           min_budgets=min_budgets, max_budgets=max_budgets).get_budgets()
# Get optimal value of clicks for the campaign (clairvoyant).
optimum = perfect_combinatorial_result[0]

# REAL ALGORITHM.
regression_errors = list()
regrets = list()
cumulative_regrets = list()
avg_regression_errors = list()
arm_obs = np.array(list())

n_experiments = 10
rewards_per_exp = []

start_time = time.time()
# Every round, pull arms and update rewards.
for t in range(1, T + 1):
    # Get new batch of users.
    env.get_new_batch()

    # Sample all the learners.
    samples = pull_gpts_arms(gpts_learners)
    # Run the DP algorithm in order to get optimal distribution of budgets between subcampaigns.
    real_combinatorial_result = DPAlgorithm(arms, n_subcampaigns, samples,
                                            min_budgets=min_budgets, max_budgets=max_budgets).get_budgets()
    # Array containing optimal allocation of budgets.
    arms_to_pull = real_combinatorial_result[1]
    # Total budget instantiated for the campaign.
    instantiated_budget = np.sum(arms_to_pull)

    # Get real number of clicks from a subcampaign for a certain budget.
    real_rewards = [number_of_clicks[idx_subcampaign][arms.tolist().index(arm)] for idx_subcampaign, arm in enumerate(arms_to_pull)]
    # Pull arms and get the rewards (number of clicks with noise).
    noisy_rewards = [env.get_rewards(budget=arm, idx_subcampaign=idx_subcampaign) for idx_subcampaign, arm in
                     enumerate(arms_to_pull)]

    # For each subcampaign, update respective learner.
    for idx_subcampaign in range(0, n_subcampaigns):
        # Get index of pulled arm.
        idx_pulled_arm = gpts_learners[idx_subcampaign].arms.tolist().index(arms_to_pull[idx_subcampaign])

        # REGRESSION ERROR OF SUBCAMPAIGN 0.

        if idx_subcampaign == 0:
            regression_error = abs(samples[0][idx_pulled_arm] - real_rewards[0])
            regression_errors.append(regression_error)
            avg_error = np.average(regression_errors)
            avg_regression_errors.append(avg_error)

        gpts_learners[idx_subcampaign].update(
            idx_pulled_arm=idx_pulled_arm,
            reward=noisy_rewards[idx_subcampaign])

        if t % (T / 2) == 0:
            # Plot every subcampaign every 50 rounds.
            # plot_regression(cur_fold, arms, env, idx_subcampaign, t)
            pass

    rewards_per_round.append(np.sum(real_rewards))
    total_reward = np.sum(real_rewards)
    regret = abs(optimum - total_reward)
    regrets.append(regret)
    cumulative_regrets.append(sum(regrets))

    # Print time necessary for 10 epochs.
    if t % 10 == 0:
        end_time = time.time()
        t_time = end_time - start_time
        print("%d - time: %d min %d sec" % (t, int(t_time / 60), int(t_time % 60)))
        start_time = time.time()
        print("Regret: %.3f" % regret)
        print("AVG regret: %.3f" % np.average(regrets))
        print("Regression error: %.3f" % regression_error)
        print("AVG regression error: %.3f\n" % avg_error)


# PLOT REGRET.

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plot1 = regrets
plt.plot(plot1, 'r')
# save log in file
plt.savefig(cur_fold + '/reg.png')
plt.show()

# PLOT CUMULATIVE REGRET.

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
# plot1 = np.cumsum(np.mean(optimum - rewards_per_exp,axis=0))
plot1 = cumulative_regrets
plt.plot(plot1, 'r')
# save log in file
plt.savefig(cur_fold + '/cumreg.png')
plt.show()

# PLOT AVERAGE REGRESSION ERROR.

plt.figure(1)
plt.xlabel("t (SUBCAMPAIGN 0)")
plt.ylabel("Avg Regression Error")
plt.plot(avg_regression_errors, 'b')
plt.savefig(cur_fold + '/avrregerr.png')
plt.show()

# PLOT REGRESSION ERROR FOR EACH SUBCAMPAIGN.

for idx_subcampaign in range(n_subcampaigns):
    plot_regression(cur_fold, arms, env, idx_subcampaign, T, save_figure=True)

log = str(sigma) + "\n" + str(user_probabilities)
f = open(cur_fold + "/log.txt", "w")
f.write(log)
f.close()

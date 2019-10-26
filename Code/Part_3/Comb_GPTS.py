from Code.Part_3.GPTS_Learner import *
from Code.Part_3.Environment import *
from Code.Part_3.dp_algorithm import *
import matplotlib.pyplot as plt
import os
import itertools
import time

# Punto 3

def built_matrix_sub_budget_clicks(n_arms, arms, n_sub_campaign, gpts_learners):
    matrix = []
    for i in range(0, n_sub_campaign):
        samples = gpts_learners[i].pull_arm()
        matrix.append(samples)

    return matrix


def built_matrix_sub_budget_clicks_without_errors(n_arms, arms, n_sub_campaign, env):
    matrix = []
    for i in range(0, n_sub_campaign):
        vet = []
        for j in range(0, n_arms):
            clicks = env.get_clicks_real(arms[j], i)
            vet.append(clicks)

        matrix.append(vet)

    return matrix

'''
Prepare environment for script
'''
sigma_env_n = [0.01, 1, 2]
bid = 10

prob_users_n = [
    [
        [0.80, 0.10, 0.10],
        [0.70, 0.10, 0.20],
        [0.50, 0.10, 0.40],
        [0.05, 0.05, 0.90],
        [0.10, 0.30, 0.60]
    ]
]

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
env_dir = outputs_dir+"2vars/"
if not os.path.exists(env_dir):
    os.mkdir(env_dir)

n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 21
total_budget = 200

min_daily_budget = 0.0
max_daily_budget = total_budget

T = 250

def plot_regression(cur_fold, arms, env, idx_subcampaign, save_figure = False):
    # Plot every subcampaign.
    real_function_y = []
    for a in arms:
        real_function_y.append(env.get_clicks_real(a, idx_subcampaign))

    p = gpts_learners[i].arms[pulled_arm]
    reward = env.get_click_noise(p, i)
    gpts_learners[i].update(pulled_arm, reward)

    x_pred = np.atleast_2d(arms).T

    y = np.array(gpts_learners[i].means)
    sigmaGP = np.array(gpts_learners[i].sigmas)

    plt.figure(t)
    plt.plot(x_pred, real_function_y, 'r:', label=r'$Real Function$')
    plt.plot(x_pred, y, 'b-', label=u'Predicted Rewards')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y - 1.96 * sigmaGP, (y + 1.96 * sigmaGP)[::-1]]),
             alpha=0.5, fc='b', ec='None', label='95% conf interval')

    plt.xlabel('$x$')
    plt.ylabel('$Real Function$')
    plt.legend(loc='lower right')
    plt.title("Subcampaign {}".format(i))
    if save_figure:
        plt.savefig('{}/prediction_subcampaign_{}.png'.format(cur_fold, idx_subcampaign))

    plt.show()

for k, s_p in enumerate(itertools.product(sigma_env_n, prob_users_n)):
    sigma_env = s_p[0]
    prob_users = s_p[1]

    cur_fold = env_dir + str(k)
    if not os.path.exists(cur_fold):
        os.mkdir(cur_fold)

    gpts_rewards_per_experiment_sub_1 = []
    gaussian_error_per_experiment_1 = []

    env = Environment(n_arms_sub, n_users_x_sub_campaign, n_sub_campaign, total_budget, bid, prob_users, sigma_env)
    arms = env.get_arms()
    env.plot()

    # Val ottimo per calcolare Regret
    matrix = built_matrix_sub_budget_clicks_without_errors(n_arms_sub, arms, n_sub_campaign, env)
    combinatorial_alg = DPAlgorithm(arms, n_sub_campaign, matrix, min_daily_budget, total_budget)
    combinatorial = combinatorial_alg.get_budgets()
    optimum = combinatorial[0]

    gpts_learners = []
    for i in range(0, n_sub_campaign):
        gpts_learners.append(GPTS_Learner(n_arms=n_arms_sub, arms=arms, sigma_gp=sigma_env, initial_sigmas=sigma_env))

    # execute once the learners to avoid wrong convergence due to bad initialization
    '''
    for subc in range(0, n_sub_campaign):
        for i,arm in enumerate(arms):
            reward = env.get_click_noise(i, subc)
            gpts_learners[subc].update(i, reward)
    '''
    rewards_per_round = []
    regression_error = []
    arm_obs = np.array([])

    start_time = time.time()
    for t in range(1, T+1):
        matrix = built_matrix_sub_budget_clicks(n_arms_sub, arms, n_sub_campaign, gpts_learners)
        combinatorial_alg = DPAlgorithm(arms, n_sub_campaign, matrix, min_daily_budget, total_budget)
        combinatorial = combinatorial_alg.get_budgets()
        pulled_arms = combinatorial[1]
        instanciated_budget = np.sum(pulled_arms)
        if instanciated_budget > total_budget:
            print("QUALQUADRA NON COSA")
        # return the campaigns reward
        rewards = env.get_clicks_noise(pulled_arms)
        # k, s_p in enumerate(itertools.product(sigma_env_n, prob_users_n))
        real_value_for_arms = [env.get_clicks_real(arm, subc) for subc, arm in enumerate(pulled_arms)]
        # ()
        current_regression_error = [abs(reward - real_value_for_arm) for (reward, real_value_for_arm) in zip(rewards, real_value_for_arms)]
        # regression error is avg(pulled_clicks - real_clicks)
        regression_error.append(np.max(np.array(current_regression_error)))

        for i in range(0, n_sub_campaign):
            pulled_arm = int(np.where(gpts_learners[i].arms == pulled_arms[i])[0])
            gpts_learners[i].update(pulled_arm, rewards[i])

            if t % 5 == 0:
                # Plot every subcampaign.
                plot_regression(cur_fold, arms, env, i)

        # rewards_per_round.append(np.sum(rewards))
        # TODO prendere arm associati a rewards e calcolare reward vero.
        rewards_per_round.append(np.sum(real_value_for_arms))

        # Print time necessary for 10 epochs.
        if t % 10 == 0:
            end_time = time.time()
            t_time = end_time - start_time
            print(str(t) + ' - time: ' + str(round(t_time, 2)) + ' sec')
            start_time = time.time()

        # gpts_errors.append(np.max( np.absolute(env.means - gpts_learner.means) ))

    # gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plot1 = np.cumsum(optimum - rewards_per_round)
    plt.plot(plot1, 'r')
    # save log in file
    plt.savefig(cur_fold+'/cumreg.png')
    plt.show()

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Avg Regression Error")
    plt.plot(regression_error, 'b')
    plt.savefig(cur_fold+'/avrregerr.png')
    plt.show()

    # save also for each subcampaign what came up from the GP
    for i in range(n_sub_campaign):
        matrix = built_matrix_sub_budget_clicks(n_arms_sub, arms, n_sub_campaign, gpts_learners)
        combinatorial_alg = DPAlgorithm(arms, n_sub_campaign, matrix, min_daily_budget, total_budget)
        combinatorial = combinatorial_alg.get_budgets()
        pulled_arms = combinatorial[1]
        pulled_arm = int(np.where(gpts_learners[i].arms == pulled_arms[i])[0])
        arm_obs = np.append(arm_obs, arms[pulled_arm])
        plot_regression(cur_fold, arms, env, i, save_figure=True)

    log = str(sigma_env)+"\n"+str(prob_users)
    f = open(cur_fold+"/log.txt","w")
    f.write(log)
    f.close()
    print("Completed {} out of {}".format(k+1, len(sigma_env_n) * len(prob_users_n)))

# gaussian_error_per_experiment.append(gpts_errors)

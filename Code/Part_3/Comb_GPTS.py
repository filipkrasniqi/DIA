from Code.Part_3.GPTS_Learner import *
from Code.Part_3.Environment import *
from Code.Part_3.dp_algorithm import *
import matplotlib.pyplot as plt
import os
import itertools

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
sigma_env_n = [0.01, 0.1, 1]
bid = 10

prob_users_n = [
    [
        [0.80, 0.10, 0.10],
        [0.70, 0.10, 0.20],
        [0.50, 0.10, 0.40],
        [0.05, 0.05, 0.90],
        [0.10, 0.30, 0.60]
    ],
    [
        [0.70, 0.10, 0.20],
        [0.10, 0.30, 0.60],
        [0.20, 0.80, 0.00],
        [0.30, 0.50, 0.20],
        [0.40, 0.10, 0.50]
    ],
    [
        [0.80, 0.10, 0.10],
        [0.80, 0.10, 0.10],
        [0.80, 0.10, 0.10],
        [0.80, 0.10, 0.10],
        [0.80, 0.10, 0.10]
    ],
    [
        [0.70, 0.10, 0.20],
        [0.90, 0.10, 0.00],
        [0.10, 0.40, 0.50],
        [0.50, 0.30, 0.20],
        [0.20, 0.20, 0.60]
    ],
    [
        [0.30, 0.30, 0.40],
        [0.30, 0.30, 0.40],
        [0.30, 0.30, 0.40],
        [0.30, 0.30, 0.40],
        [0.30, 0.30, 0.40]
    ]
]

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
env_dir = outputs_dir+"prova_finale/"
if not os.path.exists(env_dir):
    os.mkdir(env_dir)

n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 21
total_budget = 100

min_daily_budget = 0.0
max_daily_budget = total_budget

T = 40

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

    # Val ottimo per calcolare Regret
    matrix = built_matrix_sub_budget_clicks_without_errors(n_arms_sub, arms, n_sub_campaign, env)
    combinatorial_alg = DPAlgorithm(arms, n_sub_campaign, matrix, min_daily_budget, total_budget)
    combinatorial = combinatorial_alg.get_budgets()
    optimum = combinatorial[0]

    gpts_learners = []
    for i in range(0, n_sub_campaign):
        gpts_learners.append(GPTS_Learner(n_arms=n_arms_sub, arms=arms, sigma_gp=sigma_env, initial_sigmas=sigma_env))

    rewards_per_round = []
    regression_error = []

    for t in range(0, T):
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
        real_value_for_arms = [env.get_clicks_real(arm_sub[0], arm_sub[1]) for arm_sub in itertools.product(pulled_arms, list(range(n_sub_campaign)))]
        # ()
        current_regression_error = [abs(reward - real_value_for_arm) for (reward, real_value_for_arm) in zip(rewards, real_value_for_arms)]
        # regression error is avg(pulled_clicks - real_clicks)
        regression_error.append(np.average(np.array(current_regression_error)))

        for i in range(0, n_sub_campaign):
            pulled_arm = int(np.where(gpts_learners[i].arms == pulled_arms[i])[0])
            gpts_learners[i].update(pulled_arm, rewards[i])

        rewards_per_round.append(np.sum(rewards))
        print(t)

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

    log = str(sigma_env)+"\n"+str(prob_users)
    f = open(cur_fold+"/log.txt","w")
    f.write(log)
    f.close()
    print("Completed {} out of {}".format(k+1, len(sigma_env_n) * len(prob_users_n)))

# gaussian_error_per_experiment.append(gpts_errors)

# plt.figure(0)
# plt.xlabel("t")
# plt.ylabel("Regression Error")
# plt.plot(np.mean(gaussian_error_per_experiment, axis=0), 'g')
# plt.legend(["GPTS"])
# plt.show()

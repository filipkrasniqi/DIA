from Code.Part_3.GPTS_Learner import *
from Code.Part_3.Environment import *
from Code.Part_3.dp_algorithm import *
import matplotlib.pyplot as plt

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


n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 21
total_budget = 100

min_daily_budget = 0.0
max_daily_budget = total_budget

sigma_env = 0.1
bid = 10
prob_users = [
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10]
]
T = 20

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
    gpts_learners.append(GPTS_Learner(n_arms=n_arms_sub, arms=arms,sigma_gp=sigma_env,initial_sigmas=sigma_env))

rewards_per_round = []

for t in range(0, T):
    matrix = built_matrix_sub_budget_clicks(n_arms_sub, arms, n_sub_campaign, gpts_learners)
    combinatorial_alg = DPAlgorithm(arms, n_sub_campaign, matrix, min_daily_budget, total_budget)
    combinatorial = combinatorial_alg.get_budgets()
    pulled_arms = combinatorial[1]

    # return the campaigns reward
    rewards = env.get_clicks_noise(pulled_arms)

    for i in range(0, n_sub_campaign):
        pulled_arm = int( np.where(gpts_learners[i].arms == pulled_arms[i])[0])
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
plt.show()

# gaussian_error_per_experiment.append(gpts_errors)

# plt.figure(0)
# plt.xlabel("t")
# plt.ylabel("Regression Error")
# plt.plot(np.mean(gaussian_error_per_experiment, axis=0), 'g')
# plt.legend(["GPTS"])
# plt.show()

import matplotlib.pyplot as plt
from GPTS_Learner import *
from Environment import *

n_arms = 20
min_daily_budget = 0.0
# parameter in input
max_daily_budget = 3.0
# linspace generates random numbers in ascent order, extracted from the given interval
daily_budget = np.linspace(min_daily_budget, max_daily_budget, n_arms)

# from filip bidding env
sigma = 10

T = 40

n_experiments = 100
gpts_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Bidding_Environment(daily_budgets=daily_budget, sigma=sigma)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=daily_budget)

    for t in range(0, T):
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'g')  # Green color for GPTS
plt.legend(["GPTS"])
plt.show()

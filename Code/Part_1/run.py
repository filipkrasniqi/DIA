import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment
from Code.Part_1.SequentialABLearner import SequentialABLearner

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [1]
T = 720


def linear(x):
    return 1

matrix_parameters = [
    [[0.01, 200, linear], [0.01, 200, linear], [0.01, 200, linear], [0.01, 200, linear]],
    [[0.01, 300, linear], [0.02, 300, linear], [0.02, 300, linear], [0.02, 300, linear]],
    [[0.01, 200, linear], [0.02, 200, linear], [0.02, 200, linear], [0.02, 200, linear]]
]

min_price = 10
max_price = 150
n_arms = math.ceil(math.pow(T * math.log(T, 10), 0.25))
arms = np.linspace(min_price, max_price, num=n_arms)

min_confidence = 0.95   # this is 1-alpha
alpha = 1 - min_confidence
beta = 0.05
delta = 0.001
portion_samples_ab_testing = 0.5
test_T = int(portion_samples_ab_testing*T)

norm_dist = stats.norm(0, 1)
z_a, z_b = norm_dist.pdf(1-alpha), norm_dist.pdf(beta)
do_sequential_AB = False

# 1) Sequential AB testing
for sigma in sigma_env_n:
    env = Environment(arms, probabilities, sigma, matrix_parameters)
    # env.plot()

    # sequential AB testing
    # 1) Sampling and computing empirical mean
    learner = SequentialABLearner(arms)
    for t in range(test_T):
        learner.pull_arm(env, t)

    # 2) Consider one candidate and run the hypothesis test.
    best_candidate = learner.best_candidate(min_confidence)

    # 3) Exploitation: I found the best pricing, compute rounds during exploration
    for t in range(test_T, T):
        learner.pull_arm(env, t, best_candidate)

    # plot regret and reward, aka the learner
    learner.plot(env)





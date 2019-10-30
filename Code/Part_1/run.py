import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment
from Code.Part_1.SequentialABLearner import SequentialABLearner

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [0.1]
T = 360


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

min_confidence = 0.95 # this is 1-alpha
alpha = 1 - min_confidence
beta = 0.05
delta = 0.1
# TODO what is sigma
# n_candidates = int(((z(1-alpha) + z(b))**2 * sigma ** 2) / delta ** 2)

for sigma in sigma_env_n:
    env = Environment(arms, probabilities, sigma, matrix_parameters)
    env.plot()

    # sequential AB testing
    # 1) Sampling and computing empirical mean
    learner = SequentialABLearner(n_arms)
    for t in range(int(T)):
        for idx_arm in range(n_arms):
            arm = arms[idx_arm]
            reward, user = env.round(arm, t)
            learner.update(idx_arm, reward, user)

    # 2) Consider one candidate and run the hypothesis test.
    best_candidate = 0
    for i in range(1, n_arms):
        alternative = i

        Z = learner.z(best_candidate, alternative)
        p_value = stats.norm.sf(Z)
        if 1 - p_value <= min_confidence:
            # TODO current implementation: compute alternative test
            # TODO ensure that, in this case, I need to do that or whether I should just take the alternative, i.e., best_candidate = alternative
            Z = learner.z(alternative, best_candidate)
            p_value = stats.norm.sf(Z)

            if 1 - p_value <= min_confidence:
                print("mierda")
                raise ValueError("Test scannellato: ne uno ne l'altro")
            else:
                best_candidate = alternative
        # else: # h1 is true, i.e., candidate_1 = candidate_h0 is the best

        # TODO show regret

    # I found best candidate. Now compare with optimal solution to compute regret and rewards over time, i.e., run for remaining horizon





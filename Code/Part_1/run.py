import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment
from Code.Part_1.SequentialABLearner import SequentialABLearner
from Code.Part_1.ContinuousTS_Learner import TS_Learner
from Code.Part_1.UCB_Learner import UCB_Learner

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [2]
T = 600


def linear(t):
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

min_confidence = 0.95  # this is 1-alpha
alpha = 1 - min_confidence
beta = 0.05
delta = 0.001
portion_samples_ab_testing = 0.4
test_T = int(portion_samples_ab_testing * T)

norm_dist = stats.norm(0, 1)
z_a, z_b = norm_dist.pdf(1 - alpha), norm_dist.pdf(beta)
do_sequential_AB = False
do_UCB = True
do_TS = False

do_UCB_wdw = False
do_TS_wdw = False


def train(learner):
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, matrix_parameters)
        for t in range(T):
            learner.pull_arm(env, t)
        learner.plot(env)


def train_context(learner):
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, matrix_parameters)
        for t in range(T):
            if t % 33 == 32:
                context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]
                results_alternatives = []
                for alternative in context_alternatives:
                    sum = 0
                    for context in alternative:
                        is_all = len(context) == len(probabilities)
                        mu, prob = learner.avg_bounds(context, alpha), learner.prob_lower_bound(context, alpha)
                        if is_all:
                            sum += min(mu)
                        else:
                            sum += prob * min(mu)
                        results_alternatives.append(sum)
                    # print("Alternative {}: {}".format(alternative, sum))
                print("Best alternative: {}".format(np.argmax(results_alternatives)))
            learner.pull_arm(env, t)
        learner.plot(env)


# 1) Sequential AB testing
if do_sequential_AB:
    train(learner=SequentialABLearner(arms, test_T, min_confidence))

# 2) UCB
if do_UCB:
    train_context(learner=UCB_Learner(arms))
    # train(learner=UCB_Learner(arms))

# 3) TS
if do_TS:
    train(learner=TS_Learner(arms))

window_length = math.pow(T, 0.5)

# 4) UCB with window
if do_UCB_wdw:
    train(learner=UCB_Learner(arms, window_length))

# 5) TS with window
if do_TS_wdw:
    train(learner=TS_Learner(arms, window_length))

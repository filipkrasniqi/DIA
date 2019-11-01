import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment
from Code.Part_1.SequentialABLearner import SequentialABLearner
from Code.Part_1.ContinuousTS_Learner import TS_Learner
from Code.Part_1.UCB_Learner import UCB_Learner

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [2]
T = 100


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

min_confidence = 0.95   # this is 1-alpha
alpha = 1 - min_confidence
beta = 0.05
delta = 0.001
portion_samples_ab_testing = 0.4
test_T = int(portion_samples_ab_testing*T)

norm_dist = stats.norm(0, 1)
z_a, z_b = norm_dist.pdf(1-alpha), norm_dist.pdf(beta)
do_sequential_AB = False
do_UCB = False
do_TS = False

do_UCB_wdw = False
do_TS_wdw = True

def train(learner):
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, matrix_parameters)
        for t in range(T):
            learner.pull_arm(env, t)
        learner.plot(env)

# 1) Sequential AB testing
if do_sequential_AB:
    train(learner=SequentialABLearner(arms, test_T, min_confidence))

# 2) UCB
if do_UCB:
    train(learner=UCB_Learner(arms))

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





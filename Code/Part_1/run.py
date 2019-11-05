import math

import numpy as np
import scipy.stats as stats
import functools

from Code.Part_1.ContextLearner import ContextLearner
from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment
from Code.Part_1.SequentialABLearner import SequentialABLearner
from Code.Part_1.ContinuousTS_Learner import TS_Learner
from Code.Part_1.UCB_Learner import UCB_Learner

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [32]
T = 128
context_change_period = 7
batch_size = 16

def identity(t):
    return 1

def linear(m, q, t):
    return ((t % context_change_period) + 1) * m + q

def sigmoid(t):
    return 1 / (1 + math.exp(-t/10000))

def cos(D, t):
    return -1*D*(math.cos(t*2*3.14*context_change_period))# + D*(math.sin(t*10)) + D*(math.sin(t*10) ** 2)# (-1/D)*(t%context_change_period)**2# D*(math.cos(t*10)) + D*(math.sin(t*10)) + D*(math.sin(t*10) ** 2)


matrix_parameters = [
    [
        [[0.01, 200, identity], [0.01, 200, identity], [0.01, 200, identity], [0.01, 200, identity]]
    ]
]

context_alternatives = [[[0], [1], [2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0, 1, 2]]]
# sbagliato ma funge: context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]

context_matrix_parameters = [
    [
        [[0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)]],
        [[0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)]],
        [[0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)], [0.02, 10, functools.partial(linear, 1, 1)]]
    ],
    [
        [[0.02, 400, identity], [0.02, 400, identity], [0.02, 400, identity], [0.02, 400, identity]],
        [[0.02, 400, identity], [0.02, 400, identity], [0.02, 400, identity], [0.02, 400, identity]]
    ],
    [
        [[0.02, 300, identity], [0.02, 300, identity], [0.02, 300, identity], [0.02, 300, identity]],
        [[0.02, 300, identity], [0.02, 300, identity], [0.02, 300, identity], [0.02, 300, identity]]
    ],
    [
        [[0.02, 200, identity], [0.02, 200, identity], [0.02, 200, identity], [0.02, 200, identity]],
        [[0.02, 200, identity], [0.02, 200, identity], [0.02, 200, identity], [0.02, 200, identity]]
    ],
    [
        [[0.02, 10, functools.partial(cos, 10)], [0.02, 10, functools.partial(cos, 10)], [0.02, 10, functools.partial(cos, 10)], [0.02, 10, functools.partial(cos, 10)]]
    ]
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
plot_env = True


def train(learner):
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, matrix_parameters, context_alternatives = [[0, 1, 2]])
        for t in range(T):
            learner.pull_arm(env, t)
        learner.plot(env)


def train_context(learner_constructor, arms, window_length=None):
    c_learners = []
    best_context = 0
    for idx_c, alternative in enumerate(context_alternatives):
        c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length, idx_c))
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, context_matrix_parameters, context_alternatives, batch_size=batch_size)
        for t in range(T):
            if plot_env:
                env.plot(4)

            rewards_per_arm, subcontexts, users = env.round_context(t)
            idxs_arm_current_clearner = 0
            subcontext_current_clearner = None
            for idx_c, (c_learner, rewards, user, subcontext) in enumerate(zip(c_learners, rewards_per_arm, users, subcontexts)):
                idxs_arm = c_learner.pull_arm(t, rewards, user)
                if idx_c == best_context:
                    idxs_arm_current_clearner = idxs_arm
                    subcontext_current_clearner = subcontext
            # ora lo faccio sul serio!
            env.round_for_arm(idxs_arm_current_clearner, t, subcontext_current_clearner)

            if t % context_change_period == context_change_period - 1:
                results_alternatives = []
                for idx_c, context in enumerate(context_alternatives):
                    learners = c_learners[idx_c]
                    print("\n\nContext {}\n\n".format(idx_c))
                    sum = 0
                    for idx_s, subcontext in enumerate(context):
                        learner = learners.get_learner(idx_s)
                        is_all = learners.getNumberLearners() == 1
                        # mu, prob = learner.avg_bounds(subcontext, alpha), learner.prob_lower_bound(subcontext, 1-alpha)
                        mu, prob = learner.avg_bounds_fixed(alpha), learner.prob_lower_bound_fixed(alpha) / t
                        if is_all:
                            sum += min(mu)
                        else:
                            sum += prob * min(mu)
                        print("Alternative {}: {} {} => {}".format(subcontext, mu, prob, sum))
                    results_alternatives.append(sum)
                best_context = np.argmax(results_alternatives)
                print("Best alternative: {}".format(best_context))
                env.set_context(best_context)
        learner.plot(env, t)


# 1) Sequential AB testing
if do_sequential_AB:
    train(learner=SequentialABLearner(arms, test_T, min_confidence))

# 2) UCB
if do_UCB:
    train_context(learner_constructor=UCB_Learner, arms=arms)
    # train(learner=UCB_Learner(arms))

# 3) TS
if do_TS:
    train_context(learner_constructor=TS_Learner, arms=arms)
    # train(learner=TS_Learner(arms))

window_length = math.pow(T, 0.5)

# 4) UCB with window
if do_UCB_wdw:
    train(learner=UCB_Learner(arms, window_length))

# 5) TS with window
if do_TS_wdw:
    train(learner=TS_Learner(arms, window_length))

import math

import numpy as np
import scipy.stats as stats
import functools

from Code.Part_1.ContextLearner import ContextLearner
from Code.Part_1.ProjectEnvironment import ProjectEnvironment
from Code.Part_1.SequentialABLearner import SequentialABLearner
from Code.Part_1.ContinuousTS_Learner import TS_Learner
from Code.Part_1.UCB_Learner import UCB_Learner

n_users = 3
sigma_env_n = [1]#, 2, 4]# [1, 8, 16, 32, 64]

season_length = 91
number_of_seasons = 4
T = 364
context_change_period = 7
batch_size = 8

def identity(t, price):
    return 1

def gauss(coeff, sigma, t, mu, price):
    x = price - mu
    t_in_season = t % season_length + 1
    max_coeff, min_coeff = 1, 0
    m_t = (max_coeff - min_coeff) / (season_length - 1)
    q_t = min_coeff - m_t
    coeff_t = t_in_season * m_t + q_t
    coeff_price = 1
    exp_ = math.exp(-1 * (x ** 2) / (2 * sigma **2))
    fraction_ =  1 / (sigma * math.pow(np.pi * 2, 0.5))
    ret_val = coeff_price * coeff_t * coeff * fraction_ * exp_
    return ret_val

matrix_parameters = [
    [
        [[0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]]
    ]
]

context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]
# sbagliato ma funge: context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]

context_matrix_parameters = [
    [
        [[0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]]
    ],
    [
        [[0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]],

        [[0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]]
    ],
    [
        [
            [0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
             [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
             [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
             [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]
        ],
        [
        [0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
         [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]
        ]
    ],
    [
        [
            [0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]
        ],
        [
            [0.07, 200, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.02, 120, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.02, 100, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 6)],
            [0.01, 80, [20, 40, 60, 80, 100], functools.partial(gauss, 75, 8)]
        ]
    ],
    [
        [
            [0.01, 600, [], functools.partial(gauss, 1,1)],
            [0.01, 400, [], functools.partial(gauss, 1,1)],
            [0.02, 300, [], functools.partial(gauss, 1,1)],
            [0.02, 200, [], functools.partial(gauss, 1,1)],
        ],
        [
            [0.02, 800, [], functools.partial(gauss, 1,1)],
            [0.03, 600, [], functools.partial(gauss, 1,1)],
            [0.04, 300, [], functools.partial(gauss, 1,1)],
            [0.02, 600, [], functools.partial(gauss, 1,1)],
        ],
[
            [0.02, 600, [], functools.partial(gauss, 1,1)],
            [0.03, 400, [], functools.partial(gauss, 1,1)],
            [0.04, 300, [], functools.partial(gauss, 1,1)],
            [0.02, 200, [], functools.partial(gauss, 1,1)]
        ]
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
plot_env = False
plot_context = True

def num_users(idx, t):
    t_in_season = (t % season_length) + 1
    q = (1+idx) * 1000
    coeffs = [0.5, 10, 12]
    m = (1+idx) * coeffs[idx] * (-1)
    return q + m * t_in_season

num_users_functions = [functools.partial(num_users, i) for i in range(n_users)]

def train(learner_constructor, window_length = None):
    single_context_alternatives = [[[0, 1, 2]]]
    single_context_matrix_parameters = matrix_parameters
    single_best_context = 0
    plots_for_sigma = []
    for sigma in sigma_env_n:
        c_learners = []
        for idx_c, alternative in enumerate(single_context_alternatives):
            c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length, idx_c, sigma))
        env = ProjectEnvironment(arms, num_users_functions, sigma, single_context_matrix_parameters, single_context_alternatives, batch_size=batch_size)
        # env.plot(0, T)
        env.plot_context(3, T)
        for t in range(T):
            rewards_per_arm, demands_per_arm, subcontexts, users = env.round_context(t)
            idxs_arm_current_clearner = 0
            subcontext_current_clearner = None
            for idx_c, (c_learner, rewards, user, subcontext, demands) in enumerate(
                    zip(c_learners, rewards_per_arm, users, subcontexts, demands_per_arm)):
                idxs_arm = c_learner.pull_arm(t, rewards, demands, user)
                if idx_c == single_best_context:
                    idxs_arm_current_clearner = idxs_arm
                    subcontext_current_clearner = subcontext
            # ora lo faccio sul serio!
            env.round_for_arm(idxs_arm_current_clearner, t, subcontext_current_clearner)
        plots_for_sigma.append(c_learners[0].learners[0].plot(env))

    return plots_for_sigma

def train_context(learner_constructor, window_length=None):
    c_learners = []
    best_context = 0
    for idx_c, alternative in enumerate(context_alternatives):
        c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length, idx_c))
    for sigma in sigma_env_n:
        env = ProjectEnvironment(arms, num_users_functions, sigma, context_matrix_parameters, context_alternatives, batch_size=batch_size)
        if plot_env:
            env.plot(T=T)
        if plot_context:
            env.plot_context(4, T)
        for t in range(T):
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
        learner.plot(env)

all_plots = {}
# Without context
do_without_context = False

# 1) Sequential AB testing
if do_without_context:
    if do_sequential_AB:
        plots_for_sigma = train(learner_constructor=SequentialABLearner)

    # 2) UCB
    if do_UCB:
        train(learner_constructor=UCB_Learner)

    # 3) TS
    if do_TS:
        train(learner_constructor=TS_Learner)

    window_length = int(math.pow(T, 0.5))

    # 4) UCB with window
    if do_UCB_wdw:
        train(learner_constructor=UCB_Learner, window_length=window_length)

    # 5) TS with window
    if do_TS_wdw:
        train(learner_constructor=TS_Learner, window_length=window_length)

do_context = True
# With context
# TODO sistemare funzioni di contesto

if do_context:
    # 6) UCB
    if do_UCB:
        train_context(learner_constructor=UCB_Learner)

    # 7) TS
    if do_TS:
        train_context(learner_constructor=TS_Learner)

    # 8) UCB with window
    if do_UCB_wdw:
        train_context(learner_constructor=UCB_Learner, window_length=window_length)

    # 9) TS with window
    if do_TS_wdw:
        train_context(learner_constructor=TS_Learner, window_length=window_length)

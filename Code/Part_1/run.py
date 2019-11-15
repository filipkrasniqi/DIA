import math

import numpy as np
import scipy.stats as stats
import functools

from Code.Part_1.ContextLearner import ContextLearner
from Code.Part_1.ProjectEnvironment import ProjectEnvironment
from Code.Part_1.SequentialABLearner import SequentialABLearner
from Code.Part_1.ContinuousTS_Learner import TS_Learner
from Code.Part_1.UCB_Learner import UCB_Learner

import pickle

env_dir = ProjectEnvironment.get_env_dir()

n_users = 3
sigma_env_n = [1, 2, 4]

season_length = 91
number_of_seasons = 4
T = 363
context_change_period = 7
batch_size = 16

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

matrix_parameters_aggregate = [
    [[0.01, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
     [0.03, 400, [20, 40, 60, 80], functools.partial(gauss, 160, 5)],
     [0.03, 300, [20, 40, 60, 80], functools.partial(gauss, 150, 5)],
     [0.02, 300, [20, 40, 60, 80], functools.partial(gauss, 140, 5)]]
]

matrix_parameters_u1 = [
            [0.01, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.01, 400, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.02, 300, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.02, 200, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
        ]

matrix_parameters_u2 = [
            [0.02, 800, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.03, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.04, 300, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.02, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
        ]

matrix_parameters_u3 = [
            [0.02, 600, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.03, 400, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.04, 300, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
            [0.02, 200, [20, 40, 60, 80], functools.partial(gauss, 175, 5)]
]

context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]

context_matrix_parameters = [
    matrix_parameters_aggregate,
    [
        [[0.02, 700, [20, 40, 60, 80], functools.partial(gauss, 175, 5)],
         [0.02, 500, [20, 40, 60, 80], functools.partial(gauss, 160, 5)],
         [0.03, 300, [20, 40, 60, 80], functools.partial(gauss, 150, 5)],
         [0.02, 200, [20, 40, 60, 80], functools.partial(gauss, 140, 5)]],

        matrix_parameters_u3
    ],
    [
        [
            [0.02, 600, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
             [0.02, 400, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
             [0.03, 300, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
             [0.02, 200, [20, 40, 60, 80], functools.partial(gauss, 75, 8)]
        ],
        matrix_parameters_u2
    ],
    [
        [
            [0.02, 650, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
            [0.03, 480, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
            [0.04, 280, [20, 40, 60, 80], functools.partial(gauss, 75, 6)],
            [0.02, 350, [20, 40, 60, 80], functools.partial(gauss, 75, 8)]
        ],
        matrix_parameters_u1
    ],
    [
        matrix_parameters_u1,
        matrix_parameters_u2,
        matrix_parameters_u3
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
do_UCB = False
do_TS = False

coeff_window_length = 3
window_length = int(3 * math.pow(T, 0.5))
do_UCB_wdw = True
do_TS_wdw = True
plot_env = False
plot_context = False

def num_users(idx, t):
    t_in_season, season = (t % season_length) + 1, int(t / season_length)
    q = (1+idx) * 1000
    coeffs_user = [0.5, 8, 10]
    coeffs_season = [1, -2, 0.5, 0.3]
    m = (1+idx) * coeffs_user[idx] * coeffs_season[season] * (-1)
    return q + m * t_in_season

num_users_functions = [functools.partial(num_users, i) for i in range(n_users)]

def train(learner_constructor, window_length = None):
    single_context_alternatives = [[[0, 1, 2]]]
    single_context_matrix_parameters = [matrix_parameters_aggregate]
    single_best_context = 0
    plots_for_sigma = []
    for sigma in sigma_env_n:
        c_learners = []
        for idx_c, alternative in enumerate(single_context_alternatives):
            c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length, idx_c, sigma))
        env = ProjectEnvironment(arms, num_users_functions, sigma, single_context_matrix_parameters, single_context_alternatives, batch_size=batch_size)
        # env.plot(0, T)
        if plot_context:
            env.plot_context(0, T)
        for t in range(1, T+1):
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
    plots_for_sigma = []
    for sigma in sigma_env_n:
        c_learners = []
        for idx_c, alternative in enumerate(context_alternatives):
            c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length, idx_c, sigma))
        best_context = 0
        history_best_contexts = [best_context]
        history_results_each_contexts = [[] for _ in context_alternatives] # for each context saving sums
        env = ProjectEnvironment(arms, num_users_functions, sigma, context_matrix_parameters, context_alternatives, batch_size=batch_size)
        if plot_env:
            env.plot(T=T)
        if plot_context:
            env.plot_context(4, T)
        for t in range(1, T+1):
            rewards_per_arm, demands_per_arm, subcontexts, users = env.round_context(t)
            idxs_arm_current_clearner = 0
            subcontext_current_clearner = None
            for idx_c, (c_learner, rewards, user, subcontext, demands) in enumerate(zip(c_learners, rewards_per_arm, users, subcontexts, demands_per_arm)):
                idxs_arm = c_learner.pull_arm(t, rewards, demands, user)
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
                        mu, prob = learner.avg_bounds_fixed(alpha), learner.prob_lower_bound_fixed(alpha) / t
                        if is_all:
                            sum += min(mu)
                        else:
                            sum += prob * min(mu)
                        print("Alternative {}: {} {} => {}".format(subcontext, mu, prob, sum))
                    results_alternatives.append(sum)
                    history_results_each_contexts[idx_c].append(sum)
                best_context = np.argmax(results_alternatives)
                print("Best alternative: {}".format(best_context))
                env.set_context(best_context)
                history_best_contexts.append(best_context)
        best_c_learner = c_learners[best_context]
        to_plot = {"results_c_learner": best_c_learner.plot(env)}
        to_plot["idx_c"] = best_context
        to_plot["history_best_contexts"] = history_best_contexts
        to_plot["history_results_each_contexts"] = history_results_each_contexts
        plots_for_sigma.append(to_plot)
    return plots_for_sigma

all_plots = {}
# Without context
do_without_context = False
entire_plots = {"sigma": sigma_env_n}

# 1) Sequential AB testing
if do_without_context:
    if do_sequential_AB:
        name = "AB"
        plots_for_sigma = train(learner_constructor=SequentialABLearner)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 2) UCB
    if do_UCB:
        name = "UCB"
        plots_for_sigma = train(learner_constructor=UCB_Learner)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 3) TS
    if do_TS:
        name = "TS"
        plots_for_sigma = train(learner_constructor=TS_Learner)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 4) UCB with window
    if do_UCB_wdw:
        name = "UCB_{}".format(window_length)
        plots_for_sigma = train(learner_constructor=UCB_Learner, window_length=window_length)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 5) TS with window
    if do_TS_wdw:
        name = "TS_{}".format(window_length)
        plots_for_sigma = train(learner_constructor=TS_Learner, window_length=window_length)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    pickle.dump(entire_plots, open("{}/results.pickle".format(env_dir), 'wb'))

do_context = True
# With context

entire_plots_context = {"sigma": sigma_env_n}
if do_context:
    # 6) UCB
    if do_UCB:
        name = "UCB"
        plots_for_sigma = train_context(learner_constructor=UCB_Learner)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 7) TS
    if do_TS:
        name = "TS"
        plots_for_sigma = train_context(learner_constructor=TS_Learner)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 8) UCB with window
    if do_UCB_wdw:
        name = "UCB_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=UCB_Learner, window_length=window_length)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 9) TS with window
    if do_TS_wdw:
        name = "TS_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=TS_Learner, window_length=window_length)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    pickle.dump(entire_plots_context, open("{}/results_context.pickle".format(env_dir), 'wb'))

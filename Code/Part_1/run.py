import math

import numpy as np
import scipy.stats as stats
import functools

from Code.Part_1.ContextLearner import ContextLearner
from Code.Part_1.ProjectEnvironment import ProjectEnvironment
from Code.learners.ContinuousTS_Learner import TS_Learner
from Code.learners.SequentialABLearner import SequentialABLearner

import pickle

from Code.learners.UCB_Learner import UCB_Learner

env_dir = ProjectEnvironment.get_env_dir()

n_users = 3
sigma_env_n = [0.1]#, 2, 3]

season_length = 91
number_of_seasons = 4
T = 363
context_change_period = 14
batch_size = 192

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

coeff_user_season = [
    [0.001, -1, 0.5, 0.1],
    [0.001, -1, 0.3, 0.1],
    [0.001, -1,  0.75, 0.1]
]
q_user_season = [
    [5, 180, 20, 300],
    [30, 180, 10, 150],
    [65, 180, 45, 50]
]

matrix_parameters_u1 = [
    [0.097, 1000, [], functools.partial(gauss, 175, 5)],
    [0.0025, 20, [], functools.partial(gauss, 175, 5)],
    [0.005, 25, [], functools.partial(gauss, 175, 5)],
    [0.009, 15, [], functools.partial(gauss, 175, 5)],
        ]

matrix_parameters_u2 = [
    [0.10837, 20, [], functools.partial(gauss, 175, 5)],
    [0.003, 20, [], functools.partial(gauss, 175, 5)],
    [0.002, 15, [], functools.partial(gauss, 175, 5)],
    [0.004, 20, [], functools.partial(gauss, 175, 5)],
        ]

matrix_parameters_u3 = [
    [0.00036, 20, [], functools.partial(gauss, 175, 5)],
    [0.009, 25, [], functools.partial(gauss, 175, 5)],
    [0.004, 10, [], functools.partial(gauss, 175, 5)],
    [0.007, 15, [], functools.partial(gauss, 175, 5)],
]

context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]

users_matrix_parameters = [
    matrix_parameters_u1,
    matrix_parameters_u2,
    matrix_parameters_u3
]

min_price = 25
max_price = 1000
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

coeff_window_length = 2
window_length = int(coeff_window_length * math.pow(T, 0.5))
do_UCB_wdw = False
do_TS_wdw = True
plot_env = False
plot_single_user = False
plot_context = True

def num_users(idx, t):
    t_in_season, season = (t % season_length) + 1, int((t % 365) / season_length)
    q = q_user_season[idx][season]
    m = coeff_user_season[idx][season]
    return q + m * t_in_season

num_users_functions = [functools.partial(num_users, i) for i in range(n_users)]

def train_context(learner_constructor, context_alternatives, window_length=None):
    plots_for_sigma = []
    for sigma in sigma_env_n:
        c_learners = []
        for idx_c, alternative in enumerate(context_alternatives):
            c_learners.append(ContextLearner(alternative, learner_constructor, arms, batch_size, window_length, idx_c, sigma))
        best_context = 0
        history_best_contexts = [best_context]
        history_results_each_contexts = [[] for _ in context_alternatives] # for each context saving sums
        env = ProjectEnvironment(arms, num_users_functions, sigma, users_matrix_parameters, context_alternatives, batch_size=batch_size)
        if plot_env:
            env.plot(T=T)
        if plot_single_user:
            env.plot_single_users(True, T, idx_context=0)
            env.plot_single_users(True, T, idx_context=1)
            env.plot_single_users(True, T, idx_context=2)
            env.plot_single_users(True, T, idx_context=3)
            env.plot_single_users(True, T, idx_context=4)
            err
        if plot_context:
            """
            env.plot_context(0, T)
            env.plot_context(1, T)
            env.plot_context(2, T)
            env.plot_context(3, T)
            env.plot_context(4, T)
            """
            env.plot_contexts()
        for t in range(1, T+1):
            # take set of rewards for each context. This allows to call round_context only once.
            # From round_context, rewards and demands are returned for each context, and are the set of rewards and demands
            # for each context and for each arm.
            # Users is instead the set drawn users. To simplify we return a matrix. For each context we have the same set of drawn.
            # Subcontexts is instead the associated set of subcontexts so to access the correct learner, as user u refers to different subcontexts
            rewards_per_arm, demands_per_arm, subcontexts, users = env.round_context(t)
            for idx_c, (c_learner, rewards, subcontext, demands) in enumerate(zip(c_learners, rewards_per_arm, subcontexts, demands_per_arm)):
                rewards, demands = np.array(rewards).reshape(batch_size, -1), np.array(demands).reshape(batch_size, -1)
                idxs_arm_c_learner, _ = c_learner.pull_arm(t, rewards, demands, np.array(users))
                if idx_c == best_context:
                    pulled_arms_current_best_context = idxs_arm_c_learner
            # once all learners are updated, we update rewards for the current context
            env.round_for_arm(pulled_arms_current_best_context, t, users)

            if t % context_change_period == context_change_period - 1:
                results_alternatives = []
                for idx_c, context in enumerate(context_alternatives):
                    learners = c_learners[idx_c]
                    sum = 0
                    somma_prob = 0
                    for idx_s, subcontext in enumerate(context):
                        learner = learners.get_learner(idx_s)
                        is_all = learners.getNumberLearners() == 1
                        mu, prob = min(learner.avg_bounds_fixed(alpha)), learner.prob_lower_bound_fixed(alpha, batch_size)
                        if is_all:
                            sum += mu
                        else:
                            sum += prob * mu
                        somma_prob += prob
                    print("Alternative {}: prob = {}, sum = {}".format(idx_c, somma_prob, sum))
                    results_alternatives.append(sum)
                    history_results_each_contexts[idx_c].append(sum)
                best_context = np.argmax(results_alternatives)
                print("Season: {}, Time: {}, Best alternative: {}".format(int((t % 365) / season_length), (t+1) % season_length, best_context))
                env.set_context(best_context)
                history_best_contexts.append(best_context)
                # c_learners[best_context].plot(env)
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

single_context_alternatives = [[[0, 1, 2]]]
single_best_context = 0

# 1) Sequential AB testing
if do_without_context:
    if do_sequential_AB:
        name = "AB"
        plots_for_sigma = train_context(learner_constructor=SequentialABLearner, context_alternatives=single_context_alternatives)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 2) UCB
    if do_UCB:
        name = "UCB"
        plots_for_sigma = train_context(learner_constructor=UCB_Learner, context_alternatives=single_context_alternatives)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 3) TS
    if do_TS:
        name = "TS"
        plots_for_sigma = train_context(learner_constructor=TS_Learner, context_alternatives=single_context_alternatives)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 4) UCB with window
    if do_UCB_wdw:
        name = "UCB_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=UCB_Learner, context_alternatives=single_context_alternatives, window_length=window_length)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    # 5) TS with window
    if do_TS_wdw:
        name = "TS_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=TS_Learner, context_alternatives=single_context_alternatives, window_length=window_length)
        entire_plots[name] = [sigma_env_n, plots_for_sigma]

    pickle.dump(entire_plots, open("{}/results.pickle".format(env_dir), 'wb'))

do_with_context = True
# With context

entire_plots_context = {"sigma": sigma_env_n}
if do_with_context:
    # 6) UCB
    if do_UCB:
        name = "UCB"
        plots_for_sigma = train_context(learner_constructor=UCB_Learner, context_alternatives=context_alternatives)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 7) TS
    if do_TS:
        name = "TS"
        plots_for_sigma = train_context(learner_constructor=TS_Learner, context_alternatives=context_alternatives)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 8) UCB with window
    if do_UCB_wdw:
        name = "UCB_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=UCB_Learner, context_alternatives=context_alternatives, window_length=window_length)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    # 9) TS with window
    if do_TS_wdw:
        name = "TS_{}".format(window_length)
        plots_for_sigma = train_context(learner_constructor=TS_Learner, context_alternatives=context_alternatives, window_length=window_length)
        entire_plots_context[name] = [sigma_env_n, plots_for_sigma]

    pickle.dump(entire_plots_context, open("{}/results_context.pickle".format(env_dir), 'wb'))

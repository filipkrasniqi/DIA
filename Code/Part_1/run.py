import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.ContextLearner import ContextLearner
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

context_alternatives = [[[0], [1], [2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0, 1, 2]]]
# sbagliato ma funge: context_alternatives = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]

context_matrix_parameters = [
    [
        [[0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear]],
        [[0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear]],
        [[0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear], [0.1, 10, linear]]
    ],
    [
        [[0.1, 400, linear], [0.1, 400, linear], [0.1, 400, linear], [0.1, 400, linear]],
        [[0.1, 400, linear], [0.1, 400, linear], [0.1, 400, linear], [0.1, 400, linear]]
    ],
    [
        [[0.1, 300, linear], [0.1, 300, linear], [0.1, 300, linear], [0.1, 300, linear]],
        [[0.1, 300, linear], [0.1, 300, linear], [0.1, 300, linear], [0.1, 300, linear]]
    ],
    [
        [[0.1, 200, linear], [0.1, 200, linear], [0.1, 200, linear], [0.1, 200, linear]],
        [[0.1, 200, linear], [0.1, 200, linear], [0.1, 200, linear], [0.1, 200, linear]]
    ],
    [
        [[0.1, 1, linear], [0.1, 1, linear], [0.1, 1, linear], [0.1, 1, linear]]
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


def train(learner):
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, matrix_parameters)
        for t in range(T):
            learner.pull_arm(env, t)
        learner.plot(env)


def train_context(learner_constructor, arms, window_length=None):
    c_learners = []
    best_context = 0
    for alternative in context_alternatives:
        c_learners.append(ContextLearner(alternative, learner_constructor, arms, window_length))
    for sigma in sigma_env_n:
        env = Environment(arms, probabilities, sigma, context_matrix_parameters, context_alternatives)
        env.plot()
        for t in range(T):
            if t % 33 == 32:
                results_alternatives = []
                for idx_c, context in enumerate(context_alternatives):
                    learners = c_learners[idx_c]
                    print("\n\nContext {}\n\n".format(idx_c))
                    sum = 0
                    for idx_s, subcontext in enumerate(context):
                        learner = learners.get_learner(idx_s)
                        is_all = len(context) == 1
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
                env.plot()
            subcontext, user = env.sample_subcontext()
            rewards_per_arm, _ = env.round(t, subcontext)
            idx_arm_current_clearner = 0
            for idx_c, c_learner in enumerate(c_learners):
                idx_arm = c_learner.pull_arm(t, rewards_per_arm, user)
                if idx_c == best_context:
                    idx_arm_current_clearner = idx_arm
            # ora lo faccio sul serio!
            env.round_for_arm(arms[idx_arm_current_clearner], t, subcontext)
        learner.plot(env)


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

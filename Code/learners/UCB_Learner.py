import numpy as np
import math

from Code.learners.Learner import Learner

'''
UCB learner. Selects arm by creating bounds of range 
(y_ - 1.96 * sigma, y_ + 1.96 * sigma) and taking the arm s.t.
upper bound is higher. Sigma depends also on number of samples.
'''


class UCB_Learner(Learner):
    def __init__(self, arms, idx_c, idx_s, sigma, window=None):
        name_learner = "UCB learner"
        if window is not None:
            name_learner += " with window {}".format(window)
        super().__init__(arms=arms, window=window, idx_c=idx_c, idx_s=idx_s, name_learner=name_learner, sigma=sigma)

        self.average_rewards = np.zeros(self.n_arms)
        self.delta = np.zeros(self.n_arms)

    '''
    Pulls arm selecting arm s.t. upper bound is higher.
    '''

    def pull_arm(self, rewards_per_arm, demands_per_arm, user, t):
        if self.t < self.n_arms:
            idx_arm = self.t
        else:
            idx_arm = np.argmax(self.average_rewards + self.delta)
        return Learner.pull_arm(self, rewards_per_arm, demands_per_arm, user, t, idx_arm)

    '''
    Updates Learner observation + average rewards and delta 
    '''

    def update(self, pulled_arm, reward, demand, user):
        self.update_observations(pulled_arm, reward, demand, user)

        tot_n = len(self.rewards_per_arm[pulled_arm])
        if self.window is None or self.window > tot_n:
            n = tot_n
        else:
            n = self.window
        idxs = list(range(tot_n - n, tot_n))

        self.average_rewards[pulled_arm] = np.sum(
            [r for i, r in enumerate(self.rewards_per_arm[pulled_arm]) if i in idxs]) / n
        self.delta[pulled_arm] = math.sqrt(2 * math.log(self.t) / n)

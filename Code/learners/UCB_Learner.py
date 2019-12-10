import numpy as np
import math

from Code.learners.Learner import Learner

'''
UCB learner. Selects arm by creating bounds of range 
(y_ - 1.96 * sigma, y_ + 1.96 * sigma) and taking the arm s.t.
upper bound is higher. Sigma depends also on number of samples.
'''


class UCB_Learner(Learner):
    def __init__(self, arms, idx_c, idx_s, sigma, batch_size, window=None):
        name_learner = "UCB learner"
        if window is not None:
            name_learner += " with window {}".format(window)
        super().__init__(arms=arms, batch_size=batch_size, window=window, idx_c=idx_c, idx_s=idx_s, name_learner=name_learner, sigma=sigma)

        self.average_rewards = np.array([float("+inf") for _ in self.arms])# np.zeros(self.n_arms)
        self.delta = np.zeros(self.n_arms)

    """
    Best arm is the one having higher upper bound
    """
    def best_arm(self):
        return np.argmax(self.average_rewards + self.delta)

    '''
    Updates Learner observation + average rewards and delta 
    '''

    def update(self, idx_arm):
        tot_n = len(self.rewards_per_arm[idx_arm])
        N = min(self.compute_current_N(), tot_n)
        windowed_rewards = self.rewards_per_arm[idx_arm][-N:]

        self.average_rewards[idx_arm] = np.sum(windowed_rewards) / N
        self.delta[idx_arm] = math.sqrt(2 * math.log(self.t) / N)
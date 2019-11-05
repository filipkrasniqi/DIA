import numpy as np
import math

from Code.Part_1.Learner import Learner


class UCB_Learner(Learner):
    def __init__(self, arms, window=None, idx_c = -1, idx_s = -1):
        super().__init__(arms, window=window, idx_c=idx_c, idx_s=idx_s)
        # Upper Bound = AVG reward + Delta
        self.average_rewards = np.zeros(self.n_arms)
        self.delta = np.zeros(self.n_arms)

    def pull_arm(self, rewards_per_arm, user, t):
        if (self.t < self.n_arms):
            idx_arm = self.t
        else:
            idx_arm = np.argmax(self.average_rewards + self.delta)
        return Learner.pull_arm(self, rewards_per_arm, user, t, idx_arm)

    def update(self, pulled_arm, reward, user):
        self.t += 1
        self.update_observations(pulled_arm, reward, user)

        tot_n = len(self.rewards_per_arm[pulled_arm])
        if self.window is None or self.window > tot_n:
            n = tot_n
        else:
            n = self.window
        idxs = list(range(tot_n - n, tot_n))

        self.average_rewards[pulled_arm] = np.sum(
            [r for i, r in enumerate(self.rewards_per_arm[pulled_arm]) if i in idxs]) / n
        self.delta[pulled_arm] = math.sqrt(2 * math.log(self.t) / n)

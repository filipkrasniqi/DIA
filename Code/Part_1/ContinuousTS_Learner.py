import numpy as np

from Code.Part_1.Learner import Learner


class TS_Learner(Learner):
    def __init__(self, arms, window=None, idx_c=-1, idx_s=-1):
        super().__init__(arms, window=window, idx_c=idx_c, idx_s=idx_s)
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.rewards_variance = [float("+inf") for _ in range(self.n_arms)]

    def pull_arm(self, rewards_per_arm, user, t):
        idx_arm = np.argmax(np.random.normal(self.average_rewards, np.power(self.rewards_variance, 0.5)))
        return Learner.pull_arm(self, rewards_per_arm, user, t, idx_arm)

    def update(self, pulled_arm, reward, user):
        self.t += 1
        self.update_observations(pulled_arm, reward, user)
        tot_n = len(self.rewards_per_arm[pulled_arm])
        if self.window is None or self.window > tot_n:
            n = int(tot_n)
        else:
            n = int(self.window)
        idxs = list(range(tot_n - n, tot_n))
        self.average_rewards[pulled_arm] = np.sum(
            [r for i, r in enumerate(self.rewards_per_arm[pulled_arm]) if i in idxs]) / n
        self.rewards_variance[pulled_arm] = np.sum(
            [(reward - self.average_rewards[pulled_arm]) ** 2 for i, reward in
             enumerate(self.rewards_per_arm[pulled_arm]) if i in idxs]) / n - 1

    def get_arm_mean(self, arm):
        return self.average_rewards[arm]

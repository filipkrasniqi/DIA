import numpy as np

from Code.learners.Learner import Learner

'''
Thompson-Sampling learner. Each arm is associated
to a Gaussian distribution, on which we estimate mean and std.
'''


class TS_Learner(Learner):
    def __init__(self, arms, idx_c, idx_s, sigma, window=None):
        name_learner = "Thompson-Sampling"
        if window is not None:
            name_learner += " with window {}".format(window)
        super().__init__(arms, window=window, idx_c=idx_c, idx_s=idx_s, name_learner=name_learner, sigma=sigma)
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.rewards_variance = [float("+inf") for _ in range(self.n_arms)]

    '''
    Pull arm s.t. sampling from the Gaussian is higher.
    '''

    def pull_arm(self, rewards_per_arm, demands_per_arm, user, t):
        idx_arm = np.argmax(np.random.normal(self.average_rewards, np.power(self.rewards_variance, 0.5)))
        return Learner.pull_arm(self, rewards_per_arm, demands_per_arm, user, t, idx_arm)

    '''
    Updates Learner observations + own rewards mean and std
    '''

    def update(self, pulled_arm, reward, demand, user):
        self.update_observations(pulled_arm, reward, demand, user)
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
             enumerate(self.rewards_per_arm[pulled_arm]) if i in idxs]) / (n - 1)

    '''
    Returns mean given arm
    '''

    def get_arm_mean(self, arm):
        return self.average_rewards[arm]

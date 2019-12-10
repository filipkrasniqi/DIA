import numpy as np

from Code.learners.Learner import Learner

'''
Thompson-Sampling learner. Each arm is associated
to a Gaussian distribution, on which we estimate mean and std.
'''


class TS_Learner(Learner):
    def __init__(self, arms, idx_c, idx_s, sigma, batch_size, window=None):
        name_learner = "Thompson-Sampling"
        if window is not None:
            name_learner += " with window {}".format(window)
        super().__init__(arms, batch_size=batch_size, window=window, idx_c=idx_c, idx_s=idx_s, name_learner=name_learner, sigma=sigma)
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.std_reward = [float("+inf") for _ in range(self.n_arms)]

    """
    Best arm is the one that from the drawings happens to be the best
    """
    def best_arm(self):
        return np.argmax(np.random.normal(self.average_rewards, self.std_reward))

    '''
    Updates Learner observations + own rewards mean and std
    '''
    def update(self, idx_arm):
        tot_n = len(self.rewards_per_arm[idx_arm])
        N = min(self.compute_current_N(), tot_n)
        windowed_rewards = self.rewards_per_arm[idx_arm][-N:]
        self.average_rewards[idx_arm] = np.mean(windowed_rewards)
        self.std_reward[idx_arm] = np.std(windowed_rewards)

    '''
    Returns mean given arm
    '''
    def get_arm_mean(self, arm):
        return self.average_rewards[arm]

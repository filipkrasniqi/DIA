import math

import numpy as np

from Code.Part_1.Learner import Learner


class SequentialABLearner(Learner):

    def __init__(self, n_arms, n_users=3):
        Learner.__init__(self, n_arms)
        self.user_samples = [0 for _ in range(n_users)]  # samples drawn for each user
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.rewards_variance = [0 for _ in range(self.n_arms)]

    def update(self, pulled_arm, reward, user):
        self.t += 1
        self.user_samples[user] += 1
        self.update_observations(pulled_arm, reward)
        n = len(self.rewards_per_arm[pulled_arm])
        self.average_rewards[pulled_arm] = np.sum(self.rewards_per_arm[pulled_arm]) / n
        self.rewards_variance[pulled_arm] = np.sum([(reward - self.average_rewards[pulled_arm]) ** 2 for reward in self.rewards_per_arm[pulled_arm]]) / n

    def get_arm_mean(self, arm):
        return self.average_rewards[arm]

    def z(self, arm1, arm2):
        n1, n2, m1, m2 = self.num_samples(arm1), self.num_samples(arm2), self.get_arm_mean(arm1), self.get_arm_mean(
            arm2)
        return (m1 - m2) / (math.pow((self.rewards_variance[arm1] / n1 + self.rewards_variance[arm2] / n2), 0.5))
from Learner.Learner import *
import numpy as np
import math

class UCB_Learner(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        # Upper Bound = AVG reward + Delta
        self.average_rewards = np.zeros(n_arms)
        self.delta = np.zeros(n_arms)

    def pull_arm(self):
        if (self.t < self.n_arms):
            return self.t

        arm =  np.argmax(self.average_rewards + self.delta)
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.average_rewards[pulled_arm] = np.sum (self.rewards_per_arm[pulled_arm]) / len(self.rewards_per_arm[pulled_arm])
        self.delta[pulled_arm] = math.sqrt(2 * math.log(self.t) / len(self.rewards_per_arm[pulled_arm] ))
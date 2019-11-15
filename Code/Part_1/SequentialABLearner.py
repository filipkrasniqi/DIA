import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.Learner import Learner

'''
Considers sequentially arms and selects the best choice.
Exploration lasts t_start_exploit, while min_confidence is
the confidence level s.t. H0 is accepted.
average_rewards, rewards_variance are needed to compute the normalization,
and are related to each arm.
'''
class SequentialABLearner(Learner):

    def __init__(self, arms, idx_c, idx_s, sigma, window=None):
        name_learner = "Sequential AB"
        if window is not None:
            name_learner += " with window {}".format(window)
        Learner.__init__(self, arms, idx_c=idx_c, idx_s=idx_s, name_learner=name_learner, sigma = sigma)
        self.t_start_exploit = 14
        self.min_confidence = 0.95
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.rewards_variance = [0 for _ in range(self.n_arms)]
    '''
    Update values of Learner (update observations) and average + variance
    '''
    def update(self, pulled_arm, reward, demand, user):
        self.update_observations(pulled_arm, reward, demand, user)
        n = len(self.rewards_per_arm[pulled_arm])
        self.average_rewards[pulled_arm] = np.sum(self.rewards_per_arm[pulled_arm]) / n
        self.rewards_variance[pulled_arm] = np.sum([(reward - self.average_rewards[pulled_arm]) ** 2 for reward in self.rewards_per_arm[pulled_arm]]) / n
    '''
    Returns average values for arm
    '''
    def get_arm_mean(self, arm):
        return self.average_rewards[arm]
    '''
    Weighted variance given single users.
    Useful when estimating needed number of samples
    '''
    def get_variance(self):
        weighted_variance = 0
        for idx_u, num_samples in enumerate(self.user_samples):
            # get collected rewards for that and compute variance
            rewards_user = self.get_collected_rewards_user(idx_u)
            mean_rewards_user = np.mean(rewards_user)
            weighted_variance += (self.user_samples[idx_u] / np.sum(self.user_samples)) * (np.sum([(reward - mean_rewards_user) ** 2 for reward in rewards_user]) / self.user_samples[idx_u])
        return weighted_variance
    '''
    Compute z-norm
    '''
    def z(self, arm1, arm2):
        n1, n2, m1, m2 = self.num_samples(arm1), self.num_samples(arm2), self.get_arm_mean(arm1), self.get_arm_mean(
            arm2)
        return (m1 - m2) / (math.pow((self.rewards_variance[arm1] / n1 + self.rewards_variance[arm2] / n2), 0.5))
    '''
    Selection of arm in AB learner: if t is before the threshold, random choice, otherwise pulling so far best solution
    '''
    def pull_arm(self, rewards_per_arm, demands_per_arm, user, t):
        if t < self.t_start_exploit:
            idx_arm = np.random.choice(list(range(self.n_arms)), 1)[0]
        else:
            idx_arm = self.best_candidate()
        return Learner.pull_arm(self, rewards_per_arm, demands_per_arm, user, t, idx_arm)
    '''
    Finds best candidate between the two choices
    '''
    def best_candidate(self, min_confidence = None):
        if min_confidence is None:
            min_confidence = self.min_confidence
        best_candidate = 0
        for alternative in range(1, self.n_arms):
            Z = self.z(best_candidate, alternative)
            p_value = stats.norm.sf(Z)
            if 1 - p_value <= min_confidence:
                best_candidate = alternative
        return best_candidate

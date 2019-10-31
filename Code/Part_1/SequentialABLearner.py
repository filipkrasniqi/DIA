import math

import numpy as np
import scipy.stats as stats

from Code.Part_1.Learner import Learner
import matplotlib.pyplot as plt


class SequentialABLearner(Learner):

    def __init__(self, arms):
        Learner.__init__(self, arms)
        self.average_rewards = [0 for _ in range(self.n_arms)]
        self.rewards_variance = [0 for _ in range(self.n_arms)]

    def update(self, pulled_arm, reward, user):
        self.t += 1
        self.update_observations(pulled_arm, reward, user)
        n = len(self.rewards_per_arm[pulled_arm])
        self.average_rewards[pulled_arm] = np.sum(self.rewards_per_arm[pulled_arm]) / n
        self.rewards_variance[pulled_arm] = np.sum([(reward - self.average_rewards[pulled_arm]) ** 2 for reward in self.rewards_per_arm[pulled_arm]]) / n

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

    def z(self, arm1, arm2):
        n1, n2, m1, m2 = self.num_samples(arm1), self.num_samples(arm2), self.get_arm_mean(arm1), self.get_arm_mean(
            arm2)
        return (m1 - m2) / (math.pow((self.rewards_variance[arm1] / n1 + self.rewards_variance[arm2] / n2), 0.5))

    def plot(self, env):
        # plot regret, i.e., for t in T -> optimum - real_reward(t)
        regret_history = env.regret
        # plot cumulative regret, i.e., just sum the one before
        cumulative_regret_history = []
        cum_regret = 0
        for r in regret_history:
            cum_regret += r
            cumulative_regret_history.append(cum_regret)
        # plot real reward(t)
        real_rewards = env.real_rewards
        print(len(regret_history), len(cumulative_regret_history), len(real_rewards))
        x = list(range(len(real_rewards)))

        plt.plot(x, real_rewards)
        plt.title("Reward over time")
        plt.xlabel("t")
        plt.xlim((0, np.max(x)))
        plt.ylim(0, np.max(real_rewards))
        plt.ylabel("Reward")
        plt.show()

        plt.plot(x, regret_history)
        plt.title("Regret over time")
        plt.xlabel("t")
        plt.xlim((0, np.max(x)))
        plt.ylim(0, np.max(regret_history))
        plt.ylabel("Regret")
        plt.show()

        plt.plot(x, cumulative_regret_history)
        plt.title("Cumulative regret over time")
        plt.xlabel("t")
        plt.xlim((0, np.max(x)))
        plt.ylim(0, np.max(cumulative_regret_history))
        plt.ylabel("Cumulative regret")
        plt.show()

    def pull_arm(self, env, t, idx_arm = None):
        if idx_arm is None:
            idx_arm = np.random.choice(list(range(self.n_arms)), 1)[0]
        arm = self.arms[idx_arm]
        reward, user = env.round(arm, t)
        self.update(idx_arm, reward, user)

    def best_candidate(self, min_confidence = 0.95):
        '''
        variance = learner.get_variance()
        # TODO why to compute n_candidates?
        #  sigma for now is weighted variance for users (discussed with Dom)
        #  Understand the need to compute it, as we already have the number of samples
        #  I think it is just to fix the delta, but in our case it may be even different!
        n_samples = int(((z_a+z_b) ** 2 * variance) / delta ** 2)
        '''

        best_candidate = 0
        for alternative in range(1, self.n_arms):
            Z = self.z(best_candidate, alternative)
            p_value = stats.norm.sf(Z)
            if 1 - p_value <= min_confidence:
                best_candidate = alternative
                # TODO current implementation: take alternative
                #  commented solution:compute alternative test
                #  TODO ensure that, in this case, I need to do that or whether I should just take the alternative, i.e., best_candidate = alternative
                '''
                Z = self.z(alternative, best_candidate)
                p_value = stats.norm.sf(Z)
                

                if 1 - p_value <= min_confidence:
                    print("mierda")
                    raise ValueError("Test scannellato: ne uno ne l'altro")
                else:
                    best_candidate = alternative
                '''
            # else: # h1 is true, i.e., candidate_1 = candidate_h0 is the best

        return best_candidate

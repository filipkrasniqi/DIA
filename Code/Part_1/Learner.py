from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class Learner():

    def __init__(self,arms, n_users = 3):
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.drawn_user = np.array([])
        self.user_samples = [0 for _ in range(n_users)]  # samples drawn for each user

    def update_observations(self,pulled_arm,reward, user):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.user_samples[user] += 1
        self.collected_rewards = np.append(self.collected_rewards,reward)
        self.drawn_user = np.append(self.drawn_user, user)

    def get_collected_rewards_user(self, user):
        indices = [i for i, u in enumerate(self.drawn_user) if u == user]
        return self.collected_rewards[indices]

    def num_samples(self, arm):
        return len(self.rewards_per_arm[arm])

    def pull_arm(self, env, t, idx_arm = None):
        if idx_arm is None:
            raise ValueError("No arm specified")
        arm = self.arms[idx_arm]
        reward, user = env.round(arm, t)
        self.update(idx_arm, reward, user)
        return idx_arm

    @staticmethod
    def plot(env):
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

    @abstractmethod
    def update(self, pulled_arm, reward, user):
        pass


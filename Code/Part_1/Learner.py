from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class Learner:

    def __init__(self, arms, n_users=3, window=None, idx_c = -1, idx_s = -1):
        self.window = window
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.drawn_user = np.array([])
        self.user_samples = [0 for _ in range(n_users)]  # samples drawn for each user
        self.idx_c = idx_c
        self.idx_s = idx_s

    def update_observations(self, pulled_arm, reward, user):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.user_samples[user] += 1
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.drawn_user = np.append(self.drawn_user, user)

    def get_collected_rewards_user(self, user):
        indices = [i for i, u in enumerate(self.drawn_user) if u == user]
        return self.collected_rewards[indices]

    def num_samples(self, arm):
        return len(self.rewards_per_arm[arm])

    def avg_bounds(self, users, alpha):
        N = np.sum([self.user_samples[user] for user in users])  # / tot_num_samples
        rewards_user = [r for u, r in zip(self.drawn_user, self.collected_rewards) if u in users]
        mu, std = np.mean(rewards_user), np.std(rewards_user)
        t_dist = np.random.standard_t(N - 1)
        quantile = np.quantile(t_dist, 1 - alpha)
        delta = quantile * std / np.power(N, 0.5)
        return mu - delta, mu + delta

    def avg_bounds_fixed(self, alpha):
        N = np.sum(self.user_samples)  # / tot_num_samples
        rewards_user = [r for u, r in zip(self.drawn_user, self.collected_rewards)]
        mu, std = np.mean(rewards_user), np.std(rewards_user)
        t_dist = np.random.standard_t(N - 1)
        quantile = np.quantile(t_dist, 1 - alpha/2)
        delta = quantile * std / np.power(N, 0.5)
        return mu - delta, mu + delta

    # Computed with approach explained here: http://math.mit.edu/~goemans/18310S15/chernoff-notes.pdf
    # i.e., Chernoff bound for bernoulli distribution
    def prob_lower_bound(self, users, alpha):
        tot_num_samples = np.sum(self.user_samples)
        num_user = np.sum([self.user_samples[user] for user in users])  # / tot_num_samples
        delta = np.power(np.log10(1 / alpha) * 2 / num_user, 0.5)
        return ((1 - delta) * num_user) / tot_num_samples
        '''
        TODO implementazione con modulo (vedi pdf)
        delta = np.power(np.log(2 / alpha) * 3 / num_user, 0.5)
        return ((1 - delta) * num_user) / tot_num_samples
        '''

    def prob_lower_bound_fixed(self, alpha):
        num_user = np.sum(self.user_samples)  # / tot_num_samples
        delta = np.power(np.log10(1 / alpha) * 2 / num_user, 0.5)
        return (1 - delta) * num_user

    def pull_arm(self, rewards_per_arm, user, t, idx_arm=None):
        if idx_arm is None:
            raise ValueError("No arm specified")
        reward = rewards_per_arm[idx_arm]
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

    @abstractmethod
    def get_best_arm(self):
        pass

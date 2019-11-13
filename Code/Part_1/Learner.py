import os
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import pathlib

import seaborn as sns

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
env_dir = outputs_dir+"v01/"

class Learner:

    def __init__(self, arms, n_users=3, window=None, idx_c = -1, idx_s = -1, name_learner = None, sigma = None):
        if name_learner is None:
            raise ValueError("Pls name me")
        self.sigma = sigma
        self.name_learner = name_learner
        self.window = window
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.drawn_user = np.array([])
        self.user_samples = [0 for _ in range(n_users)]  # number of samples drawn for each user
        self.idx_c = idx_c
        self.idx_s = idx_s
        self.samples_user = [([],[]) for _ in range(n_users)]   # values of the samples drawn for each user

    def update_observations(self, pulled_arm, reward, demand, user):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.user_samples[user] += 1
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.drawn_user = np.append(self.drawn_user, user)
        self.samples_user[user][0].append(pulled_arm)
        self.samples_user[user][1].append(demand)

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
        N = np.sum(self.user_samples)
        rewards_user = [r for u, r in zip(self.drawn_user, self.collected_rewards)]
        mu, std = np.mean(rewards_user), np.std(rewards_user)
        delta = self.bounds(std, N, alpha)
        return mu - delta, mu + delta

    @staticmethod
    def bounds(std, N, alpha):
        return np.quantile(np.random.standard_t(N - 1), 1 - alpha/2) * std / np.power(N, 0.5)

    # Computed with approach explained here: http://math.mit.edu/~goemans/18310S15/chernoff-notes.pdf
    # i.e., Chernoff bound for bernoulli distribution
    def prob_lower_bound(self, users, alpha):
        tot_num_samples = np.sum(self.user_samples)
        num_user = np.sum([self.user_samples[user] for user in users])  # / tot_num_samples
        delta = np.power(np.log10(1 / alpha) * 2 / num_user, 0.5)
        return ((1 - delta) * num_user) / tot_num_samples

    def prob_lower_bound_fixed(self, alpha):
        num_user = np.sum(self.user_samples)  # / tot_num_samples
        delta = np.power(np.log10(1 / alpha) * 2 / num_user, 0.5)
        return (1 - delta) * num_user

    def pull_arm(self, rewards_per_arm, demands_per_arm, user, t, idx_arm):
        reward, demand = rewards_per_arm[idx_arm], demands_per_arm[idx_arm]
        self.update(idx_arm, reward, demand, user)
        return idx_arm

    def plot(self, env):
        curr_dir_env = "{}/{}/learner_{}/{}_{}/".format(env_dir, self.sigma, self.name_learner, self.idx_c, self.idx_s)
        pathlib.Path(curr_dir_env).mkdir(parents=True, exist_ok=True)
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

        ax = sns.lineplot(x=x, y=real_rewards, markers=True, label='Reward')
        ax.legend()
        ax = sns.lineplot(x=x, y=regret_history, markers=True, label='Regret')
        ax.legend()
        ax.set_title("{}: evolution of reward and regret".format(self.name_learner))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value in $")

        plt.plot(x, cumulative_regret_history)
        plt.title("{} - sigma {}: Cumulative regret over time".format(self.name_learner, self.sigma))
        plt.xlabel("t")
        plt.xlim((0, np.max(x)))
        plt.ylim(0, np.max(cumulative_regret_history))
        plt.ylabel("Cumulative regret")
        plt.show()
        plt.savefig("{}cumulative_regret.png".format(curr_dir_env))

        # print functions
        demand_mapping = {}
        for user, (x, y) in enumerate(self.samples_user):
            demand_mapping[user] = {}
            for (x_val, y_val) in zip(x, y):
                if x_val not in demand_mapping[user].keys():
                    demand_mapping[user][x_val] = []
                demand_mapping[user][x_val].append(y_val)

        for user in demand_mapping.keys():
            lower_bounds, avg_demand, upper_bounds = [], [], []
            for idx_arm, arm in enumerate(self.arms):
                y = demand_mapping[user][idx_arm]
                mu, std = np.mean(y), np.std(y)
                delta = self.bounds(std, len(y), alpha=0.95)
                lower_bounds.append(mu - delta)
                avg_demand.append(mu)
                upper_bounds.append(mu + delta)

            plt.plot(self.arms, avg_demand)

            plt.fill(np.concatenate([self.arms, self.arms[::-1]]),
                     np.concatenate([lower_bounds, upper_bounds[::-1]]),
                     alpha=0.5, fc='b', ec='None', label='95% conf interval')

            plt.title("Predicted function for context {}, subcontext {} - sigma {}".format(self.idx_c, self.idx_s, self.sigma))
            plt.xlabel("t")
            plt.xlim((0, np.max(self.arms)))
            plt.ylim(0, np.max(upper_bounds))
            plt.ylabel("Demand")
            plt.show()

            plt.savefig("{}demand_{}.png".format(curr_dir_env, user))

        return x, real_rewards, regret_history, cumulative_regret_history, (self.idx_c, self.idx_s, demand_mapping)

    @abstractmethod
    def update(self, pulled_arm, reward, demand, user):
        pass

    @abstractmethod
    def get_best_arm(self):
        pass

import os
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import pathlib

import seaborn as sns
import scipy

sns.set(font_scale=2.5)
sns.set(rc={'figure.figsize':(15,9), 'axes.formatter.limits': (-5, 5), 'axes.titlesize': 'medium', 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'})

from Code.Part_1.ProjectEnvironment import ProjectEnvironment

env_dir = ProjectEnvironment.get_env_dir()


class Learner:

    def __init__(self, arms, n_users=3, window=None, idx_c=-1, idx_s=-1, name_learner=None, sigma=None, batch_size = None):
        if name_learner is None:
            raise ValueError("Pls name me")
        self.sigma = sigma
        self.name_learner = name_learner
        self.window = window
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.batch_size = batch_size
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.drawn_user_to_print = np.array([])
        self.drawn_users_per_t = np.array([])
        self.user_samples = [0 for _ in range(n_users)]  # number of samples drawn for each user
        self.idx_c = idx_c
        self.idx_s = idx_s
        self.samples_user = [([], []) for _ in range(n_users)]  # values of the samples drawn for each user
        self.history_drawn_users = np.array([])

    def update_observations(self, pulled_arm, reward, users):
        self.rewards_per_arm[pulled_arm].append(reward)
        # TODO non va bene prendere drawn users, bisogna prenderne la window. Problema: non so a quale t sono associati
        for user in users:
            self.user_samples[user] += 1
            self.drawn_user_to_print = np.append(self.drawn_user_to_print, user)
        self.drawn_users_per_t = np.append(self.drawn_users_per_t, len(users))
        self.collected_rewards = np.append(self.collected_rewards, reward)

        # history drawn users
        try:
            self.history_drawn_users[self.t - 1] += 1
        except:
            self.history_drawn_users = np.append(self.history_drawn_users, 1)
        self.samples_user[user][0].append(pulled_arm)
        # TODO just for log purposes. Removed demand!
        demand = 1
        self.samples_user[user][1].append(demand)

    def num_samples(self, arm):
        return len(self.rewards_per_arm[arm])

    def compute_current_N(self):
        if self.window is None or self.window > self.t:
            n = self.t
        else:
            n = self.window

        return n

    def avg_bounds_fixed(self, alpha):
        t_ = self.t
        if self.window is not None:
            t_ = self.window

        collected_rewards = self.collected_rewards[-t_:]
        mu, std = np.mean(collected_rewards), np.std(collected_rewards)
        delta = self.bounds(std, len(collected_rewards), alpha)
        return mu - delta, mu + delta

    @staticmethod
    def bounds(std, N, alpha):
        value_ppf = (1 + (1 - alpha)) / 2 # 1-alpha/2
        scipy_quantile = scipy.stats.t.ppf([value_ppf], N-1)[0]
        return scipy_quantile * std / np.power(N, 0.5)

    def prob_lower_bound_fixed(self, alpha, batch_size):
        # last M values from self.drawn_user, being M the sum of the last N history_... taken from the batch
        t_ = self.t
        if self.window is not None:
            t_ = self.window
        num_users = np.sum(self.drawn_users_per_t[-t_:])
        delta = np.power(np.log10(1 / alpha) * 2 / num_users, 0.5)
        lower_bound_num_users = (1 - delta) * num_users
        lower_bound_prob = lower_bound_num_users / (t_ * batch_size)
        return lower_bound_prob

    def pull_arm(self, rewards_per_arm, users, t):
        self.t = t
        idx_arm = self.best_arm()
        # reward = np.mean(rewards_per_arm[:, idx_arm])
        reward = rewards_per_arm[idx_arm]
        # TODO should avg the reward for the probability of the current subcontext to happen
        self.update_observations(idx_arm, reward, users)
        self.update(idx_arm)
        return idx_arm

    def best_arm(self):
        pass

    @staticmethod
    def plot_regret_reward(x, real_rewards, regret_history, cumulative_regret_history, name_learner, sigma,
                           curr_dir_env=None, also_cumulative=False):
        all_y_vals = real_rewards + regret_history
        if also_cumulative:
            all_y_vals += cumulative_regret_history
        ax = sns.lineplot(x=x, y=real_rewards, markers=True, label='Reward')
        ax = sns.lineplot(x=x, y=regret_history, markers=True, label='Regret')
        if also_cumulative:
            ax = sns.lineplot(x=x, y=cumulative_regret_history, markers=True, label='C. Regret')
        ax.legend()
        ax.set_title(
            "{} - sigma {}: reward, regret, cumulative regret".format(name_learner, sigma))
        ax.set_xlabel("Time [day]")
        ax.set_ylabel("Value [$]")
        plt.xlim((0, np.max(x) + 1))
        max_val = np.max(all_y_vals)
        delta = max_val / 10
        plt.ylim(0, max_val + delta)
        plt.show()
        postfix = ""
        if also_cumulative:
            postfix = "_cumulative"
        # plt.savefig("{}regret_reward{}.png".format(curr_dir_env, postfix))

        fig = ax.get_figure()
        if curr_dir_env is None:
            name = "temp.png"
        else:
            name = "{}regret_reward{}.png".format(curr_dir_env, postfix)
        fig.savefig(name)

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
        # print(len(regret_history), len(cumulative_regret_history), len(real_rewards))
        x = list(range(len(real_rewards)))
        """
        Learner.plot_regret_reward(x, real_rewards, regret_history, cumulative_regret_history, self.name_learner,
                                   self.sigma, curr_dir_env)
        """
        Learner.plot_regret_reward(x, real_rewards, regret_history, cumulative_regret_history, self.name_learner,
                                   self.sigma, curr_dir_env, also_cumulative=True)
        # print functions
        demand_mapping = []
        for user, (x, y) in enumerate(self.samples_user):
            demand_mapping.append({})
            for (x_val, y_val) in zip(x, y):
                if x_val not in demand_mapping[user].keys():
                    demand_mapping[user][x_val] = []
                demand_mapping[user][x_val].append(y_val)

        # TODO plot è sbagliata. Devo stamparne solo una facendo la media considerando gli utenti associati al subxont4ext corrente,
        # TODO poichè alcuni non hanno valori. Poichè è inutile e serviva solo per debug, commentata
        """
        for user in range(len(demand_mapping)):
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
        """
        return x, real_rewards, regret_history, cumulative_regret_history, (self.idx_c, self.idx_s, demand_mapping), self.drawn_user_to_print

    @abstractmethod
    def update(self, idx_arm):
        pass

    @abstractmethod
    def get_best_arm(self):
        pass

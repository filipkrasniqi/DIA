import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

random.seed(17)

BATCH_SIZE = 100


class User:
    def __init__(self, max_budget, bid, slope, sigma, idx, max_value_subcampaign, click_value_subcampaign):
        self.max_budget = max_budget
        self.bid = bid
        self.slope = slope
        self.sigma = sigma
        self.idx = idx
        self.max_clicks_subcampaign = max_value_subcampaign
        self.click_value_subcampaign = click_value_subcampaign

    def get_clicks_real(self, bid):
        clicks = max(self.max_clicks_subcampaign * (1 - np.exp(-self.slope * (bid - self.bid))), 0)
        return round(clicks * self.click_value_subcampaign, 3)

    def get_clicks_noise(self, bid):
        y = self.get_clicks_real(bid)
        mu, sigma = 0, self.sigma
        noise = np.random.normal(mu, sigma)
        return round(max(0, y + noise), 3)

    def plot(self, idx_subcampaign):
        x = np.linspace(0, self.max_budget, 100)
        y = [self.get_clicks_real(bid) for bid in x]
        plt.plot(x, y)
        plt.title("Subcampaign {}, User {}".format(idx_subcampaign, self.idx))
        plt.xlabel("Daily budget [€]")
        plt.xlim((0, self.max_budget))
        plt.xticks(np.arange(0, self.max_budget + 1, 20))
        plt.ylabel("Number of clicks")
        plt.ylim((0, self.max_clicks_subcampaign + self.max_clicks_subcampaign / 10))
        plt.yticks(np.arange(0, max(y) + max(y) / 10, max(y) / 10))
        plt.grid()
        plt.show()


class Subcampaign:
    def __init__(self, n_arms, n_users, user_probabilities, min_budget, max_budget,
                 sigma, idx, bids, slopes, max_clicks, click_value):
        self.n_arms = n_arms
        self.n_users = n_users
        # One probability for each user.
        self.user_probabilities = user_probabilities
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.max_clicks = max_clicks
        self.bids = bids
        self.slopes = slopes
        self.sigma = sigma
        self.users = list()
        self.idx = idx
        self.click_value = click_value
        self.users_in_batch = [0, 0, 0]
        self.generate()

    def generate(self):
        # Generate a curve for each user.
        for idx_user in range(self.n_users):
            new_user = User(
                max_budget=self.max_budget,
                bid=self.bids[idx_user],
                slope=self.slopes[idx_user],
                sigma=self.sigma,
                idx=idx_user,
                max_value_subcampaign=self.max_clicks[idx_user],
                click_value_subcampaign=self.click_value)
            self.users.append(new_user)

        # Get batch of users.
        self.get_new_batch()

    def get_new_batch(self):
        self.users_in_batch = [0, 0, 0]

        for _ in range(0, BATCH_SIZE):
            # Select user clicking on ads depending on probabilities.
            user_to_sample = len(self.user_probabilities) - 1
            for i, prob in enumerate(self.user_probabilities):
                rand = random.random()
                if rand <= prob:
                    user_to_sample = i
                    break
            self.users_in_batch[user_to_sample] += 1

    def get_clicks_real(self, bid):
        number_of_clicks = 0
        for idx, user in enumerate(self.users):
            number_of_clicks += user.get_clicks_real(bid) * self.users_in_batch[idx]
        return round(number_of_clicks / BATCH_SIZE, 3)

    def get_clicks_noise(self, bid):
        y = self.get_clicks_real(bid)
        mu, sigma = 0, self.sigma
        sample = np.random.normal(mu, sigma)
        return round(max(0, y + sample), 3)

    def plot(self):
        for user in self.users:
            user.plot(self.idx)
        x = np.linspace(0, self.max_budget, 100)
        y = [self.get_clicks_real(bid) for bid in x]
        plt.plot(x, y)
        plt.title("Subcampaign {}, aggregated curve".format(self.idx))
        plt.xlabel("Daily budget [€]")
        plt.xlim((0, self.max_budget))
        plt.xticks(np.arange(0, self.max_budget + 1, 20))
        plt.ylabel("Number of clicks")
        plt.ylim((0, max(self.max_clicks) + max(self.max_clicks) / 10))
        plt.yticks(np.arange(0, max(y) + max(y) / 10, max(y) / 10))
        plt.grid()
        plt.show()


class Environment:

    def __init__(self, n_arms, n_users, n_subcampaigns, min_budgets, max_budgets, total_budget, user_probabilities, sigma, bids, slopes,
                 max_clicks, click_values):
        self.n_arms = n_arms
        self.n_users = n_users
        self.n_subcampaigns = n_subcampaigns
        self.min_budgets = min_budgets
        self.max_budgets = max_budgets
        self.total_budget = total_budget
        self.user_probabilities = user_probabilities
        self.sigma = sigma
        self.click_values = click_values
        self.subcampaigns = list()

        for idx_subcampaign in range(0, n_subcampaigns):
            new_subcampaign = Subcampaign(
                n_arms=self.n_arms,
                n_users=self.n_users,
                user_probabilities=self.user_probabilities[idx_subcampaign],
                min_budget=self.min_budgets[idx_subcampaign],
                max_budget=self.max_budgets[idx_subcampaign],
                bids=bids[idx_subcampaign],
                slopes=slopes[idx_subcampaign],
                sigma=self.sigma,
                idx=idx_subcampaign,
                max_clicks=max_clicks[idx_subcampaign],
                click_value=click_values[idx_subcampaign])
            self.subcampaigns.append(new_subcampaign)

    def get_new_batch(self):
        for subcampaign in self.subcampaigns:
            subcampaign.get_new_batch()

    def get_arms(self):
        arms = np.linspace(0, self.total_budget, self.n_arms)
        return arms

    def get_rewards(self, budget, idx_subcampaign):
        subcampaign = self.subcampaigns[idx_subcampaign]
        return subcampaign.get_clicks_noise(budget)

    def get_clicks_real(self, budget, idx_subcampaign):
        subcampaign = self.subcampaigns[idx_subcampaign]
        return subcampaign.get_clicks_real(budget)

    def plot(self):
        """
                for idx_subcampaign in range(0, self.n_subcampaigns):
                    self.subcampaigns[idx_subcampaign].plot()
        """
        # need to sample users to plot them
        users_in_batch, subcampaigns = [], []
        for idx_s, probabilities_current_ in enumerate(self.user_probabilities):
            for _ in range(0, 256):
                users_in_batch.append(np.random.choice(len(probabilities_current_), 1, p=probabilities_current_)[0])
                subcampaigns.append(idx_s)
            df = pd.DataFrame(data={"Class": users_in_batch, "Subcampaign": subcampaigns})
            ax = sns.barplot(x="Class", y="Class", data=df, estimator=lambda x: len(x) / len(df) * 100)
            ax.set(ylabel="Percent")

            # g = sns.FacetGrid(pd.DataFrame(data={"Class": users_in_batch, "Subcampaign": subcampaigns}), size=10, row="Subcampaign", legend_out=True)
            # g = g.map(sns.barplot, "Class")
            # for t in t_vals:
            # current_df = df.where(df["Time"] == t).dropna()
            # sns.countplot(data=current_df, x="Class")
            # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

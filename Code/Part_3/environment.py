import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(17)

BATCH_SIZE = 16


class User:
    def __init__(self, max_budget, bid, slope, sigma, idx, max_value_subcampaign):
        self.max_budget = max_budget
        self.bid = bid
        self.slope = slope
        self.sigma = sigma
        self.idx = idx
        self.max_clicks_subcampaign = max_value_subcampaign

    def get_clicks_real(self, bid):
        return max(self.max_clicks_subcampaign * (1 - np.exp(-self.slope * (bid - self.bid))), 0)

    def get_clicks_noise(self, bid):
        y = self.get_clicks_real(bid)
        mu, sigma = 0, self.sigma
        noise = np.random.normal(mu, sigma)
        return max(0, y + noise)

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
    def __init__(self, n_arms, n_users, user_probabilities, max_budget, sigma, idx, bids, slopes, max_clicks):
        self.n_arms = n_arms
        self.n_users = n_users
        # One probability for each user.
        self.user_probabilities = user_probabilities
        self.max_budget = max_budget
        self.max_clicks = max_clicks
        self.bids = bids
        self.slopes = slopes
        self.sigma = sigma
        self.users = list()
        self.idx = idx
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
                max_value_subcampaign=self.max_clicks[idx_user])
            self.users.append(new_user)

        # Get batch of users.
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
        return number_of_clicks / BATCH_SIZE

    def get_clicks_noise(self, bid):
        y = self.get_clicks_real(bid)
        mu, sigma = 0, self.sigma
        sample = np.random.normal(mu, sigma)
        return max(0, y + sample)

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

    def __init__(self, n_arms, n_users, n_subcampaigns, max_budget, user_probabilities, sigma, bids, slopes,
                 max_clicks):
        self.n_arms = n_arms
        self.n_users = n_users
        self.n_subcampaigns = n_subcampaigns
        self.max_budget = max_budget
        self.user_probabilities = user_probabilities
        self.sigma = sigma
        self.subcampaigns = list()

        for idx_subcampaign in range(0, n_subcampaigns):
            new_subcampaign = Subcampaign(
                n_arms=self.n_arms,
                n_users=self.n_users,
                user_probabilities=self.user_probabilities[idx_subcampaign],
                max_budget=self.max_budget,
                bids=bids[idx_subcampaign],
                slopes=slopes[idx_subcampaign],
                sigma=self.sigma,
                idx=idx_subcampaign,
                max_clicks=max_clicks[idx_subcampaign])
            self.subcampaigns.append(new_subcampaign)

    def get_arms(self):
        arms = np.linspace(0, self.max_budget, self.n_arms)
        return arms

    def get_rewards(self, budget, idx_subcampaign):
        return self.subcampaigns[idx_subcampaign].get_clicks_noise(budget)

    def get_clicks_real(self, budget, idx_subcampaign):
        return self.subcampaigns[idx_subcampaign].get_clicks_real(budget)

    def plot(self):
        for idx_subcampaign in range(0, self.n_subcampaigns):
            self.subcampaigns[idx_subcampaign].plot()

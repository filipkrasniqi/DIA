from random import random

import matplotlib.pyplot as plt
import numpy as np

RESOLUTION = 20


class User:
    def __init__(self, max_budget, bid, slope, max_clicks, idx):
        self.max_budget = max_budget
        self.bid = bid
        self.slope = slope
        self.max_clicks = max_clicks
        self.x = list()
        self.y = list()
        self.idx = idx
        self.generate_values()

    def generate_values(self):
        # Linear space from 0 to bid value, y is 0 for everything.
        x_0 = np.linspace(0, self.bid, self.bid * RESOLUTION)
        y_0 = np.zeros(self.bid * RESOLUTION)

        # Linear space from bid value to max_budget, y is exponential.
        x_1 = np.linspace(self.bid, self.max_budget, (self.max_budget - self.bid) * RESOLUTION)
        x = np.linspace(0, self.max_budget - self.bid, (self.max_budget - self.bid) * RESOLUTION)
        y_1 = (1 - np.exp(self.slope * x)) * self.max_clicks

        self.x = np.append(x_0, x_1)
        self.y = np.append(y_0, y_1)

    def plot(self, idx_subcampaign):
        plt.plot(self.x, self.y)
        plt.title("Subcampaign {}, User {}".format(idx_subcampaign, self.idx))
        plt.xlabel("Daily budget")
        plt.xlim((0, self.max_budget))
        plt.ylabel("Clicks")
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
        self.x = list()
        self.y = list()
        self.idx = idx
        self.generate()

    def generate(self):
        # Generate a curve for each user.
        for idx_user in range(self.n_users):
            new_user = User(self.max_budget,
                            self.bids[idx_user],
                            self.slopes[idx_user],
                            self.max_clicks[idx_user],
                            idx_user)
            self.users.append(new_user)
        # Create subcampaign curve.
        self.x = self.users[0].x
        self.y = self.x.copy()
        # Create perfect curve with known probabilities.
        for i in range(len(self.x)):
            self.y[i] = 0
            y_temp = 0
            for pos in range(0, self.n_users):
                y_temp = y_temp + self.users[pos].y[i] * self.user_probabilities[pos]
            self.y[i] = y_temp

    def get_clicks_noise(self, budget):
        y = self.get_clicks_real(budget)
        mu, sigma = 0, self.sigma
        sample = np.random.normal(mu, sigma)
        return max(0, y + sample)

    def get_clicks_real(self, budget):
        # Search for the index representing the right budget on the x axis.
        index = 0
        while budget > self.x[index]:
            index = index + 1

        # Return number of clicks.
        return self.y[index]

    def sample(self, budget):
        # Search for the index representing the right budget on the x axis.
        index = 0
        while budget > self.x[index]:
            index = index + 1

        # Generate random value in order to select user to sample.
        user_sampled = len(self.user_probabilities)
        for i, prob in enumerate(self.user_probabilities):
            rand = random()
            if rand <= prob:
                user_sampled = i
                break

        # User curve sampling.
        y = self.users[user_sampled].y[index]

        return y, user_sampled

    def plot(self):
        for u in range(0, self.n_users):
            self.users[u].plot(self.idx)
        plt.plot(self.x, self.y)
        plt.title("Subcampaign {}".format(self.idx))
        plt.xlabel("Daily budget")
        plt.xlim((0, self.max_budget))
        plt.ylabel("Clicks")
        plt.show()


class Environment:

    def __init__(self, n_arms, n_users, n_subcampaigns, max_budget, prob_users, sigma, bids, slopes, max_clicks):
        # Arms.
        self.n_arms = n_arms
        self.n_users = n_users
        self.n_subcampaigns = n_subcampaigns
        self.max_budget = max_budget
        self.prob_users = prob_users
        self.sigma = sigma
        self.subcampaigns = []

        for idx_subcampaign in range(0, n_subcampaigns):
            new_subcampaign = Subcampaign(
                n_arms=self.n_arms,
                n_users=self.n_users,
                user_probabilities=self.prob_users[idx_subcampaign],
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

    def get_clicks_real(self, budget, idx_subcampaign):
        return self.subcampaigns[idx_subcampaign].get_clicks_real(budget)

    def get_clicks_noise(self, budget, idx_subcampaign):
        return self.subcampaigns[idx_subcampaign].get_clicks_noise(budget)

    def plot(self):
        for idx_subcampaign in range(0, self.n_subcampaigns):
            self.subcampaigns[idx_subcampaign].plot()

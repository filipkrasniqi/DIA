import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import itertools
from random import random

# Punto 2

RESOLUTION = 20


class User:
    def __init__(self, max_budget, bid, slope, max_clicks, idx):
        self.max_budget = max_budget
        self.bid = bid
        self.slope = slope
        self.max_clicks = max_clicks
        self.x = []
        self.y = []
        self.generate()
        self.idx = idx

    # Returns a tuple containing x and y values of a user clicks curve.
    def generate(self):
        # Linear space from 0 to bid value, y is 0 for everything.
        x_0 = np.linspace(0, self.bid, self.bid * RESOLUTION)
        y_0 = np.zeros(self.bid * RESOLUTION)

        # Linear space from bid value to max_budget, y is exponential.
        x_1 = np.linspace(self.bid, self.max_budget, (self.max_budget - self.bid) * RESOLUTION)
        x = np.linspace(0, self.max_budget - self.bid, (self.max_budget - self.bid) * RESOLUTION)
        y_1 = (1 - np.exp(self.slope * x)) * self.max_clicks

        self.x = np.append(x_0, x_1)
        self.y = np.append(y_0, y_1)

    def plot(self, subc):
        plt.plot(self.x, self.y)
        plt.title("Subcampaign {}, User {}".format(subc, self.idx))
        plt.xlabel("Daily budget")
        plt.xlim((0, self.max_budget))
        plt.ylabel("Clicks")
        plt.show()


class Subcampaign:
    def __init__(self, n_arms, n_users, prob_users, max_budget, sigma, idx, bids, slopes, max_clicks):
        self.n_arms = n_arms
        self.n_users = n_users
        # One probability for each user.
        self.prob_users = prob_users
        self.max_budget = max_budget
        self.max_clicks = max_clicks
        self.bids = bids
        self.slopes = slopes
        self.sigma = sigma
        self.users = []
        self.x = []
        self.y = []
        self.generate()
        self.idx = idx

    def generate(self):

        # Generate a curve for each user.
        for i in range(self.n_users):
            new_user = User(self.max_budget, self.bids[i], self.slopes[i], self.max_clicks[i], i)
            self.users.append(new_user)

        # Create subcampaign curve.
        self.x = self.users[0].x
        self.y = self.x.copy()

        # Creation of perfect curve with known probabilities.
        for i in range(len(self.x)):
            self.y[i] = 0
            y_temp = 0
            for pos in range(0, self.n_users):
                y_temp = y_temp + self.users[pos].y[i] * self.prob_users[pos]
            self.y[i] = y_temp

    '''
    Other implementation of function with noise
    '''

    def get_clicks_noise_summed(self, budget):
        y = self.get_clicks_real(budget)
        mu, sigma = 0, self.sigma
        '''
        lower, upper = -y, y+2*self.sigma
        mu, sigma = 0, self.sigma
        sample = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
        '''
        sample = np.random.normal(mu, sigma)
        return max(0, y + sample)

    def get_clicks_real(self, x_value):
        # Search for the index representing the right x_value on the x axis.
        index = 0
        while x_value > self.x[index]:
            index = index + 1
            
        y = self.y[index]
        # Check for negativity.
        if y < 0:
            print("FUNZIONE NEGATIVA!!!")
        return y

    def get_clicks_noise(self, budgets):
        lower, upper = 0, self.get_clicks_real(budgets) + self.sigma
        mu, sigma = self.get_clicks_real(budgets), self.sigma
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        return X.rvs(1)
        # return max(0, np.random.normal(self.get_clicks_real(x_value), self.sigma))

    def sample(self, budget):
        # Search for the index representing the right budget on the x axis.
        index = 0
        while budget > self.x[index]:
            index = index + 1

        # Generate random value in order to select user to sample.
        user_sampled = len(self.prob_users)
        for i, prob in enumerate(self.prob_users):
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

    def __init__(self, n_arms, n_users, n_subcampaign, max_budget, prob_users, sigma, bids, slopes, max_clicks):
        # Arms.
        self.n_arms = n_arms
        self.n_users = n_users
        self.n_subcampaigns = n_subcampaign
        self.max_budget = max_budget
        self.prob_users = prob_users
        self.sigma = sigma
        self.subcampaigns = []

        for s in range(0, n_subcampaign):
            new_subc = Subcampaign(
                n_arms=self.n_arms,
                n_users=self.n_users,
                prob_users=self.prob_users[s],
                max_budget=self.max_budget,
                bids=bids[s],
                slopes = slopes[s],
                sigma=self.sigma,
                idx = s,
                max_clicks = max_clicks[s])
            self.subcampaigns.append(new_subc)

    def get_arms(self):
        arms = np.linspace(0, self.max_budget, self.n_arms)
        return arms

    def get_clicks_real(self, x_value, subc):
        return self.subcampaigns[subc].get_clicks_real(x_value)

    def get_click_noise(self, x_value, subc):
        return self.subcampaigns[subc].get_clicks_noise_summed(x_value)

    def get_clicks_noise(self, budgets):
        '''
        click = []
        for i, budget in enumerate(budgets):
            click.append(self.get_click_noise(budget,i))
        return click
        '''
        click = []
        for budget, n_subc in itertools.product(budgets, list(range(self.n_subcampaigns))):
            click.append(self.get_click_noise(budget, n_subc))
        return click

    def plot(self):
        for s in range(0, self.n_subcampaigns):
            self.subcampaigns[s].plot()
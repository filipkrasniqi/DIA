import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Punto 2

class User:
    def __init__(self, max_budget, bid, slope):
        self.max_budget = max_budget
        self.bid = bid
        self.slope = slope
        self.x = []
        self.y = []
        self.generate()

    # Returns a tuple containing x and y values of a user clicks curve.
    def generate(self):
        # Linear space from 0 to bid value, y is 0 for everything.
        x_0 = np.linspace(0, self.bid, self.bid * 20)
        y_0 = np.zeros(self.bid * 20)

        # Linear space from bid value to max_budget, y is exponential.
        x_1 = np.linspace(self.bid, self.max_budget, self.max_budget * 20)
        x = np.linspace(0, self.max_budget - self.bid, self.max_budget * 20)
        y_1 = (1 - np.exp(self.slope * x)) * 10

        self.x = np.append(x_0, x_1)
        self.y = np.append(y_0, y_1)

    def plot(self):
        plt.plot(self.x, self.y)
        plt.title("User")
        plt.xlabel("Daily budget")
        plt.xlim((0, self.max_budget))
        plt.ylabel("Clicks")
        plt.show()


class Subcampaign:
    def __init__(self, n_arms, n_users, prob_users, max_budget, bid, sigma):
        self.n_arms = n_arms
        self.n_users = n_users
        # One probability for each user.
        self.prob_users = prob_users
        self.max_budget = max_budget
        self.bid = bid
        self.sigma = sigma
        self.users = []
        self.x = []
        self.y = []
        self.generate()

    def generate(self):
        # Slope for each user curve.
        slopes = [-0.1 * (1), -0.1 * (2), -0.1 * (3)]

        # Generate a curve for each user.
        for i in range(self.n_users):
            new_user = User(self.max_budget, self.bid, slopes[i])
            self.users.append(new_user)

        # Create subcampaign curve.
        self.x = self.users[0].x
        self.y = self.x.copy()

        for i in range(len(self.x)):
            self.y[i] = 0
            y_temp = 0
            for pos in range(0, self.n_users):
                y_temp = y_temp + self.users[pos].y[i] * self.prob_users[pos]
            self.y[i] = y_temp

    def get_clicks_real(self, x_value):
        index = 0
        while x_value > self.x[index]:
            index = index + 1
        y = self.y[index]
        return y

    def get_clicks_noise(self, x_value):
        lower, upper = 0, self.get_clicks_real(x_value) + self.sigma
        mu, sigma = self.get_clicks_real(x_value),self.sigma
        X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        return X.rvs(1)
        #return max(0, np.random.normal(self.get_clicks_real(x_value), self.sigma))

    def plot(self):
        for u in range(0, self.n_users):
            self.users[u].plot()
        plt.plot(self.x, self.y)
        plt.title("Subcampaign")
        plt.xlabel("Daily budget")
        plt.xlim((0, self.max_budget))
        plt.ylabel("Clicks")
        plt.show()


class Environment:

    def __init__(self, n_arms, n_users, n_subcampaign, max_budget, bid, prob_users, sigma):
        # Arms.
        self.n_arms = n_arms
        self.n_users = n_users
        self.n_subcampaigns = n_subcampaign
        self.max_budget = max_budget
        self.bid = bid
        self.prob_users = prob_users
        self.sigma = sigma
        self.subcampaigns = []

        for s in range(0, n_subcampaign):
            new_subc = Subcampaign(
                n_arms=self.n_arms,
                n_users=self.n_users,
                prob_users=self.prob_users[s],
                max_budget=self.max_budget,
                bid=self.bid,
                sigma=self.sigma
            )
            self.subcampaigns.append(new_subc)

    def get_arms(self):
        arms = np.linspace(0, self.max_budget, self.n_arms)
        return arms

    def get_clicks_real(self, x_value, subc):
        return self.subcampaigns[subc].get_clicks_real(x_value)

    def get_click_noise(self, x_value, subc):
        return self.subcampaigns[subc].get_clicks_noise(x_value)

    def get_clicks_noise(self, x_value):
        click = []
        for i,val in enumerate(x_value):
            click.append(self.get_click_noise(val,i))

        return click

        #return [self.get_click_noise(val,i) for i,val in enumerate(x_value)]

    def plot(self):
        for s in range(0, self.n_subcampaigns):
            self.subcampaigns[s].plot()


'''user = User(max_budget=70, bid=10, slope=-0.1)
user.plot()

subc = Subcampaign(n_users=2, prob_users=[0.5, 0.5], max_budget=70, bid=10, daily_budget_values=[10, 20, 30])
subc.plot()
'''



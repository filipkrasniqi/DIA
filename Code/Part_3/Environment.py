import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Punto 2

def plot_user_curve(x, y,max_budget):
    plt.plot(x, y)
    plt.title("User")
    plt.xlabel("Daily budget")
    plt.xlim((0, max_budget))
    plt.ylabel("Clicks")
    plt.show()

def plot_subc_curve(x, y,max_budget):
    plt.plot(x, y)
    plt.title("Subcampaign")
    plt.xlabel("Daily budget")
    plt.xlim((0, max_budget))
    plt.ylabel("Clicks")
    plt.show()

# Returns a tuple containing x and y values of a user clicks curve.
def generate_clicks_curve(bid, max_budget, slope):
    slope = slope + add_noise(mu=0.01, std=0.05)

    # Linear space from 0 to bid value, y is 0 for everything.
    x_0 = np.linspace(0, bid, bid * 20)
    y_0 = np.zeros(bid * 20)

    # Linear space from bid value to max_budget, y is exponential.
    x_1 = np.linspace(bid, max_budget, max_budget * 20)
    x = np.linspace(0, max_budget - bid, max_budget * 20)
    y_1 = (1 - np.exp(slope * x)) * 10

    x = np.append(x_0, x_1)
    y = np.append(y_0, y_1)

    plot_user_curve(x, y)

    return (x, y)

def generate_subcampaign(bid, max_budget, prob):
    n_user = len(prob)

    # Slope for each user curve.
    slopes = [-0.1 * (1), -0.1 * (2), -0.1 * (3)]

    # Generate a curve for each user.
    user_curves = []
    for user in range(n_user):
        user_curve = generate_clicks_curve(bid, max_budget, slopes[user])
        user_curves.append(user_curve)

    # Create subcampaign curve.
    x_subc = user_curves[0][0]
    y_subc = []

    for i in range(len(x_subc)):
        y_temp = user_curves[0][1][i] * prob[0] + user_curves[1][1][i] * prob[1] + user_curves[2][1][i] * prob[2]
        y_subc.append(y_temp)

    plot_subc_curve(x_subc, y_subc)

    return (x_subc, y_subc)


# map the daily budget to the corresponding number of click
def fun(x):
    # return 100* (1.0 - np.exp(-4*x+3*x**3))

    slope = 1
    return (1 - np.exp(-slope * x)) * 100


class Environment():
    def __init__(self, daily_budgets, sigma):
        self.daily_budgets = daily_budgets
        self.means = fun(daily_budgets)
        self.sigmas = np.ones(len(daily_budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

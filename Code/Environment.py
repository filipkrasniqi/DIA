import numpy as np


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

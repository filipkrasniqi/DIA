import numpy as np
import matplotlib.pyplot as plt

from Code.Part_1.Environment import Environment

season_length = 90
number_of_seasons = 4

class ProjectEnvironment(Environment):
    def __init__(self, n_arms, probabilities, sigma, matrix_parameters):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.sigma = sigma
        self.users = [User(i,parameters, self.sigma) for i,parameters in enumerate(matrix_parameters)]

    def round(self, pulled_arm, t):
        choices = [i for i in range(self.n_users)]
        user = np.random.choice(choices, 1, self.probabilities)
        sample = self.users[user].update_samples(pulled_arm, t)  # domanda
        return sample * pulled_arm  # reward

    def season(self, t):
        return int((t % 365) / season_length)

    def plot(self, n_arms):
        for u in self.users:
            u.plot(n_arms)


class User():
    def __init__(self,idx,parameters, sigma):
        self.phase_function = [(s,D,f_smooth) for s,D,f_smooth in parameters]
        self.samples = []
        self.sigma = sigma
        self.idx = idx

    def update_samples(self, price, t):
        sample = self.noised_demand(price, t)
        self.samples.append(sample)
        return sample

    def noised_demand(self, price, t):
        return self.demand(price, t) + np.random.normal(0, self.sigma)

    def demand(self, price, t, season = None):
        if season is None:
            season = self.season(t)
        return np.exp(price * (-1) * self.phase_function[season][0]) * self.phase_function[season][1] * self.phase_function[season][2](t)

    def season(self, t):
        return int((t % 365) / season_length)

    def plot(self, n_arms):
        bins = np.linspace(0,n_arms)
        seasons = list(range(number_of_seasons))
        demands = []
        for s in seasons:
            for p in bins:
                demands.append(self.demand(p,(s+1)*season_length, s))

        demands = np.array(demands).reshape(len(seasons),-1)

        for s,demand in enumerate(demands):
            plt.plot(bins, demand)
            plt.title("Demand Curve User {},Season {}".format(self.idx,s))
            plt.xlabel("Prices")
            plt.xlim((0, np.max(bins)))
            plt.ylim(0, np.max(demand))
            plt.ylabel("Demand")
            plt.show()

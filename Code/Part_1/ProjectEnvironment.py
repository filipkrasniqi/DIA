import math

import numpy as np
import matplotlib.pyplot as plt

from Code.Part_1.Environment import Environment

season_length = 360
number_of_seasons = 4

class ProjectEnvironment(Environment):
    def __init__(self, arms, probabilities, sigma, matrix_parameters):
        self.arms = arms
        self.n_arms = len(self.arms)
        self.probabilities = probabilities
        self.sigma = sigma
        self.users = [User(i, arms, parameters, self.sigma) for i,parameters in enumerate(matrix_parameters)]
        self.regret = []

    def round(self, pulled_arm, t):
        choices = [i for i in range(len(self.users))]
        user = np.random.choice(choices, 1, self.probabilities)[0]
        sample = self.users[user].update_samples(pulled_arm, t)  # demand
        reward = sample * pulled_arm
        real_sample = self.users[user].demand(pulled_arm, t)
        real_reward = real_sample * pulled_arm
        optimum, optimum_arm = self.users[user].optimum(t)
        self.regret.append(optimum - real_reward)
        return reward, user  # reward

    def season(self, t):
        return int((t % 365) / season_length)

    def plot(self):
        for u in self.users:
            u.plot()


class User():
    def __init__(self, idx, arms, parameters, sigma):
        self.phase_function = [(s,D,f_smooth) for s,D,f_smooth in parameters]
        self.samples = []
        self.sigma = sigma
        self.idx = idx
        self.arms = arms
        self.n_arms = len(self.arms)

    def update_samples(self, price, t):
        sample = self.noised_demand(price, t)
        self.samples.append(sample)
        return sample

    def noised_demand(self, price, t):
        return max(0, self.demand(price, t) + np.random.normal(0, self.sigma))

    def demand(self, price, t, season = None):
        if season is None:
            season = self.season(t)
        if season >= number_of_seasons:
            print("mierda")
        return np.exp(price * (-1) * self.phase_function[season][0]) * self.phase_function[season][1] * self.phase_function[season][2](t)

    def optimum(self, t, season = None):
        all_demands = [self.demand(arm, t, season) * arm for arm in self.arms]
        return np.max(all_demands), np.argmax(all_demands)

    def season(self, t):
        return int((t / season_length)) % number_of_seasons

    def plot(self):
        bins = np.linspace(0,max(self.arms))
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

import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTSLearner:
    def __init__(self, arms, sigma):
        self.name_learner = "GPTS learner"
        self.arms = arms
        self.means = np.ones(len(arms))
        self.sigmas = np.ones(len(arms)) * sigma
        self.pulled_arms = list()
        self.collected_rewards = list()

        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma ** 2, normalize_y=True, n_restarts_optimizer=10)

    def update(self, idx_pulled_arm, reward):
        self.pulled_arms.append(self.arms[idx_pulled_arm])
        self.collected_rewards.append(reward)
        # Split pulled_arms in elements with at least 2 arrays each one.
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        # X: Training Data
        # Y: Target Values
        self.gp = self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
        self.means = np.maximum(self.means, 0)

    def pull_arms(self):
        """sampled_values = []
                for mu, sigma in zip(self.means, self.sigmas):
                    lower, upper = 0, mu + 2 * sigma
                    mu, sigma = mu, sigma
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                    sampled_values.append(X.rvs(1))
                return sampled_values"""
        return self.gp.sample_y(np.atleast_2d(self.arms).T)
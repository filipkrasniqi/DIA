import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTSLearner:
    def __init__(self, arms, sigma,restart_optimizer = 0,cost_kernel=1,lenght_scale_kernel = 0.1):
        self.name_learner = "GPTS learner"
        self.arms = arms
        self.means = np.ones(len(arms))
        self.sigmas = np.ones(len(arms)) * sigma
        self.pulled_arms = list()
        self.collected_rewards = list()
        self.users_sampled = [0, 0, 0]

        kernel = C(cost_kernel, (1e-3, 1e3)) * RBF(lenght_scale_kernel, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-7,
            normalize_y=True,
            n_restarts_optimizer=restart_optimizer
        )

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

    def sample_arms(self):
        return np.random.normal(self.means,self.sigmas)

        #values = self.gp.sample_y(np.atleast_2d(self.arms).T)
        #return np.where(values > 0, values, 0)

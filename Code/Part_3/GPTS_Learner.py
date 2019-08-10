from Code.Part_1.Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms, sigma_gp,initial_sigmas):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.ones(n_arms)
        self.sigmas = np.ones(n_arms) * initial_sigmas
        self.pulled_arms = []
        alpha = sigma_gp

        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=10)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        # Split pulled_arms in elements with at least 2 arrays each one.
        # np.atleast_3d(3.0) --> array([[[ 3.]]])

        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        # X: Training Data
        # Y: Target Values
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return sampled_values

        #return np.argmax(sampled_values)



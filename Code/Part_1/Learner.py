import numpy as np

class Learner():

    def __init__(self,arms, n_users = 3):
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.drawn_user = np.array([])
        self.user_samples = [0 for _ in range(n_users)]  # samples drawn for each user

    def update_observations(self,pulled_arm,reward, user):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.user_samples[user] += 1
        self.collected_rewards = np.append(self.collected_rewards,reward)
        self.drawn_user = np.append(self.drawn_user, user)

    def get_collected_rewards_user(self, user):
        indices = [i for i, u in enumerate(self.drawn_user) if u == user]
        return self.collected_rewards[indices]

    def num_samples(self, arm):
        return len(self.rewards_per_arm[arm])


import numpy as np
import matplotlib.pyplot as plt

from Code.Part_1.Environment import Environment

season_length = 720
number_of_seasons = 4


class ProjectEnvironment(Environment):
    def __init__(self, arms, probabilities, sigma, context_matrix_parameters, context_alternatives):
        self.arms = arms
        self.n_arms = len(self.arms)
        self.probabilities = probabilities
        self.sigma = sigma
        """
        List of contexts.
        A single context is a list of subcontexts.
        A single subcontexts is a list of users.
        Each subcontext is associated to a function to learn.
        """
        self.contexts = []
        for idx_c, context_parameters in enumerate(context_matrix_parameters):
            self.contexts.append([])
            for idx_s, subcontext_parameters in enumerate(context_parameters):
                self.contexts[idx_c].append(User(idx_s, arms, subcontext_parameters, sigma))
        self.regret = []
        self.real_rewards = []
        self.drawn_users = []
        self.selected_context = 0
        self.contexts_alternatives = context_alternatives
        self.all_rewards = []  # list of arrays. Each array contains <#arms> values

    def set_context(self, selected_context):
        self.selected_context = selected_context

    def probabilities_context(self, context):
        choices = list(range(len(self.contexts[context])))
        probabilities_context = [0 for _ in choices]
        subcontexts = self.contexts_alternatives[context]
        for idx_u, p in enumerate(self.probabilities):
            idx_s_containing_u = [idx for idx, s in enumerate(subcontexts) if idx_u in s][0]
            probabilities_context[idx_s_containing_u] += p
        return probabilities_context

    """
    Execute round. Sequentially:
    - 
    """

    def round(self, t, user=None):
        choices = [i for i in range(len(self.contexts[self.selected_context]))]
        probabilities_context = self.probabilities_context(self.selected_context)
        users = self.contexts[self.selected_context]
        if user is None:
            user = np.random.choice(choices, 1, probabilities_context)[0]
        samples = []
        for arm in self.arms:
            samples.append(users[user].update_samples(arm, t) * arm)  # demand
        self.all_rewards.append(samples)
        return samples, user

    def round_for_arm(self, pulled_arm, t, user=None):
        choices = [i for i in range(len(self.contexts[self.selected_context]))]
        users = self.contexts[self.selected_context]
        subcontexts = self.contexts_alternatives[self.selected_context]
        if user is None:
            user = np.random.choice(choices, 1, self.probabilities)[0]
        subcontext = self.get_subcontext_from_user(subcontexts, user)
        idx_arm = [idx_arm for idx_arm, arm in enumerate(self.arms) if arm == pulled_arm][0]
        reward = self.all_rewards[-1][idx_arm]
        real_sample = users[subcontext].demand(pulled_arm, t)
        real_reward = real_sample * pulled_arm
        contexts_optimals = []
        for idx_c, context in enumerate(self.contexts):
            context_optimal = 0
            for user, p in zip(context, self.probabilities_context(idx_c)):
                optimum, optimum_arm = user.optimum(t)
                context_optimal += p * optimum
            contexts_optimals.append(context_optimal)
        best_context = np.max(contexts_optimals)
        self.regret.append(best_context - real_reward)
        self.real_rewards.append(real_reward)
        self.drawn_users.append(user)
        return reward, user

    def sample_subcontext(self):
        subcontexts = self.contexts_alternatives[self.selected_context]
        user = np.random.choice([i for i in range(len(self.probabilities))], 1, self.probabilities)[0]
        subcontext = self.get_subcontext_from_user(subcontexts, user)
        return subcontext, user

    def get_subcontext_from_user(self, subcontexts, user):
        return [idx_s for idx_s, subcontext in enumerate(subcontexts) if user in subcontext][0]

    def get_last_round(self, pulled_arm, t):
        user = self.drawn_users[-1]
        return self.users[user].samples[-1], user

    def season(self, t):
        return int((t % 365) / season_length)

    def plot(self):
        users = self.contexts[self.selected_context]
        for u in users:
            u.plot()


"""
A single user identifies a function.
In case of contexts, it may be associated to a set of users.
"""


class User():
    def __init__(self, idx, arms, parameters, sigma):
        self.phase_function = [(s, D, f_smooth) for s, D, f_smooth in parameters]
        self.samples = []
        self.sigma = sigma
        self.idx = idx
        self.arms = arms
        self.n_arms = len(self.arms)

    def n_samples(self):
        return len(self.samples)

    def update_samples(self, price, t):
        sample = self.noised_demand(price, t)
        self.samples.append(sample)
        return sample

    def noised_demand(self, price, t):
        return max(0, self.demand(price, t) + np.random.normal(0, self.sigma))

    def demand(self, price, t, season=None):
        if season is None:
            season = self.season(t)
        if season >= number_of_seasons:
            print("mierda")
        return np.exp(price * (-1) * self.phase_function[season][0]) * self.phase_function[season][1] * \
               self.phase_function[season][2](t)

    def optimum(self, t, season=None):
        all_demands = [self.demand(arm, t, season) * arm for arm in self.arms]
        return np.max(all_demands), np.argmax(all_demands)

    def season(self, t):
        return int((t / season_length)) % number_of_seasons

    def plot(self):
        bins = np.linspace(0, max(self.arms))
        seasons = list(range(number_of_seasons))
        demands = []
        for s in seasons:
            for p in bins:
                demands.append(self.demand(p, (s + 1) * season_length, s))

        demands = np.array(demands).reshape(len(seasons), -1)

        for s, demand in enumerate(demands):
            plt.plot(bins, demand)
            plt.title("Demand Curve User {},Season {}".format(self.idx, s))
            plt.xlabel("Prices")
            plt.xlim((0, np.max(bins)))
            plt.ylim(0, np.max(demand))
            plt.ylabel("Demand")
            plt.show()

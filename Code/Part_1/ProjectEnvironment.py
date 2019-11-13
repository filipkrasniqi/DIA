import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set(rc={'figure.figsize':(15,9)})
sns.set(font_scale=2)  # crazy big

import pandas as pd

from Code.Part_1.Environment import Environment
import itertools

"""
Environment for online pricing.
Env = (A, P, C), being
- A = set of arms
- P = set of probabilities, one for each user
- C = set of contexts

Each context is a set of subcontext. Each subcontext is a set of users,
and it is characterized by a demand function.

Given a subcontext s, his demand function may vary in time, so we have
demand_s = f_s(price, t).

Change in time are:
- seasonal changes, i.e., abrupt changes
- smooth changes (i.e., small variation in a season)
"""
class ProjectEnvironment(Environment):
    def __init__(self, arms, probabilities, sigma, context_matrix_parameters, context_alternatives, batch_size = 16, season_length = 91, number_of_seasons = 4):
        self.arms = arms
        self.n_arms = len(self.arms)
        self.probabilities = probabilities
        self.sigma = sigma
        self.batch_size = batch_size
        self.season_length = season_length
        self.number_of_season = number_of_seasons
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
                self.contexts[idx_c].append(User(idx_s, arms, subcontext_parameters, sigma, self.season_length, self.number_of_season))
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
    Execute round by sampling, for each subcontext, batch_size values for each arm.
    eg: context of size 3, batch_size = 8, 5 arms => 120 samples
    Arm is pulled afterwards because for now we know nothing.
    Assumption: these samples are available
    """
    def round_context(self, t):
        rewards, demands, sub_contexts_to_return, users_to_return = [[] for _ in self.contexts], [[] for _ in self.contexts], [[] for _ in self.contexts], [[] for _ in self.contexts]
        list_attempts = list(range(self.batch_size))
        for idx_attempt, (idx_c, context) in itertools.product(list_attempts, enumerate(self.contexts)):
            sub_context, user = self.sample_subcontext(idx_c)
            sub_contexts_to_return[idx_c].append(sub_context)
            users_to_return[idx_c].append(user)
            for arm in self.arms:
                demand = context[sub_context].update_samples(arm, t)
                demands[idx_c].append(demand)
                rewards[idx_c].append(demand * arm)  # demand

        for samples_context in rewards:
            for sample in samples_context:
                self.all_rewards.append(sample)
        return rewards, demands, sub_contexts_to_return, users_to_return

    """
    Given a single arm, it pulls more arms with associated users.
    Called after round_context to compute the regret, as
    the learners need the execution of round_context to know what's going on
    """
    def round_for_arm(self, idx_pulled_arms, t, users):
        regret_t, real_reward_t = 0, 0
        for idx, (idx_arm, user) in enumerate(zip(idx_pulled_arms, users)):
            users = self.contexts[self.selected_context]
            subcontexts = self.contexts_alternatives[self.selected_context]
            subcontext = self.get_subcontext_from_user(subcontexts, user)
            pulled_arm = self.arms[idx_arm]
            idx_reward = self.batch_size - idx
            reward = self.all_rewards[idx_reward]
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
            regret_t += best_context - real_reward
            real_reward_t += real_reward
            self.drawn_users.append(user)
        self.regret.append(regret_t)
        self.real_rewards.append(real_reward_t)
        return reward, user
    """
    Samples given a context ID
    """
    def sample_subcontext(self, context = None):
        if context is None:
            context = self.selected_context
        sub_contexts = self.contexts_alternatives[context]
        user = np.random.choice([i for i in range(len(self.probabilities))], 1, self.probabilities)[0]
        sub_context = self.get_subcontext_from_user(sub_contexts, user)
        return sub_context, user
    """
    Returns, given a user and the subcontexts, the idx in subcontexts where the user is contained.
    """
    def get_subcontext_from_user(self, subcontexts, user):
        return [idx_s for idx_s, subcontext in enumerate(subcontexts) if user in subcontext][0]
    """
    Returns number of season given t
    """
    def season(self, t):
        return int((t % 365) / self.season_length)
    """
    Plots contexts functions
    """
    def plot(self, contexts = None, T = None):
        if T is None:
            T = 16
        if contexts is None:
            contexts = self.contexts
        elif isinstance(contexts, list):
            # expect list of scalars
            contexts = [self.contexts[idx_c] for idx_c in contexts]
        else:
            # expects single scalar
            contexts = [self.contexts[contexts]]

        for idx_c, users in enumerate(contexts):
            for u in users:
                u.plot(T)


"""
A single user identifies a function.
In case of contexts, it may be associated to a set of users.
"""
class User():
    def __init__(self, idx, arms, parameters, sigma, season_length, number_of_seasons):
        self.phase_function = parameters # made of [(s, D, mu_values, f_smooth)
        self.samples = []
        self.sigma = sigma
        self.idx = idx
        self.arms = arms
        self.n_arms = len(self.arms)
        self.season_length = season_length
        self.number_of_seasons = number_of_seasons

    def n_samples(self):
        return len(self.samples)

    def update_samples(self, price, t):
        sample = self.noised_demand(price, t)
        self.samples.append(sample)
        return sample

    def noised_demand(self, price, t):
        return max(0, self.demand(price, t) + np.random.normal(0, self.sigma ** 2))

    def demand(self, price, t, season=None):
        if season is None:
            season = self.season(t)
        demand_val = np.exp(price * (-1) * self.phase_function[season][0]) * self.phase_function[season][1]
        for mu in self.phase_function[season][2]:
            demand_val += self.phase_function[season][3](t, mu, price)
        # self.phase_function[season][2](t)
        return demand_val

    def optimum(self, t, season=None):
        all_demands = [self.demand(arm, t, season) * arm for arm in self.arms]
        return np.max(all_demands), np.argmax(all_demands)

    def season(self, t):
        return int((t / self.season_length)) % self.number_of_seasons

    def hue_map(self, value):
        HUE_MAX = 0.9
        hue = pow(1 - value, 2) * HUE_MAX
        rgb = colors.hsv_to_rgb((hue, 1, 1))
        hex = colors.to_hex(rgb)
        return hex

    def plot(self, T):
        bins = np.linspace(0, max(self.arms))
        delta_t = 30
        first_t = 5
        num_t = int(((T - 1) - first_t) / delta_t)
        t_bins = np.linspace(first_t, T - 1, num_t)
        t_bins = [int(t) for t in t_bins]
        demands = []
        seasons_df, t_df = [], []
        for p in bins:
            for t in t_bins:
                s = self.season(t)
                demands.append(self.demand(p, t, s))
                seasons_df.append(s)
                t_df.append(t)
                # demands.append(self.demand(p, (s + 1) * season_length, s))

        df = pd.DataFrame(columns=["Demand", "Time", "Season", "Price"], data = {"Demand": demands, "Season": seasons_df, "Time": t_df, "Price": np.repeat(bins, len(t_bins))})
        colors_palette = [self.hue_map(t / T) for t in df.Time.unique()]
        palette = {
            t:colors_palette[i] for i, t in enumerate(df.Time.unique())
        }
        # demands = np.array(demands).reshape(len(seasons), -1)

        # for t in range(T):
        # current_df = df.where(df["Time"] == t).dropna()
        g = sns.FacetGrid(df, size=10, col="Season", hue="Time", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Demand").add_legend()
            # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.savefig("jolakm.png")

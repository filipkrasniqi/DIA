import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set(rc={'figure.figsize':(15,9)})
sns.set(font_scale=2.5)
import os

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
env_dir = outputs_dir+"v01_with_context/"

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
    def __init__(self, arms, number_users_function, sigma, context_matrix_parameters, context_alternatives, batch_size = 16, season_length = 91, number_of_seasons = 4, num_users = 3):
        self.arms = arms
        self.n_arms = len(self.arms)
        self.number_users_function = number_users_function
        self.sigma = sigma
        self.batch_size = batch_size
        self.season_length = season_length
        self.number_of_season = number_of_seasons
        self.num_users = num_users
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

    @staticmethod
    def get_env_dir():
        return env_dir

    """
    Returns the number of users for each of them depending on current probability function
    """
    def num_samples(self, t):
        return [self.number_users_function[idx_u](t) for idx_u in list(range(self.num_users))]

    """
    Returns the current probabilities for each user
    """
    def probabilities(self, t):
        num_samples = self.num_samples(t)
        sum = np.sum(num_samples)
        return [num_sample / sum for num_sample in num_samples]

    def set_context(self, selected_context):
            self.selected_context = selected_context

    def probabilities_context(self, context, t):
        choices = list(range(len(self.contexts[context])))
        probabilities_context = [0 for _ in choices]
        subcontexts = self.contexts_alternatives[context]
        for idx_u, p in enumerate(self.probabilities(t)):
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
            sub_context, user = self.sample_subcontext(t, idx_c)
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
            price = self.arms[idx_arm]
            idx_reward = self.batch_size - idx
            reward = self.all_rewards[idx_reward]
            real_sample = users[subcontext].demand(price, t)
            real_reward = real_sample * price
            contexts_optimals = []
            for idx_c, context in enumerate(self.contexts):
                context_optimal = 0
                for user, p in zip(context, self.probabilities_context(idx_c, t)):
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
    Return, given T, best rewards for each t
    """
    def best_rewards_t(self, T):
        best_rewards = []
        for t in range(1, T+1):
            for idx_c, context in enumerate(self.contexts):
                context_optimal = 0
                contexts_optimals = []
                for user, p in zip(context, self.probabilities_context(idx_c, t)):
                    optimum, optimum_arm = user.optimum(t)
                    context_optimal += p * optimum
                contexts_optimals.append(context_optimal)
            best_rewards.append(np.max(contexts_optimals))
        return best_rewards
    """
    Samples given a context ID
    """
    def sample_subcontext(self, t, context = None):
        if context is None:
            context = self.selected_context
        sub_contexts = self.contexts_alternatives[context]
        probabilities = self.probabilities(t)
        user = np.random.choice(len(probabilities), 1, p=probabilities)[0]
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

    def plot_context(self, idx_context, T, t_vals = [0, 40, 80, 91, 131, 171, 182, 222, 262, 273, 313, 353], real_demand = True):
        attempts_number = 128
        list_attempts = list(range(attempts_number))
        resolution = 32
        arms_to_plot = np.linspace(0, max(self.arms), resolution)
        demands, rewards, sub_contexts_to_return, users_to_return = [], [], [[] for _ in arms_to_plot], []
        t_df, season_df = [], []
        t_df_total, season_df_total = [], []

        for idx_arm, arm in enumerate(arms_to_plot):
            for idx_t, t in enumerate(t_vals):
                demands_arm = []
                for _ in list_attempts:
                    sub_context, user = self.sample_subcontext(t, idx_context)
                    sub_contexts_to_return[idx_arm].append(sub_context)
                    users_to_return.append("Class {}".format(user))
                    t_df_total.append(t % self.season_length)
                    season_df_total.append(self.season(t))
                    if real_demand:
                        demand = self.contexts[idx_context][sub_context].demand(arm, t)
                    else:
                        demand = self.contexts[idx_context][sub_context].noised_demand(arm, t)
                    demands_arm.append(demand)
                avg_demand = np.mean(demands_arm)
                demands.append(avg_demand)
                rewards.append(avg_demand * arm)
                t_df.append(t % self.season_length)
                season_df.append(self.season(t))
                # users_to_return[idx_t].append(user)

        df = pd.DataFrame(columns=["Demand", "Time", "Season", "Price"],
                          data={"Demand": demands, "Season": season_df, "Time": t_df,
                                "Price": np.repeat(arms_to_plot, len(t_vals))})
        colors_palette = [User.hue_map(t / T) for t in df.Time.unique()]
        palette = {
            t: colors_palette[i] for i, t in enumerate(df.Time.unique())
        }
        # demands = np.array(demands).reshape(len(seasons), -1)

        # for t in range(T):
        # current_df = df.where(df["Time"] == t).dropna()
        g = sns.FacetGrid(df, size=10, col="Season", hue="Time", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Demand").add_legend()
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        df = pd.DataFrame(columns=["Time", "Season", "Class"],
                          data={"Season": season_df_total, "Time": t_df_total,
                                "Class": users_to_return})
        colors_palette = [User.hue_map(t / T) for t in df.Time.unique()]
        palette = {
            t: colors_palette[i] for i, t in enumerate(df.Time.unique())
        }
        # demands = np.array(demands).reshape(len(seasons), -1)

        # for t in range(T):
        # current_df = df.where(df["Time"] == t).dropna()
        g = sns.FacetGrid(df, size=10, row="Season", col="Time", legend_out=True)
        g = g.map(sns.countplot, "Class")
        # for t in t_vals:
            # current_df = df.where(df["Time"] == t).dropna()
            # sns.countplot(data=current_df, x="Class")
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


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
        return demand_val

    def optimum(self, t, season=None):
        all_demands = [self.demand(arm, t, season) * arm for arm in self.arms]
        return np.max(all_demands), np.argmax(all_demands)

    def season(self, t):
        return int((t / self.season_length)) % self.number_of_seasons

    @staticmethod
    def hue_map(value):
        HUE_MAX = 0.9
        hue = pow(1 - value, 2) * HUE_MAX
        rgb = colors.hsv_to_rgb((hue, 1, 1))
        hex = colors.to_hex(rgb)
        return hex

    def plot(self, T):
        bins = np.linspace(0, max(self.arms))
        delta_t = 89
        first_t = 0
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
        colors_palette = [User.hue_map(t / T) for t in df.Time.unique()]
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
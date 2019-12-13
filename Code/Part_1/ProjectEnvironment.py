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
env_dir = outputs_dir+"sw_ucb_v3/"

import pandas as pd

from Code.Part_1.Environment import Environment


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
    def __init__(self, arms, number_users_function, sigma, users_matrix_parameters, context_alternatives, batch_size = 16, season_length = 91, number_of_seasons = 4, num_users = 3):
        self.arms = arms
        self.n_arms = len(self.arms)
        self.number_users_function = number_users_function
        self.sigma = sigma
        self.batch_size = batch_size
        self.season_length = season_length
        self.number_of_season = number_of_seasons
        self.num_users = num_users
        self.best_contexts = []
        """
        List of contexts.
        A single context is a list of subcontexts.
        A single subcontexts is a list of users.
        Each subcontext is associated to a function to learn.
        """
        self.contexts = []
        """
        
        """
        for idx_c, context_alternative in enumerate(context_alternatives):
            self.contexts.append([])
            for idx_s, subcontexts in enumerate(context_alternative):
                subcontext_parameters = [params for idx_p, params in enumerate(users_matrix_parameters) if idx_p in subcontexts]# users_matrix_parameters[idx_s]
                self.contexts[idx_c].append(User("{} {}".format(idx_c, idx_s), arms, subcontext_parameters, sigma, self.season_length, self.number_of_season, subcontexts))
        self.regret = []
        self.real_rewards = []
        self.drawn_users = []
        self.selected_context = 0
        self.contexts_alternatives = context_alternatives

    @staticmethod
    def get_env_dir():
        return env_dir

    """
    Given a subcontext, aggregates the curves related to it
    """
    def aggregate_curves(self, subcontext):
        pass

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
        rewards, rewards_context_arm, users = [[] for _ in self.contexts], [[] for _ in self.contexts], []
        batch = range(self.batch_size)
        # sample users <batch_size> times
        current_probabilities = self.probabilities(t)
        for _ in batch:
            user = self.sample_user(t)
            users.append(user)

        users = np.array(users)

        for idx_c, sub_contexts in enumerate(self.contexts_alternatives):
            for _ in self.arms:
                rewards[idx_c].append([])
                rewards_context_arm[idx_c].append([])

        # for each context
        for idx_c, context in enumerate(self.contexts):
            sub_contexts_alternatives_for_current_context = self.contexts_alternatives[idx_c]
            # for each arm
            for idx_arm, arm in enumerate(self.arms):
                # retrieve batch of demands corresponding to each sub context
                demands = [[] for _ in sub_contexts_alternatives_for_current_context]
                for user in users:
                    sub_context = self.get_subcontext_from_user(sub_contexts_alternatives_for_current_context, user)
                    demands[sub_context].append(context[sub_context].weighted_update_samples(arm, current_probabilities, t))
                # update reward for each sub_context by averaging those available
                for sub_context, users_in_sub_context in enumerate(sub_contexts_alternatives_for_current_context):
                    rewards[idx_c][idx_arm].append(np.mean(demands[sub_context]) * arm)  # demand

        return rewards, users

    """
    Given a single arm, it pulls more arms with associated users.
    Called after round_context to compute the regret, as
    the learners need the execution of round_context to know what's going on
    """
    def round_for_arm_old(self, idx_pulled_arms, t, drawn_users):
        sub_contexts_for_current_context = self.contexts[self.selected_context]
        sub_contexts_alternatives_for_current_context = self.contexts_alternatives[self.selected_context]
        best_reward_tot, real_reward_tot = 0, 0
        for idx_s, (subcontext, idx_pulled_arm) in enumerate(zip(sub_contexts_alternatives_for_current_context, idx_pulled_arms)):
            price = self.arms[idx_pulled_arm]
            # take rewards associated to this subcontext
            idxs_reward_current_user = [idx for idx, u in enumerate(drawn_users) if u in subcontext]
            best_rewards_batch, real_rewards_batch = [], []
            best_price_batch, real_price_batch = [], []
            for idx_drawn in idxs_reward_current_user:
                idx_user = drawn_users[idx_drawn]
                current_best_reward, current_best_price = sub_contexts_for_current_context[idx_s].optimum_aggregate(t)
                best_rewards_batch.append(current_best_reward)

                real_sample = sub_contexts_for_current_context[idx_s].weighted_aggregate_demand(price, self.probabilities(t), t)
                current_real_reward = real_sample * price
                real_rewards_batch.append(current_real_reward)
                best_price_batch.append(self.arms[current_best_price])
                real_price_batch.append(price)
            best_reward_tot += np.mean(best_rewards_batch)
            real_reward_tot += np.mean(real_rewards_batch)

        self.drawn_users.append(drawn_users)
        self.regret.append(best_reward_tot - real_reward_tot)
        self.real_rewards.append(real_reward_tot)
        return real_reward_tot

    def round_for_arm(self, idx_pulled_arms, t, save_best_ctx):
        probabilities = self.probabilities(t)
        best_reward_tot, best_ctx = self.get_current_best_context(t)
        sub_contexts_for_current_context = self.contexts[self.selected_context]
        sub_contexts_alternatives_for_current_context = self.contexts_alternatives[self.selected_context]
        real_reward_tot = 0
        if save_best_ctx:
            self.best_contexts.append(best_ctx)

        for idx_s, (subcontext, idx_pulled_arm) in enumerate(zip(sub_contexts_alternatives_for_current_context, idx_pulled_arms)):
            price = self.arms[idx_pulled_arm]
            users_current_subcontext = sub_contexts_alternatives_for_current_context[idx_s]
            probability_current_subcontext = np.sum(
                [p for u, p in enumerate(probabilities) if u in users_current_subcontext])

            # takes reward associated to this subcontext for pulled arm
            real_sample = sub_contexts_for_current_context[idx_s].weighted_aggregate_demand(price, probabilities, t)
            current_real_reward = real_sample * price
            real_reward_tot += current_real_reward * probability_current_subcontext

        self.regret.append(best_reward_tot - real_reward_tot)
        self.real_rewards.append(real_reward_tot)
        return real_reward_tot

    """
    Computes best context and value
    """
    def get_current_best_context(self, t):
        probabilities = self.probabilities(t)
        best_rewards_for_context = []
        for idx_c, context in enumerate(self.contexts):
            sub_contexts_for_current_context = self.contexts[idx_c]
            sub_contexts_alternatives_for_current_context = self.contexts_alternatives[idx_c]
            best_reward_current_ctx = 0
            for idx_s, subcontext in enumerate(sub_contexts_for_current_context):
                users_current_subcontext = sub_contexts_alternatives_for_current_context[idx_s]
                probability_current_subcontext = np.sum(
                    [p for u, p in enumerate(probabilities) if u in users_current_subcontext])
                reward, _ = subcontext.weighted_optimum_aggregate(t, probabilities)
                best_reward_current_ctx += reward * probability_current_subcontext
            best_rewards_for_context.append(best_reward_current_ctx)

        return np.max(best_rewards_for_context), np.argmax(best_rewards_for_context)
    """
    Return, given T, best rewards for each t
    """
    def best_rewards_t(self, T, users):
        best_rewards = []
        for t in range(1, T+1):
            for idx_c, context in enumerate(self.contexts):
                context_optimal = 0
                contexts_optimals = []
                optimum, _ = users[subcontext].optimum(t)
                contexts_optimals.append(optimum)
                for user, p in zip(context, self.probabilities_context(idx_c, t)):
                    optimum, optimum_arm = user.optimum(t)
                    context_optimal += optimum
                contexts_optimals.append(context_optimal)
            best_rewards.append(np.max(contexts_optimals))
        return best_rewards
    """
    Samples given a context ID. Returns both user and subcontext
    """
    def sample_subcontext(self, t, context = None):
        if context is None:
            context = self.selected_context
        sub_contexts = self.contexts_alternatives[context]
        probabilities = self.probabilities(t)
        user = np.random.choice(len(probabilities), 1, p=probabilities)[0]
        sub_context = self.get_subcontext_from_user(sub_contexts, user)
        return sub_context, user, probabilities
    """
    Samples user regardless from the context
    """
    def sample_user(self, t):
        probabilities = self.probabilities(t)
        user = np.random.choice(len(probabilities), 1, p=probabilities)[0]
        return user
    """
    Returns, given a user and the sub_contexts, the idx in sub_contexts where the user is contained.
    """
    def get_subcontext_from_user(self, sub_contexts, user):
        return [idx_s for idx_s, subcontext in enumerate(sub_contexts) if user in subcontext][0]
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

    def plot_distribution(self, idx_context = 4, t_vals = [0, 40, 80, 91, 131, 171, 182, 222, 262, 273, 313, 353]):
        attempts_number = 256
        list_attempts = list(range(attempts_number))
        users_to_return = []
        t_df, season_df, t_df_total, season_df_total = [], [], [], []

        for idx_t, t in enumerate(t_vals):
            for _ in list_attempts:
                _, user, _ = self.sample_subcontext(t, idx_context)
                users_to_return.append("Class {}".format(user))
                t_df_total.append(t % self.season_length)
                season_df_total.append(self.season(t))
            t_df.append(t % self.season_length)
            season_df.append(self.season(t))

        users_seasons_times = zip(users_to_return, season_df_total, t_df_total)
        users_seasons_times = sorted(users_seasons_times)
        users_to_return, season_df_total, t_df_total = [u for u, s, t in users_seasons_times], [s for u, s, t in users_seasons_times], [t for u, s, t in users_seasons_times]
        df = pd.DataFrame(columns=["Time", "Season", "Class"],
                          data={"Season": season_df_total, "Time": t_df_total,
                                "Class": users_to_return})

        g = sns.FacetGrid(df, size=10, row="Season", col="Time", legend_out=True)
        g = g.map(sns.countplot, "Class")
        plt.show()

    def plot_aggregate(self, t_vals = [0, 91, 182, 273]):
        arms_df, season_df, season_even_df, season_odd_df, t_df, demands_df, rewards_df = [], [], [], [], [], [], []

        for idx_t, t in enumerate(t_vals):
            for arm in self.arms:
                demand_tot, reward_tot = 0, 0
                idx_context = 0

                sub_contexts_for_current_context = self.contexts[idx_context]
                sub_contexts_alternatives_for_current_context = self.contexts_alternatives[idx_context]
                probabilities = self.probabilities(t)

                for idx_s, subcontext in enumerate(sub_contexts_alternatives_for_current_context):
                    demand = sub_contexts_for_current_context[idx_s].weighted_aggregate_demand(arm, probabilities, t)
                    current_real_reward = demand * arm

                    users_current_subcontext = sub_contexts_alternatives_for_current_context[idx_s]
                    probability_current_subcontext = np.sum(
                        [p for u, p in enumerate(probabilities) if u in users_current_subcontext])

                    reward_tot += current_real_reward * probability_current_subcontext
                    demand_tot += demand * probability_current_subcontext

                arms_df.append(arm)
                t_df.append(t % self.season_length)
                season = self.season(t)
                if season == 0:
                    row, column = 0, 0
                if season == 1:
                    row, column = 0, 1
                if season == 2:
                    row, column = 1, 0
                if season == 3:
                    row, column = 1, 1
                season_df.append(season)
                season_even_df.append(row)
                season_odd_df.append(column)
                demands_df.append(demand_tot)
                rewards_df.append(reward_tot)

        df = pd.DataFrame(columns=["Demand", "Reward", "Time", "Season", "Price", "idx_context", "row", "column"],
                          data={"Demand": demands_df, "Reward": rewards_df, "row": season_even_df, "column": season_odd_df, "Season": season_df, "Time": t_df,
                                "Price": arms_df})

        colors_palette = [User.hue_map((t + 1) / 363) for t in t_vals]
        palette = {
            t: colors_palette[idx_t] for idx_t, t in enumerate(t_vals)
        }

        g = sns.FacetGrid(df, size=10, col="column", row="row")
        g = g.map(sns.lineplot, "Price", "Reward")
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        g = sns.FacetGrid(df, size=10, col="column", row="row")
        g = g.map(sns.lineplot, "Price", "Demand")
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def plot_contexts(self):
        attempts_number = 32
        list_attempts = list(range(attempts_number))
        resolution = 64
        arms_in_plots = self.arms# np.linspace(0, max(self.arms), resolution)
        demands, rewards, sub_contexts_to_return, users_to_return = [], [], [[] for _ in arms_in_plots], []
        t_df, season_df, t_df_total, season_df_total, arms_to_plot, idx_context_df = [], [], [], [], [], []
        t = 1
        for idx_context, subcontexts_current_context in enumerate(self.contexts):
            for idx_arm, arm in enumerate(arms_in_plots):
                demands_arm = []
                demands_subcontext = [[] for _ in subcontexts_current_context]
                numbers_subcontext = [0 for _ in subcontexts_current_context]
                for _ in list_attempts:
                    sub_context, user, current_probabilities = self.sample_subcontext(t, idx_context)
                    sub_contexts_to_return[idx_arm].append(sub_context)
                    users_to_return.append("Class {}".format(user))
                    t_df_total.append(t % self.season_length)
                    season_df_total.append(self.season(t))
                    demand = subcontexts_current_context[sub_context].aggregate_demand(arm, t)
                    demands_arm.append(demand)
                    demands_subcontext[sub_context].append(demand)
                    numbers_subcontext[sub_context] += 1
                avg_demand = 0
                for d_s, n_s in zip(demands_subcontext, numbers_subcontext):
                    avg_demand += np.mean(d_s) * (n_s / np.sum(numbers_subcontext))
                demands.append(avg_demand)
                rewards.append(avg_demand * arm)
                t_df.append(t % self.season_length)
                season_df.append(self.season(t))
                idx_context_df.append(idx_context)
                arms_to_plot.append(arm)

        df = pd.DataFrame(columns=["Demand", "Reward", "Time", "Season", "Price", "idx_context"],
                          data={"Demand": demands, "Reward": rewards, "Season": season_df, "Time": t_df,
                                "Price": arms_to_plot, "idx_context": idx_context_df})
        colors_palette = [User.hue_map((idx_context + 1) / 9) for idx_context, _ in enumerate(self.contexts)]
        palette = {
            idx_context: colors_palette[idx_context] for idx_context, _ in enumerate(self.contexts)
        }
        # demands = np.array(demands).reshape(len(seasons), -1)

        # for t in range(T):
        # current_df = df.where(df["Time"] == t).dropna()
        """
        g = sns.FacetGrid(df, size=10, col="Season", hue="Time", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Demand").add_legend()
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        """
        g = sns.FacetGrid(df, size=10, col="Season", hue="idx_context", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Reward").add_legend()
        # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        users_seasons_times = zip(users_to_return, season_df_total, t_df_total)
        users_seasons_times = sorted(users_seasons_times)
        users_to_return, season_df_total, t_df_total = [u for u, s, t in users_seasons_times], [s for u, s, t in users_seasons_times], [t for u, s, t in users_seasons_times]
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

    def plot_single_users(self, real_demand = True, T = 363, t_vals = [1, 101, 201, 301], idx_context = 4):# , 91, 181, 271]):
        attempts_number = 1
        list_attempts = list(range(attempts_number))
        resolution = len(self.arms)
        arms_in_plots = self.arms# np.linspace(0, max(self.arms), resolution)
        df = pd.DataFrame()
        for idx_t, t in enumerate(t_vals):
            t_in_season = t % self.season_length
            season = self.season(t)
            for user in self.contexts_alternatives[idx_context]:
                user = user[0]
                demands, rewards, users = [], [], []
                t_df, season_df, t_df_total, season_df_total, arms_to_plot = [], [], [], [], []
                for idx_arm, arm in enumerate(arms_in_plots):
                    if real_demand:
                        demand = self.contexts[idx_context][self.get_subcontext_from_user(self.contexts_alternatives[idx_context], user)].aggregate_demand(arm, t)
                    else:
                        demand = self.contexts[idx_context][self.get_subcontext_from_user(self.contexts_alternatives[idx_context], user)].aggregate_noised_demand(arm, t)
                    demands.append(demand)
                    rewards.append(demand * arm)

                    t_df.append(t % self.season_length)
                    season_df.append(season)
                    arms_to_plot.append(arm)
                    users.append(user)

                current_df = pd.DataFrame(
                                  data={"Demand": demands, "Rewards": rewards, "Season": season_df, "Time": t_df,
                                        "Price": arms_to_plot, "User": users})

                df = pd.concat([df, current_df])
                # demands = np.array(demands).reshape(len(seasons), -1)

                # for t in range(T):
                # current_df = df.where(df["Time"] == t).dropna()
                """
                g = sns.FacetGrid(df, size=10, col="Season", hue="Time", palette=palette, legend_out=True)
                g = g.map(sns.lineplot, "Price", "Demand").add_legend()
                # sns.lineplot(data = current_df, x="Bins", y="Demand", palette=palette)
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title("Demand of user {}".format(user))
                plt.show()
                """
        colors_palette = [User.hue_map(t / T) for t in df.Time.unique()]
        palette = {
            t: colors_palette[i] for i, t in enumerate(df.Time.unique())
        }
        g = sns.FacetGrid(df, size=10, col="Season", hue="Time", row="User", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Rewards").add_legend()
        plt.title("Reward of user {}, time {}, season {}, context {}".format(user,t,season, idx_context))
        g = sns.FacetGrid(df, size=10, col="Season", hue="Time", row="User", palette=palette, legend_out=True)
        g = g.map(sns.lineplot, "Price", "Demand").add_legend()
        plt.title("Reward of user {}, time {}, season {}, context {}".format(user, t, season, idx_context))

    plt.show()


"""
A single user identifies a function.
In case of contexts, it may be associated to a set of users.
"""
class User():
    def __init__(self, idx, arms, parameters_per_user, sigma, season_length, number_of_seasons, users):
        self.phase_function = parameters_per_user # made of [(s, D, mu_values, f_smooth)
        self.samples = []
        self.sigma = sigma
        self.idx = idx
        self.arms = arms
        self.n_arms = len(self.arms)
        self.season_length = season_length
        self.number_of_seasons = number_of_seasons
        self.users = users

    def n_samples(self):
        return len(self.samples)

    def update_samples(self, price, t):
        sample = self.aggregate_noised_demand(price, t)
        self.samples.append(sample)
        return sample

    def weighted_update_samples(self, price, probabilities, t):
        sample = self.weighted_aggregate_noised_demand(price, probabilities, t)
        self.samples.append(sample)
        return sample

    #TODO inutile
    def aggregate_noised_demand(self, price, t):
        return max(0, self.aggregate_demand(price, t) + np.random.normal(0, self.sigma ** 2))

    def weighted_aggregate_noised_demand(self, price, probabilities, t):
        return max(0, self.weighted_aggregate_demand(price, probabilities, t) + np.random.normal(0, self.sigma ** 2))

    def noised_demand(self, price, t, user):
        return max(0, self.demand(price, t, user) + np.random.normal(0, self.sigma ** 2))

    # TODO devo farla? ragionare, in teoria si, e dopo chiamare questa. Il resto sembra corretto
    def aggregate_demand(self, price, t, season = None):
        return np.sum([self.demand(price, t, user, season) for user in self.users])

    """
    Weighted aggregate demand.
    Here I am considering a sub context.
    Reminder: a sub context is made of more users.
    I have to weight the demand wrt the probability of the user inside the sub context.
    probabilities: distribution of all users. I filter them with current_probabilities
    The weight is probability / sum(current_probabilities)
    """
    def weighted_aggregate_demand(self, price, probabilities, t, season = None):
        current_probabilities = [probabilities[user] for user in self.users]
        tot_probabilities = np.sum(current_probabilities)
        return np.sum([self.demand(price, t, user, season) * (probability / tot_probabilities) for user, probability in zip(self.users, current_probabilities)])

    def demand(self, price, t, user, season=None):
        # idx_user_in_subcontext identifies the idx of the user inside this subcontext
        if season is None:
            season = self.season(t)
        demand_val = 0
        # for phase_function in self.phase_function:
        idx_user_in_subcontext = [i for i, u in enumerate(self.users) if user == u][0]
        phase_function = self.phase_function[idx_user_in_subcontext]
        current_function = phase_function[season]
        demand_val += np.exp(price * (-1) * current_function[0]) * current_function[1]
        for mu in current_function[2]:
            demand_val += current_function[3](t, mu, price)
        return demand_val

    def optimum(self, t, user, season=None):
        all_rewards = [self.demand(arm, t, user, season) * arm for arm in self.arms]
        return np.max(all_rewards), np.argmax(all_rewards)

    def optimum_aggregate(self, t, season=None):
        all_rewards = [self.aggregate_demand(arm, t, season) * arm for arm in self.arms]
        return np.max(all_rewards), np.argmax(all_rewards)

    def weighted_optimum_aggregate(self, t, probabilities, season=None):
        all_rewards = [self.weighted_aggregate_demand(arm, probabilities, t, season) * arm for arm in self.arms]
        return np.max(all_rewards), np.argmax(all_rewards)

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
        bins = self.arms# np.linspace(0, max(self.arms))
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
                user = None
                demands.append(self.demand(p, t, user, s))
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
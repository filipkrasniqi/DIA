"""
Class containing the definition of a context.
For each subcontext (i.e., the set of users to which I associate a function to learn)
there is a learner
"""
class ContextLearner:
    def __init__(self, subcontexts, learner_constructor, arms, batch_size, window_length=None, idx_c = -1, sigma = -1):
        self.subcontexts = subcontexts
        self.learners = [learner_constructor(arms, idx_c, idx_s, sigma, batch_size, window_length) for idx_s, s in enumerate(subcontexts)]
        self.idx_c = idx_c
    """
    Returns number of learners, i.e., number of subcontexts
    """
    def getNumberLearners(self):
        return len(self.learners)
    """
    Returns learner given idx
    """
    def get_learner(self, idx_learner):
        return self.learners[idx_learner]
    """
    Pulls arm given a user. Needs also the rewards that each arm would provide,
    as a ContextLearner contains N learners, one for each subcontext,
    and the rewards for each arm would be different.
    Demands are useful only for plotting reasons (to show what has been estimated).
    """
    def pull_arm(self, t, rewards_per_arm, demands_per_arm, users):
        idx_arms = []
        real_reward = 0
        for idx_learner, subcontext in enumerate(self.subcontexts):
            # take rewards associated to this subcontext
            idxs_reward_current_user = [idx for idx, u in enumerate(users) if u in subcontext]
            rewards_per_arm_current_subcontext = rewards_per_arm[idxs_reward_current_user, :]
            demands_per_arm_current_subcontext = demands_per_arm[idxs_reward_current_user, :]
            users_current_subcontext = users[idxs_reward_current_user]
            idx_arm = self.learners[idx_learner].pull_arm(rewards_per_arm_current_subcontext, demands_per_arm_current_subcontext, users_current_subcontext, t)
            idx_arms.append(idx_arm)
            real_reward += rewards_per_arm_current_subcontext[idx_arm]
        # TODO idx_arms vanno tornati in ordine di sampling
        return idx_arms, real_reward

    """
    Plots learners of the ContextLearner (one for each subcontext)
    """
    def plot(self, env):
        learners_plots = []
        for learner in self.learners:
            learners_plots.append(learner.plot(env))
        return learners_plots

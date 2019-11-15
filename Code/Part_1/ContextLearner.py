"""
Class containing the definition of a context.
For each subcontext (i.e., the set of users to which I associate a function to learn)
there is a learner
"""
class ContextLearner:
    def __init__(self, subcontexts, learner_constructor, arms, window_length=None, idx_c = -1, sigma = -1):
        self.subcontexts = subcontexts
        self.learners = [learner_constructor(arms, idx_c, idx_s, sigma, window_length) for idx_s, s in enumerate(subcontexts)]
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
    def pull_arm(self, t, rewards_per_arm, demands_per_arm, user):
        idx_arms = []
        for reward, user in zip(rewards_per_arm, user):
            idx_learner = [i for i, s in enumerate(self.subcontexts) if user in s][0]
            idx_arm = self.learners[idx_learner].pull_arm(rewards_per_arm, demands_per_arm, user, t)
            idx_arms.append(idx_arm)
        return idx_arms
    """
    Plots learners of the ContextLearner (one for each subcontext)
    """
    def plot(self, env):
        learners_plots = []
        for learner in self.learners:
            learners_plots.append(learner.plot(env))
        return learners_plots

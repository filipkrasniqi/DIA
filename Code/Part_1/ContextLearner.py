"""
Class containing the definition of a context.
For each subcontext (i.e., the set of users to which I associate a function to learn)
there is a learner
"""


class ContextLearner:
    def __init__(self, subcontexts, learner_constructor, arms, window_length=None, idx_c = -1):
        self.subcontexts = subcontexts
        self.learners = [learner_constructor(arms, window_length, idx_c, idx_s) for idx_s, s in enumerate(subcontexts)]
        self.idx_c = idx_c

    def getNumberLearners(self):
        return len(self.learners)

    def update(self, arm, reward, user):
        idx_learner = [i for i, s in enumerate(self.subcontexts) if user in s][0]
        self.learners[idx_learner].update(arm, reward, user)

    def get_learner(self, idx_learner):
        return self.learners[idx_learner]

    def pull_arm(self, t, rewards_per_arm, user):
        idx_arms = []
        for reward, user in zip(rewards_per_arm, user):
            idx_learner = [i for i, s in enumerate(self.subcontexts) if user in s][0]
            idx_arm = self.learners[idx_learner].pull_arm(rewards_per_arm, user, t)
            idx_arms.append(idx_arm)
        return idx_arms

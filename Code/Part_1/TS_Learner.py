import numpy as np

from Code.Part_1.Learner import Learner

class TS_Learner(Learner):
    def __init__(self,arms):
        super().__init__(arms)
        # 2 parameter for each arm
        self.beta_parameters = np.ones((self.n_arms,2))

    def pull_arm(self, env, t):
        idx_arm = np.argmax(np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1]))
        Learner.pull_arm(self, env, t, idx_arm)
        return idx_arm

    def update(self,pulled_arm,reward, user):
        self.t += 1
        # TODO not ok to compute with reward. We must just add whether choice was correct or not to correct (alfa, beta)
        self.update_observations(pulled_arm,reward, user)
        self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + 1.0 - reward
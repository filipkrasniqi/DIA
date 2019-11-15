import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary_Environment import *
from TS_Learner import *
from SWTS_Learner import *

n_arms = 4
p = np.array([ [0.15,0.1,0.2,0.35],[0.35,0.21,0.2,0.35],[0.5,0.1,0.1,0.15],[0.5,0.1,0.1,0.15],[0.8,0.21,0.1,0.15]] )

T = 400

n_experiments = 100
ts_reward_per_experiment = []
swts_reward_per_experiment = []
window_size = int(np.sqrt(T))

for e in range(0,n_experiments):
    ts_env = Non_Stationary_Environment(n_arms,probabilities=p,horizon = T)
    ts_learner = TS_Learner(n_arms = n_arms)

    swts_env = Non_Stationary_Environment(n_arms,probabilities=p,horizon = T)
    swts_learner = SWTS_Learner(n_arms = n_arms,window_size = window_size)

    for t in range(0,T):
        pulled_arm = ts_learner.pull_arms()
        reward = ts_env.execute_round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        pulled_arm = swts_learner.pull_arms()
        reward = swts_env.execute_round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    swts_reward_per_experiment.append(swts_learner.collected_rewards)

ts_istantaneous_regret = np.zeros(T)
swts_istantaneous_regret = np.zeros(T)
n_phases = len(p)
phases_len = int (T/n_phases)
opt_per_phases = p.max(axis = 1)
optimum_per_round = np.zeros(T)

for i in range(0,n_phases):
    optimum_per_round[ i*phases_len : (i+1)*phases_len ] = opt_per_phases[i]
    ts_istantaneous_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(ts_reward_per_experiment, axis = 0 )[i*phases_len: (i+1)*phases_len]
    swts_istantaneous_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(swts_reward_per_experiment, axis = 0 )[i*phases_len: (i+1)*phases_len]

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_reward_per_experiment, axis=0),'r')
plt.plot(np.mean(swts_reward_per_experiment, axis=0),'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS","SW-TS","Optimum"])
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.consum(ts_istantaneous_regret),'r')
plt.plot(np.consum(swts_istantaneous_regret),'b')
plt.legend(["TS","SW-TS"])
plt.show()
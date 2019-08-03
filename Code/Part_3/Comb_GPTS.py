from Part_3.GPTS_Learner import *
from Part_3.Environment import *
# Punto 3

n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 20
total_budget = 10

min_daily_budget = 0.0
max_daily_budget = 3.0

arms = np.linspace(min_daily_budget, max_daily_budget, n_arms_sub)

sigma_env = 10
T = 100

#gpts_rewards_per_experiment_sub_1 = []
#gaussian_error_per_experiment_1 = []

env = Environment(n_sub_campaign,arms,sigma_env,n_users_x_sub_campaign,total_budget)

gpts_learners = []
for i in range(0,n_sub_campaign):
    gpts_learners.append( GPTS_Learner(n_arms=n_arms_sub, arms=arms) )

#gpts_errors = []

for t in range(0, T):
    # pulled_arm = gpts_learner.pull_arm()

    # --> Costruire matrice da passare
    # --> chiamare algoritmo Filip

    reward = env.round(pulled_arm)
    gpts_learner.update(pulled_arm, reward)


    # gpts_errors.append(np.max( np.absolute(env.means - gpts_learner.means) ))

# gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)


#gaussian_error_per_experiment.append(gpts_errors)

#plt.figure(0)
#plt.xlabel("t")
#plt.ylabel("Regression Error")
#plt.plot(np.mean(gaussian_error_per_experiment, axis=0), 'g')
#plt.legend(["GPTS"])
#plt.show()

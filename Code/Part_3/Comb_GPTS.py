from Part_3.GPTS_Learner import *
from Part_3.Environment import *
from Part_3.dp_algorithm import *
# Punto 3

def built_matrix_sub_budget_clicks(arms,n_sub_campaign,gpts_learners):
    matrix = []
    for i in range(0,n_sub_campaign):
        vet = []
        for j in range(0,arms):
            vet.append(gpts_learners[i].means[j])

        matrix.append(vet)

    return matrix

def built_matrix_sub_budget_clicks_without_errors(arms,n_sub_campaign,env):
    matrix = []
    for i in range(0, n_sub_campaign):
        vet = []
        for j in range(0, arms):
            vet.append(env.means[j])

        matrix.append(vet)

    return matrix

n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 20
total_budget = 10

min_daily_budget = 0.0
max_daily_budget = total_budget

sigma_env = 5
bid=10,
prob_users=[
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10]
    ]
T = 100

gpts_rewards_per_experiment_sub_1 = []
gaussian_error_per_experiment_1 = []

env = Environment(n_arms_sub, n_users_x_sub_campaign, n_sub_campaign,total_budget,bid,prob_users  ,sigma_env)
arms = env.get_arms()

# Val ottimo per calcolare Regret
matrix = built_matrix_sub_budget_clicks_without_errors(arms,n_sub_campaign,env)
optimum = alg_filip.Val_ottimo(arms,n_sub_campaign,matrix,min_daily_budget,total_budget)

gpts_learners = []
for i in range(0,n_sub_campaign):
    gpts_learners.append( GPTS_Learner(n_arms=n_arms_sub, arms=arms) )

rewards_per_round = []

for t in range(0, T):
    matrix = built_matrix_sub_budget_clicks(arms,n_sub_campaign,gpts_learners)
    pulled_arms = alg_filip (arms,n_sub_campaign,matrix,min_daily_budget,total_budget)
    rewards = env.round(pulled_arms)
    for i in range(0, n_sub_campaign):
        gpts_learners[i].update(pulled_arms[i],rewards[i])

    rewards_per_round.append(np.sum(rewards))

    # gpts_errors.append(np.max( np.absolute(env.means - gpts_learner.means) ))

# gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plot1 = np.cumsum(optimum - rewards_per_round)
plt.plot(plot1,'r')

#gaussian_error_per_experiment.append(gpts_errors)

#plt.figure(0)
#plt.xlabel("t")
#plt.ylabel("Regression Error")
#plt.plot(np.mean(gaussian_error_per_experiment, axis=0), 'g')
#plt.legend(["GPTS"])
#plt.show()

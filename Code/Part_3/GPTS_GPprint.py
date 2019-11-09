import matplotlib.pyplot as plt
from Code.Part_3.GPTS_Learner import *
from Code.Part_3.Environment_vecchio import *

n_sub_campaign = 5
n_users_x_sub_campaign = 3
n_arms_sub = 21
total_budget = 100

min_daily_budget = 0.0
max_daily_budget = total_budget

sigma_env = 1
bid = 10
prob_users = [
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10],
    [0.80, 0.10, 0.10]
]
T = 50

env = Environment(n_arms_sub, n_users_x_sub_campaign, n_sub_campaign, total_budget, bid, prob_users, sigma_env)
arms = env.get_arms()

initial_sigmas = 1
gpts_learner = GPTS_Learner(n_arms=n_arms_sub, arms=arms,sigma_gp =sigma_env,initial_sigmas=initial_sigmas)
arm_obs = np.array([])

real_function_y = []
for a in arms:
    real_function_y.append(env.get_clicks_real(a, 0))

for t in range(0,T):
    # GP TS
    pulled_arm = gpts_learner.pull_arm()
    p = gpts_learner.arms[pulled_arm]
    reward = env.get_click_noise(p,0)
    gpts_learner.update(pulled_arm, reward)

    arm_obs = np.append(arm_obs,arms[pulled_arm])

    X = np.atleast_2d(arm_obs).T
    Y = gpts_learner.collected_rewards.ravel()
    x_pred = np.atleast_2d(arms).T

    y = np.array(gpts_learner.means)
    sigmaGP = np.array(gpts_learner.sigmas)

    plt.figure(t)
    plt.plot(x_pred, real_function_y, 'r:', label=r'$Real Function$')
    plt.plot(X.ravel(), Y, 'ro', label=u'Observed Arms')
    plt.plot(x_pred, y, 'b-', label=u'Predicted Rewards')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y - 1.96 * sigmaGP,( y + 1.96 * sigmaGP)[::-1]]),
             alpha=0.5, fc='b', ec='None', label='95% conf interval')

    plt.xlabel('$x$')
    plt.ylabel('$Real Function$')
    plt.legend(loc='lower right')
    plt.show()

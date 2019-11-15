import os, pickle

from Code.Part_1.Learner import Learner

curr_dir = os.getcwd()
outputs_dir = curr_dir+"/outputs/"
env_dir = outputs_dir+"v05_without_context/"

results = pickle.load(open("{}/results.pickle".format(env_dir), 'rb'))
sigmas = results["sigma"]
for learner_name in [key for key in results.keys() if "sigma" not in key]:
    print(learner_name)
    sigmas, output_per_sigma = results[learner_name][0], results[learner_name][1]
    for sigma, output_per_sigma in zip(sigmas, output_per_sigma):
        (_, real_rewards, regret_history, cumulative_regret_history, (idx_c, idx_s, demand_mapping)) = output_per_sigma
        x = list(range(len(real_rewards)))
        Learner.plot_regret_reward(x, real_rewards, regret_history, cumulative_regret_history, learner_name,
                                   sigma)
        Learner.plot_regret_reward(x, real_rewards, regret_history, cumulative_regret_history, learner_name,
                               sigma, also_cumulative=True)

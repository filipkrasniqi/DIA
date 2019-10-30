import math

from Code.Part_1.ProjectEnvironment import ProjectEnvironment as Environment

probabilities = [0.4, 0.3, 0.3]
sigma_env_n = [2]
T = 730


def linear(x):
    return 1


matrix_parameters = [
    [[0.1, 5, linear], [10, 50, linear], [1, 500, linear], [1, 5000, linear]],
]

n_arms = math.ceil(math.pow(T * math.log(T, 10), 0.25))

for sigma in sigma_env_n:
    env = Environment(n_arms, probabilities, sigma, matrix_parameters)
    env.plot(n_arms)
import pymc as pm
import numpy as np
import arviz as az
from matplotlib import pyplot as plt


def solve():

    observed_sound_levels = [56, 60, 58, 55, 57, 59, 61, 56, 58, 60]
    x_mean = np.mean(observed_sound_levels)  # x should be the mean of the observed_data

    with pm.Model() as solver_model:
        # prior mu
        mu = pm.Normal('mu', mu=x_mean, sigma=100)

        # prior sigma
        sigma = pm.HalfNormal('sigma', sigma=10)

        # likelihood X
        X = pm.Normal('X', mu=mu, sigma=sigma, observed=observed_sound_levels)  # trebuie adaugate si datele observate

        trace = pm.sample(1000, return_inferencedata=True)

    az.plot_trace(trace, var_names=['mu', 'sigma'])
    plt.show()

    az.plot_posterior(trace, var_names=['mu', 'sigma'], hdi_prob=0.95)  # vrem HDI 95
    plt.show()


if __name__ == '__main__':
    solve()

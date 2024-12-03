import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt


def generate_data(observations=100, lower_bound_experience=0, upper_bound_experience=20):

    true_alpha = 3.5
    true_beta = 1.12
    true_epsilon = np.random.normal(0, 0.8, size=observations)

    raw_x = np.random.normal(10, 3, observations)
    clipped_x = np.clip(raw_x, lower_bound_experience, upper_bound_experience)

    true_y = true_alpha + true_beta * raw_x
    y = true_y + true_epsilon

    print('alpha = ', true_alpha)
    print('beta = ', true_beta)
    print('epsilon.mean = ', np.mean(true_epsilon))

    print('y = ', y)

    _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(clipped_x, y, 'C0.')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y', rotation=0)
    ax[0].plot(clipped_x, true_y, 'k')
    az.plot_kde(y, ax=ax[1])
    ax[1].set_xlabel('y')
    plt.tight_layout()
    # plt.show()

    return clipped_x, y


def solve(x, y):

    y_std = np.std(y)

    with pm.Model() as salary_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        epsilon = pm.HalfNormal('epsilon', sigma=10 * y_std)
        mu = pm.Deterministic('mu', alpha + beta * x)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y)

        trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True)

    az.plot_trace(trace, var_names=['alpha', 'beta', 'epsilon'])
    plt.show()

    print(az.summary(trace))


if __name__ == '__main__':
    x_experience, y_salary = generate_data(100)
    solve(x_experience, y_salary)

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def import_data(file_name='Prices.csv', output_file='computer_prices.csv'):
    df = pd.read_csv(file_name)

    columns = [column for column in df.columns]
    df = df[columns]

    df['Premium'] = df['Premium'].apply(lambda x: 1 if x == 'yes' else 0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    df.to_csv(output_file, header=False, index=False)
    return df

def build_price_model(data):

    """
    y ~ N(miu, sigma)
    miu = alpha + beta1 * x1 + beta2 * x2

    y = df['Price']
    x1 = df['Speed']
    x2 = ln(df['HardDrive'])
    """

    y = data['Price']
    x1 = data['Speed']
    x2 = np.log(data['HardDrive'])

    with pm.Model() as price_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        # mu = mean, sigma = stddev, nu = degrees of freedom (a smaller number translates to 'heavier' tails)
        epsilon = pm.StudentT('epsilon', mu=0, sigma=10, nu=3)
        miu = pm.Deterministic('miu', alpha + beta1 * x1 + beta2 * x2)

        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    # 1b_plots
    """
    az.plot_posterior(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
    plt.show()
    """

    print(az.summary(trace, var_names=['alpha', 'beta1', 'beta2', 'epsilon']))

    """
    posterior = trace.posterior.stack(samples=("chain", "draw"))
    alpha_m = posterior['alpha'].mean().item()
    beta1_m = posterior['beta1'].mean().item()
    beta2_m = posterior['beta2'].mean().item()

    plt.scatter(x1, y, color='blue', alpha=0.5, label='Observed data')
    
    x1_line = np.linspace(x1.min(), x1.max(), 100)
    x2_fixed = x2.mean()  # Fix x2 at its mean for plotting

    y_line = alpha_m + beta1_m * x1_line + beta2_m * x2_fixed
    plt.plot(x1_line, y_line, color='red',
             label=f"Regression line (x2 fixed): y={alpha_m:.2f} + {beta1_m:.2f}x1 + {beta2_m:.2f}x2")

    for a, b1, b2 in zip(posterior['alpha'].values.flatten()[:50],
                         posterior['beta1'].values.flatten()[:50],
                         posterior['beta2'].values.flatten()[:50]):
        y_sample = a + b1 * x1_line + b2 * x2_fixed
        plt.plot(x1_line, y_sample, color='gray', alpha=0.3)

    plt.xlabel("x1")
    plt.ylabel("y")
    plt.title("Linear Regression with Two Predictors")
    plt.legend()
    plt.show()
    """
    return trace

def build_price_model2(data):

    """
    y ~ N(miu, sigma)
    miu = alpha + beta1 * x1 + beta2 * x2

    y = df['Price']
    x1 = df['Speed']
    x2 = ln(df['HardDrive'])
    x3 = df['Premium']
    """

    y = data['Price']
    x1 = data['Speed']
    x2 = np.log(data['HardDrive'])
    x3 = data['Premium']

    with pm.Model() as price_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta3 = pm.Normal('beta3', mu=0, sigma=10)

        # mu = mean, sigma = stddev, nu = degrees of freedom (a smaller number translates to 'heavier' tails)
        epsilon = pm.StudentT('epsilon', mu=0, sigma=10, nu=3)
        miu = pm.Deterministic('miu', alpha + beta1 * x1 + beta2 * x2 + beta3 * x3)

        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    # new_plot
    """
    az.plot_posterior(trace, var_names=['beta1', 'beta2', 'beta3'], hdi_prob=0.95)
    plt.show()
    """

    print(az.summary(trace, var_names=['alpha', 'beta1', 'beta2', 'beta3', 'epsilon']))

# 1_d (1_4)
def predict_price(trace_p, x1, x2):
    """
    :param trace_p: the sampled values from the posterior distribution
    :param x1: processor speed (MHz)
    :param x2: hdd size (MB), after taking the natural logarithm of it
    :return: the predicted price for that computer
    """

    posterior = trace_p.posterior.stack(samples=("chain", "draw"))
    alpha_samples = posterior['alpha'].values.flatten()  # 1D array of all values
    beta1_samples = posterior['beta1'].values.flatten()
    beta2_samples = posterior['beta2'].values.flatten()

    y_pred_samples = alpha_samples + beta1_samples * x1 + beta2_samples * x2

    return y_pred_samples

def predict_price_full_y(trace_p, x1, x2):

    posterior = trace_p.posterior.stack(samples=("chain", "draw"))
    alpha_samples = posterior['alpha'].values.flatten()
    beta1_samples = posterior['beta1'].values.flatten()
    beta2_samples = posterior['beta2'].values.flatten()
    sigma_samples = posterior['epsilon'].values.flatten()

    mu_samples = alpha_samples + beta1_samples * x1 + beta2_samples * x2
    return np.random.normal(mu_samples, sigma_samples)


if __name__ == '__main__':

    df = import_data()
    trace = build_price_model(df)
    build_price_model2(df)

    """
    # 1_d
    x1_in = [33]  # MHz
    x2_in = [np.log(540)]  # Natural log of HDD size in MB

    pred_samples = predict_price(trace, x1_in, x2_in)
    pred_samples_y = predict_price_full_y(trace, x1_in, x2_in)

    pred_mean1 = pred_samples.mean()
    pred_hdi1 = az.hdi(pred_samples, hdi_prob=0.90)

    pred_mean2 = pred_samples_y.mean()
    pred_hdi2 = az.hdi(pred_samples_y, hdi_prob=0.90)

    hdd_size = np.exp(x2_in).item()

    # result = 1587.78,  90% HDI = [1530.57660787 1645.3350707 ]

    print(f"Predicted price (only mu) for a computer with {x1_in} MHz speed and {hdd_size:.2f} MB HDD size"
          f" is {pred_mean1:.2f}, 90% HDI = {pred_hdi1}")
    """

    """
    # result = 1605.95, 90% HDI = [ 554.09555262 2728.04776355]
    print(f"Predicted price (full y) for a computer with {x1_in} MHz speed and {hdd_size:.2f} MB HDD size"
          f" is {pred_mean2:.2f}, 90% HDI = {pred_hdi2}")
    """





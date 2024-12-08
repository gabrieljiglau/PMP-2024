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

def build_price_model(df):

    """
    y ~ N(miu, sigma)
    miu = alpha + beta1 * x1 + beta2 * x2

    y = df['Price']
    x1 = df['Speed']
    x2 = ln(df['HardDrive'])
    """

    y = df['Price']
    x1 = df['Speed']
    x2 = np.log(df['HardDrive'])

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

def predict_price(trace_p, x1_in, x2_in):
    """
    :param trace_p: the sampled values from the posterior distribution
    :param x1_in: processor speed (MZz)
    :param x2_in: hdd size (MB), after taking the natural logarithm of it
    :return: the predicted price for that computer
    """

    posterior = trace_p.posterior.stack(samples=("chain", "draw"))
    alpha_samples = posterior['alpha'].values.flatten()  # 1D array of all values
    beta1_samples = posterior['beta1'].values.flatten()
    beta2_samples = posterior['beta2'].values.flatten()

    prediction_list = []
    for processor_speed, ln_size in zip(x1_in, x2_in):
        y_pred_samples = alpha_samples + beta1_samples * processor_speed + beta2_samples * ln_size
        prediction_list.append(y_pred_samples)

    return prediction_list


if __name__ == '__main__':

    df = import_data()
    trace = build_price_model(df)

    # 1_d
    x1_new = [33]
    x2_new = [np.log(540)]
    predictions = predict_price(trace, x1_new, x2_new)

    for i, (x1_in, x2_in) in enumerate(zip(x1_new, x2_new)):
        pred_samples = predictions[i]
        pred_mean = pred_samples.mean()
        pred_hdi = az.hdi(pred_samples, hdi_prob=0.90)
        hdd_size = np.power(2.713, x2_new)
        print(f"Predicted price for a computer with {str(x1_in)} MHz speed and {str(hdd_size)} Hdd size"
              f"is {pred_mean:.2f}, 90% HDI = {pred_hdi:.2f} ")



















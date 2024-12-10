import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def import_data(file_name='real_estate_data.csv'):
    df = pd.read_csv(file_name)

    return df

def predict_price(trace_p, x1, x2, x3):
    posterior = trace_p.posterior.stack(samples=("chain", "draw"))
    alpha_samples = posterior['alpha'].values.flatten()  # 1D array of all values
    beta1_samples = posterior['beta1'].values.flatten()
    beta2_samples = posterior['beta2'].values.flatten()
    beta3_samples = posterior['beta3'].values.flatten()

    y_pred_samples = alpha_samples + beta1_samples * x1 + beta2_samples * x2 + beta3_samples * x3

    return y_pred_samples


def standardize_data(data):
    scaler = StandardScaler()
    data[['Surface_area', 'Rooms', 'Distance_to_center']] = scaler.fit_transform(data[['Surface_area',
                                                                                       'Rooms', 'Distance_to_center']])
    return data

def build_real_estate_model(data):

    """
    y ~ N(miu, sigma)
    miu = alpha + beta1 * x1 + beta2 * x2 + beta3 * x3

    y = df['Price']
    x1 = df['Surface_area']
    x2 = df['Rooms']
    x3 = df['Distance_to_center']
    """

    data = standardize_data(data)

    y = data['Price']
    x1 = data['Surface_area']
    x2 = data['Rooms']
    x3 = data['Distance_to_center']

    y_std = np.std(y)

    with pm.Model() as price_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        beta3 = pm.Normal('beta3', mu=0, sigma=2)

        # mu = mean, sigma = stddev, nu = degrees of freedom (a smaller number translates to 'heavier' tails)
        epsilon = pm.HalfNormal('epsilon', sigma=10 * y_std)
        miu = pm.Deterministic('miu', alpha + beta1 * x1 + beta2 * x2 + beta3 * x3)

        y_pred = pm.StudentT('y_pred', mu=miu, sigma=epsilon, nu=20, observed=y)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    # plots
    az.plot_posterior(trace, var_names=['beta1', 'beta2', 'beta3'], hdi_prob=0.95)
    plt.show()
    print(az.summary(trace, var_names=['alpha', 'beta1', 'beta2', 'beta3', 'epsilon']))

    return trace


if __name__ == '__main__':
    data_i = import_data()
    trace_i = build_real_estate_model(data_i)

    x1_in = [100]
    x2_in = [2]
    x3_in = [5]

    pred_samples = predict_price(trace_i, x1_in, x2_in, x3_in)

    pred_mean1 = pred_samples.mean()
    pred_hdi1 = az.hdi(pred_samples, hdi_prob=0.90)

    # result = 7567.02, 90% HDI = [7480.33783184 7656.88887327]

    print(f"Predicted price for an apartment with {x1_in} square feet and {x2_in} rooms and "
          f" {x3_in} distance from the center is {pred_mean1:.2f}, 90% HDI = {pred_hdi1}")

    # ce atribut influenteaza cel mai mult decizia ? cel care are asociat scalarul cel mai mare,
    # deci primul, suprafata locuibila

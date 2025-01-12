import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def import_data(file_id='152q2BM5m7Lkvj3GPk4iBRzGdHrTbQLmB'):

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    df = pd.read_csv(url)
    df.to_csv('data1.csv', index=False)
    print(f"Data from {url} saved successfully")

    return df

def generate_500_points(df):

    x_1 = df.iloc[:, 0].values
    y_1 = df.iloc[:, 1].values

    x_new = np.linspace(x_1.min(), x_1.max(), 500)

    interpolator = interp1d(x_1, y_1, kind='cubic')
    y_new = interpolator(x_new)

    # small noise to the interpolated data to simulate real-world variability
    noise = np.random.normal(0, 0.1, size=y_new.shape)
    y_noisy = y_new + noise

    return pd.DataFrame({'x': x_new, 'y': y_noisy})

def standardize_data(x, y, order):
    x_p = np.vstack([x ** i for i in range(1, order + 1)])
    x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
    y_s = (y - y.mean()) / y.std()
    return x_s, y_s

def ex1():

    df = import_data() if not os.path.exists('data1.csv') else pd.read_csv('data1.csv', header=None)

    x_1 = df.iloc[:, 0]
    y_1 = df.iloc[:, 1]
    # Linear model (order = 1)
    x_1s, y_1s = standardize_data(x_1, y_1, order=1)
    with pm.Model() as model_l:
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Scalar
        beta = pm.Normal('beta', mu=0, sigma=10, shape=1)  # Vector of size "order"
        epsilon = pm.HalfNormal('epsilon', 5)  # Scalar
        mu = alpha + beta * x_1s[0]  # Shape: (num_points,)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_l = pm.sample(100, init='jitter+adapt_diag', return_inferencedata=True)

    # Quadratic model (order = 2)
    x_1s, y_1s = standardize_data(x_1, y_1, order=2)
    with pm.Model() as model_q:
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Scalar
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)  # Vector of size "order"
        epsilon = pm.HalfNormal('epsilon', 5)  # Scalar
        mu = alpha + pm.math.dot(beta, x_1s)  # Shape: (num_points,)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_q = pm.sample(100, init='jitter+adapt_diag', return_inferencedata=True)

    # Cubic model (order = 3)
    x_1s, y_1s = standardize_data(x_1, y_1, order=3)
    with pm.Model() as model_c:
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Scalar
        beta = pm.Normal('beta', mu=0, sigma=10, shape=3)  # Vector of size "order"
        epsilon = pm.HalfNormal('epsilon', 5)  # Scalar
        mu = alpha + pm.math.dot(beta, x_1s)  # Shape: (num_points,)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_c = pm.sample(100, init='jitter+adapt_diag', return_inferencedata=True)

    pm.compute_log_likelihood(idata_l, model=model_l)
    pm.compute_log_likelihood(idata_q, model=model_q)
    pm.compute_log_likelihood(idata_c, model=model_c)

    # Compute WAIC and compare models
    cmp_df = az.compare(
        {'model_l': idata_l, 'model_q': idata_q, 'model_c': idata_c},
        method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(f"Comparison DataFrame:\n{cmp_df}")
    az.plot_compare(cmp_df)
    plt.show()


if __name__ == '__main__':
    ex1()

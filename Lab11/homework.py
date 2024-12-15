import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def univariate_logistic_regression(x, y):

    x = np.array(x)

    with pm.Model() as logistic_model:

        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)

        miu = alpha + pm.math.dot(x, beta)
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        decision_boundary = pm.Deterministic('decision_boundary', -alpha/beta)

        y_pred = pm.Bernoulli('y_pred', p=theta, observed=y)

        trace = pm.sample(1000, return_inferencedata=True)

    print(az.summary(trace, var_names=['alpha', 'beta', 'theta', 'decision_boundary']))

    decision_boundary_samples = trace.posterior['decision_boundary'].values.flatten()

    # Generate HDI for the decision boundary samples
    hdi = az.hdi(decision_boundary_samples, hdi_prob=0.94)
    print(hdi)

def import_data(file_id='13sqazcavr0Z0pBS9MihI1kqBnf5hUM2Y'):

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    df = pd.read_csv(url)
    df.to_csv('admission.csv', index=False)
    print(f"Data from {url} saved successfully")

    return df

def build_admission_model(imported_data):

    x = imported_data.iloc[:, 1:].values
    y = imported_data.iloc[:, 0].values

    """
    with pm.Model() as admission_model:

        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=le)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        miu = alpha + pm.math.dot(beta1, x1) + pm.math.dot(beta2, x2)
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        decision_boundary = pm.Deterministic('decision_boundary', (-))

        y_pred = pm.Categorical('y_pred', p=theta, observed=y)
        trace = pm.sample(1000, return_inferencedata=True)

    print(az.summary(trace, var_names=['alpha', 'beta1', 'beta2', 'theta', 'decision_boundary']))
    """


if __name__ == '__main__':

    x_in = [1, 3, 4, 5, 6, 8]
    y_out = [0, 0, 1, 1, 1, 1]

    # univariate_logistic_regression(x_in, y_out)

    data = import_data()
    build_admission_model(data)

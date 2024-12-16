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
    x = imported_data.iloc[:, 1:].values  # GRE and GPA
    y = imported_data.iloc[:, 0].values

    num_predictors = x.shape[1]

    with pm.Model() as admission_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=num_predictors)

        mu = alpha + pm.math.dot(x, beta)
        theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
        decision_boundary = pm.Deterministic('decision_boundary', -alpha/beta[1] - beta[0]/beta[1] * x[:, 0])

        y_pred = pm.Bernoulli('y_pred', p=theta, observed=y)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    """
    boundary_post = trace.posterior['decision_boundary'].values.mean()
    print('decision boundary mean: ', boundary_post)

    idata_1 = trace
    x_1 = x  # GRE and GPA
    y_0 = y  # Admission outcomes
    x_n = ['GRE', 'GPA']  # feature names

    # sort x_1[:,0] (GRE) for plotting
    idx = np.argsort(x_1[:, 0])
    bd = idata_1.posterior['decision_boundary'].mean(("chain", "draw"))[idx]

    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_0], label='Admission Data')

    plt.plot(x_1[:, 0][idx], bd, color='k', label='Mean Decision Boundary')

    az.plot_hdi(x_1[:, 0], idata_1.posterior['decision_boundary'], color='k', fill_kwargs={'alpha': 0.3})

    # Label the plot
    plt.xlabel(x_n[0])  # GRE
    plt.ylabel(x_n[1])  # GPA
    plt.title('Decision Boundary with 94% HDI')
    plt.legend()
    plt.show()
    """

    return trace


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_admission_probability(trace, x_1, x_2):

    posterior = trace.posterior.stack(samples=("chain", "draw"))
    alpha_samples = posterior['alpha'].values.flatten()
    beta_samples = posterior['beta'].values

    y_pred_samples = sigmoid(alpha_samples + np.dot(x_1, beta_samples[0]) + np.dot(x_2, beta_samples[1]))
    print(f"y_probabilities  = {y_pred_samples}")

    return y_pred_samples


if __name__ == '__main__':

    # ex1
    """
    x_in = [1, 3, 4, 5, 6, 8]
    y_out = [0, 0, 1, 1, 1, 1]

    univariate_logistic_regression(x_in, y_out)
    """

    data = import_data()
    trace_p = build_admission_model(data)

    x1 = [550, 3.5]  # 1.0, 90% HDI = [1 1]
    x2 = [500, 3.2]  # 0.6176214828644073, 90% HDI = [0.21141248 1]

    pred_samples = predict_admission_probability(trace_p, x1[0], x1[1])
    pred_mean = pred_samples.mean()
    pred_hdi = az.hdi(pred_samples, hdi_prob=0.90)

    print(f"The mean probability of student with GRE {x1[0]} and GPA {x1[1]} of being admitted to university is "
          f"{pred_mean}, 90% HDI = {pred_hdi}")

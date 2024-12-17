import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_and_normalize_data(filepath):
    data = pd.read_csv(filepath)

    x = data[['Ore_Studiu', 'Ore_Somn']].values
    y = data['Promovare'].values

    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    print("Mean of normalized predictors: ", np.mean(x_normalized, axis=0))
    print("Standard deviation: ", np.std(x_normalized, axis=0))

    return x_normalized, y, data.columns


def build_promotion_model(x, y):
    num_predictors = x.shape[1]

    with pm.Model() as promotion_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=num_predictors)

        mu = alpha + pm.math.dot(x, beta)
        theta = pm.Deterministic('theta', pm.math.sigmoid(mu))

        decision_boundary = pm.Deterministic('decision_boundary', -alpha / beta[1] - beta[0] / beta[1] * x[:, 0])

        y_pred = pm.Bernoulli('y_pred', p=theta, observed=y)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    return trace, decision_boundary


def plot_decision_boundary(trace, x, y):
    decision_boundary_mean = trace.posterior['decision_boundary'].mean(("chain", "draw"))

    boundary_post = trace.posterior['decision_boundary'].values.mean()

    print('Mean decision_boundary', boundary_post)

    idx = np.argsort(x[:, 0])
    bd = decision_boundary_mean[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=['C0' if i == 0 else 'C1' for i in y], label='Students that passed')
    plt.plot(x[:, 0][idx], bd, color='k', label='Mean Decision Boundary')
    az.plot_hdi(x[:, 0], trace.posterior['decision_boundary'], color='k', fill_kwargs={'alpha': 0.3})

    plt.xlabel('Normalized Study Hours')
    plt.ylabel('Normalized Sleep Hours')
    plt.title('Decision Boundary with 94% HDI')
    plt.legend()
    plt.show()


def is_dataset_balanced(y, tolerance=0.1):

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution: ", dict(zip(unique, counts)))

    if abs(counts[0] - counts[1]) < len(y) * tolerance:
        print("The dataset is balanced.")
    else:
        print("The dataset is not balanced.")


def solve(filepath):
    x, y, column_names = load_and_normalize_data(filepath)
    is_dataset_balanced(y)

    trace, decision_boundary = build_promotion_model(x, y)
    plot_decision_boundary(trace, x, y)

    print(az.summary(trace, var_names=['alpha', 'beta']))


if __name__ == '__main__':
    solve('date_promovare_examen.csv')

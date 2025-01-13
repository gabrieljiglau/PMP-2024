import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def generate_data():
    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, -5]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    # plt.show()

    return mix

def iqr(x, a=0):
    return np.subtract(*np.percentile(x, [75, 25], axis=a))

def ex1():

    mixture_data = np.array(generate_data())

    clusters = [2, 3, 4]
    models = []
    i_datas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(mixture_data.min(), mixture_data.max(), cluster),
                              sigma=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)

            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mixture_data)
            i_data = pm.sample(100, target_accept=0.9, random_seed=123, return_inferencedata=True)
            i_datas.append(i_data)
            models.append(model)

    [pm.compute_log_likelihood(i_datas[i], model=models[i]) for i in range(3)]
    comp = az.compare(dict(zip([str(c) for c in clusters], i_datas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")

    print(comp)
    az.plot_compare(comp)
    plt.show()


if __name__ == '__main__':

    ex1()

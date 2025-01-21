import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pymc import math


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
            i_data = pm.sample(200, target_accept=0.9, random_seed=123, return_inferencedata=True)
            i_datas.append(i_data)
            models.append(model)

    [pm.compute_log_likelihood(i_datas[i], model=models[i]) for i in range(3)]

    comp = az.compare(dict(zip([str(cluster) for cluster in clusters], i_datas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")

    print('comparing with waic: ', comp)
    az.plot_compare(comp)
    plt.show()

    comp = az.compare(dict(zip([str(cluster) for cluster in clusters], i_datas)),
                      method='BB-pseudo-BMA', ic='loo', scale='deviance')
    print('comparing with loo: ', comp)
    az.plot_compare(comp)
    plt.show()


def import_data(file_id='1Svtgg3gHxIPitqkzzOfYPiMU_0PBtfJG', new_filename='cholesterol.csv'):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)

    if not os.path.exists(new_filename):
        # index = False, meaning the columns are not indexed
        # header = False, meaning the column names do not appear
        df.to_csv(new_filename, index=False, header=True)
        # print(f"Data from {url} saved successfully")

    return df


def ex2(groups_arr):
    data = import_data()
    if data is None:
        print('Data import failed')
        return

    models = []
    traces = []

    x1 = data['Ore_Exercitii'].values
    y = data['Colesterol'].values

    for group_index in groups_arr:
        with pm.Model() as model:
            # a = the concentration parameters of the Dirichlet distribution
            weights = pm.Dirichlet('weights', a=np.ones(group_index))

            alpha = pm.Normal('alpha', mu=0, sigma=10, shape=group_index)
            beta1 = pm.Normal('beta1', mu=0, sigma=10, shape=group_index)
            gamma = pm.Normal('gamma', mu=0, sigma=10, shape=group_index)

            # pm.Data -> share a variable without rebuilding the pymc model
            x_shared = pm.Data('x_shared', x1)

            mu = pm.Deterministic('mu', math.stack([alpha[k] + beta1[k] * x_shared + gamma[k] * x_shared**2
                                                    for k in range(group_index)]))
            mu_means = pm.Deterministic('mu_means', mu.mean(axis=0))
            sigma = pm.HalfNormal('sigma', sigma=10)

            y_pred = pm.NormalMixture('y_pred', w=weights, mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(1000, return_inferencedata=True)

            traces.append(trace)
            models.append(model)

    for trace in traces:
        print(az.summary(trace))
        az.plot_trace(trace, var_names=['mu_means', 'alpha', 'beta', 'gamma'])
        plt.show()

    [pm.compute_log_likelihood(traces[i], model=models[i]) for i in range(len(traces))]
    comp = az.compare(dict(zip([str(group) for group in groups_arr], traces)),
                      method='BB-pseudo-BMA', ic='waic', scale='deviance')

    az.plot_compare(comp)
    print(f"comparing with waic \n{comp}")
    plt.show()

    comp = az.compare(dict(zip([str(group) for group in groups_arr], traces)),
                      method='BB-pseudo-BMA', ic='loo', scale='deviance')

    az.plot_compare(comp)
    print(f"comparing with loo \n{comp}")
    plt.show()


if __name__ == '__main__':
    # ex1()
    groups = [3, 4, 5]
    ex2(groups)

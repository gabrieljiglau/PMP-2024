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


def build_second_model(number_of_subpopulations):
    data = import_data()
    if data is None:
        print('Data import failed')
        return

    x1 = data['Ore_Exercitii'].values
    y = data['Colesterol'].values

    with pm.Model() as model:
        # a = the concentration parameters of the Dirichlet distribution
        weights = pm.Dirichlet('weights', a=np.ones(number_of_subpopulations))

        alpha = pm.Normal('alpha', mu=0, sigma=10, shape=number_of_subpopulations)
        beta1 = pm.Normal('beta1', mu=0, sigma=10, shape=number_of_subpopulations)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=number_of_subpopulations)

        mu = alpha[:, None] + beta1[:, None] * x1 + gamma[:, None] * x1**2
        sigma = pm.HalfNormal('sigma', sigma=10, shape=number_of_subpopulations)

        y_pred = pm.Mixture('y_obs', w=weights, comp_dists=pm.Normal.dist(mu=mu.T, sigma=sigma), observed=y)
        trace = pm.sample(100, return_inferencedata=True)

    return model, trace

def plot_posterior(posterior_trace):
    print(az.summary(posterior_trace))
    az.plot_trace(trace, var_names=['alpha', 'beta1', 'gamma'])
    plt.show()

def compare_models(models_list, traces_list, groups_arr):

    [pm.compute_log_likelihood(traces_list[i], model=models_list[i]) for i in range(len(traces_list))]
    comp = az.compare(dict(zip([str(group) for group in groups_arr], traces_list)),
                      method='BB-pseudo-BMA', ic='waic', scale='deviance')

    az.plot_compare(comp)
    print(f"comparing with waic \n{comp}")
    plt.show()

    comp = az.compare(dict(zip([str(group) for group in groups_arr], traces_list)),
                      method='BB-pseudo-BMA', ic='loo', scale='deviance')

    az.plot_compare(comp)
    print(f"comparing with loo \n{comp}")
    plt.show()


if __name__ == '__main__':
    # ex1()

    models = []
    traces = []
    groups = [3, 4, 5]
    for group in groups:
        model, current_trace = build_second_model(group)
        models.append(model)
        traces.append(current_trace)

    for trace in traces:
        plot_posterior(trace)

    compare_models(models, traces, groups_arr=groups)

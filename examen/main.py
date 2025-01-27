import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def map_flower_name(species: str):
    if isinstance(species, (int, float)):
        return species

    if species == 'setosa':
        return 0
    elif species == 'versicolor':
        return 1
    elif species == 'virginica':
        return 2


def save_to_excel(data, file_name='cleaned_iris_dataset'):
    data.to_excel(file_name, index=False)


# age mapping max will be 12, so values aren't too sparse

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


def build_second_model(number_of_subpopulations, attribute_name):

    flower_data = pd.read_csv('iris.csv')
    flower_data['species'] = flower_data['species'].apply(map_flower_name)

    x1 = flower_data[attribute_name].values
    y = flower_data['species']

    with pm.Model() as model:
        # a = the concentration parameters of the Dirichlet distribution
        weights = pm.Dirichlet('weights', a=np.ones(number_of_subpopulations))

        alpha = pm.Normal('alpha', mu=0, sigma=10, shape=number_of_subpopulations)
        beta1 = pm.Normal('beta1', mu=0, sigma=10, shape=number_of_subpopulations)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=number_of_subpopulations)

        mu = alpha[:, None] + beta1[:, None] * x1 + gamma[:, None] * x1 ** 2
        sigma = pm.HalfNormal('sigma', sigma=10, shape=number_of_subpopulations)

        y_pred = pm.Mixture('y_obs', w=weights, comp_dists=pm.Normal.dist(mu=mu.T, sigma=sigma), observed=y)
        trace = pm.sample(100, return_inferencedata=True)

    return model, trace


def plot_posterior(posterior_trace):
    print(az.summary(posterior_trace))
    az.plot_trace(posterior_trace, var_names=['alpha', 'beta1', 'gamma'])
    plt.show()


if __name__ == '__main__':
    attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    models = []
    traces = []

    for attribute in attributes:
        model, current_trace = build_second_model(number_of_subpopulations=3, attribute_name=attribute)
        models.append(model)
        traces.append(current_trace)

    for trace in traces:
        plot_posterior(trace)

    """
    compare_models(models, traces, groups_arr=groups)
    """

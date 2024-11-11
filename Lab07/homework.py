import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

def ex1():

    observed_k = [10, 15, 20, 5, 25, 12, 18]
    observed_n = [100, 120, 150, 90, 200, 110, 130]

    with pm.Model() as first_model:
        # apriori
        p = pm.Beta('p', alpha=2, beta=2, shape=7)  # there are 7 days, bruh

        # likelihood / verosimilitatea
        k = pm.Binomial('k', n=observed_n, p=p, observed=observed_k)

        p_mean = pm.Deterministic('p_mean', pm.math.mean(p))

        trace = pm.sample(1000, return_inferencedata=True)

    az.plot_trace(trace, var_names=['p', 'p_mean'])  # sampling process for each parameter
    plt.show()

    az.plot_posterior(trace)  # summary of the posterior distribution
    plt.show()


def import_data():
    file_id = "1n4TU_x0jCdxpWHJktYLXndS-7USn2NQZ"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    df = pd.read_csv(url)
    df.to_csv('trafic.csv')  # to_csv returns None
    return df

def get_intervals(df):

    first_interval = df[0:180]  # 4 to 7
    second_interval = df[180:240]  # 7 to 8
    third_interval = df[240:720]  # 8 to 16
    fourth_interval = df[720:900]  # 16 to 19
    fifth_interval = df[900:1200]  # 19 to 24

    return list[first_interval, second_interval, third_interval, fourth_interval, fifth_interval]

def ex2():
    data = import_data()  # Assuming this returns the correct dataframe.
    intervals = get_intervals(data)  # Should return list of intervals as dicts or DataFrames.
    interval_lengths = [180, 60, 480, 180, 300]  # Duration of each interval in minutes.

    with pm.Model() as second_model:
        # Define Exponential priors for lambda in each interval
        lambda_1 = pm.Exponential('lambda_1', 1)  # 4 AM to 7 AM, low traffic
        lambda_2 = pm.Exponential('lambda_2', 1.5)  # 7 AM to 8 AM, traffic increase
        lambda_3 = pm.Exponential('lambda_3', 1)  # 8 AM to 4 PM, moderate traffic
        lambda_4 = pm.Exponential('lambda_4', 2)  # 4 PM to 7 PM, high traffic peak
        lambda_5 = pm.Exponential('lambda_5', 1)  # 7 PM to midnight, traffic decrease

        # Check if 'traffic' exists in intervals
        traffic_1 = pm.Poisson('traffic_1', mu=lambda_1, observed=intervals[0]['traffic'])
        traffic_2 = pm.Poisson('traffic_2', mu=lambda_2, observed=intervals[1]['traffic'])
        traffic_3 = pm.Poisson('traffic_3', mu=lambda_3, observed=intervals[2]['traffic'])
        traffic_4 = pm.Poisson('traffic_4', mu=lambda_4, observed=intervals[3]['traffic'])
        traffic_5 = pm.Poisson('traffic_5', mu=lambda_5, observed=intervals[4]['traffic'])

        # Compute weighted lambda for each interval (multiply by duration)
        weighted_lambda = (lambda_1 * interval_lengths[0] +
                           lambda_2 * interval_lengths[1] +
                           lambda_3 * interval_lengths[2] +
                           lambda_4 * interval_lengths[3] +
                           lambda_5 * interval_lengths[4])

        # Compute the mean lambda, normalizing by total interval length
        lambda_mean = pm.Deterministic(
            'lambda_mean', weighted_lambda / sum(interval_lengths)  # Normalize by total length
        )

        # Sampling from the model
        trace = pm.sample(1000, return_inferencedata=True)
    az.plot_trace(trace, var_names=['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'lambda_mean'])
    plt.show()

    az.plot_posterior(trace, var_names=['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5', 'lambda_mean'])
    plt.show()


if __name__ == '__main__':
    # ex1()
    ex2()





















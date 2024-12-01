import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def import_data(file_id="1gN9RrNoLWaDgqJT_pLYRug1SB9fKBWex"):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)
    df.to_csv('auto.csv')

    columns = [column for column in df.columns if column == 'mpg' or column == 'horsepower']
    df = df[columns]

    # removing non-integer or missing data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    df.to_csv('cleaned_auto.csv', header=None, index=False)
    return df


def plot_mpg_hp_correlation(df):
    slope, intercept, r_value, p_value, std_err = linregress(df['horsepower'], df['mpg'])

    x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
    y_values = intercept + slope * x_values

    plt.figure(figsize=(10, 6))
    plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, label='Observed Data')
    plt.plot(x_values, y_values, color='blue', label='Linear Regression Line')
    plt.title('Linear Regression')
    plt.xlabel('Horsepower (HP)')
    plt.ylabel('Miles Per Gallon (MPG)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Linear Regression Coefficients:")
    print("Intercept:", intercept)
    print("Slope:", slope)

def build_model(df):
    x = df['horsepower']
    y = df['mpg']

    y_stddev = y.std()

    with pm.Model() as mpg_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        epsilon = pm.HalfNormal('epsilon', sigma=10 * y_stddev)
        mu = pm.Deterministic('mu', alpha + beta * x)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    az.plot_trace(trace, var_names=['alpha', 'beta', 'epsilon'])
    plt.show()
    print(az.summary(trace))

    posterior = trace.posterior.stack(samples=("chain", "draw"))

    # calculate the mean regression line
    alpha_m = posterior['alpha'].mean().item()
    beta_m = posterior['beta'].mean().item()

    plt.scatter(x, y, color='blue', alpha=0.5, label='Observed data')

    x_line = np.linspace(x.min(), x.max(), 100)  # 100 evenly spaced values
    y_line = alpha_m + beta_m * x_line

    plt.plot(x_line, y_line, color='red', label=f"Regression line: y={alpha_m:.2f} + {beta_m:2f}x")
    for a, b in zip(posterior['alpha'].values.flatten()[:50], posterior['beta'].values.flatten()[:50]):
        plt.plot(x_line, a + b * x_line, color='gray', alpha=0.3)

    """
    # posterior hdi 0.95 plot
    ppc = pm.sample_posterior_predictive(trace, model=mpg_model)
    plt.plot(x, y, 'b.')
    plt.plot(x, alpha_m + beta_m * x, c='k',
             label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    az.plot_hdi(x, ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray')
    az.plot_hdi(x, ppc.posterior_predictive['y_pred'], color='gray')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.show()
    """

    plt.xlabel("Horsepower")
    plt.ylabel("Mpg")
    plt.legend()
    plt.title("Regression lines from posterior sample")
    plt.show()

def build_second_model():
    x_advertising = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                              6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])

    y_sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                        15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

    y_stddev = y_sales.std()

    with pm.Model() as sales_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)

        epsilon = pm.HalfNormal('epsilon', sigma=10 * y_stddev)
        mu = pm.Deterministic('mu', alpha + beta * epsilon)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_sales)

        trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True)

    az.plot_trace(trace, var_names=['alpha', 'beta', 'epsilon'])
    # plt.show()

    print('summary \n', az.summary(trace))
    print(az.summary(trace, var_names=['alpha']))

    # extracting only a certain 'characteristic' of alpha, let's say mean:
    print('alpha_mean = ', trace.posterior['alpha'].mean().item())

    # plotting the posterior
    """
    posterior = trace.posterior.stack(samples=("chain", "draw"))

    # calculate the mean regression line
    alpha_m = posterior['alpha'].mean().item()
    beta_m = posterior['beta'].mean().item()

    plt.scatter(x_advertising, y_sales, color='blue', alpha=0.5, label='Observed data')

    x_line = np.linspace(x_advertising.min(), x_advertising.max(), 100)  # 100 evenly spaced values
    y_line = alpha_m + beta_m * x_line

    plt.plot(x_line, y_line, color='red', label=f"Regression line: y={alpha_m:.2f} + {beta_m:2f}x")
    for a, b in zip(posterior['alpha'].values.flatten()[:50], posterior['beta'].values.flatten()[:50]):
        plt.plot(x_line, a + b * x_line, color='gray', alpha=0.3)

    plt.xlabel("Advertising")
    plt.ylabel("Sales")
    plt.legend()
    plt.title("Regression lines from posterior sample")
    plt.show()
    """

    return trace

def predict_sales(trace, x_advertising):
    posterior = trace.posterior.stack(samples=('chain', 'draw'))
    alpha_samples = posterior['alpha'].values.flatten()  # 1D array of all samples
    beta_samples = posterior['beta'].values.flatten()

    prediction_list = []
    for price in x_advertising:
        y_pred_samples = alpha_samples + price * beta_samples
        prediction_list.append(y_pred_samples)

    return np.array(prediction_list)


if __name__ == '__main__':
    """
    # 1_a
    data = import_data()
    # plot_mpg_hp_correlation(data)

    # 1_b, 1_c, 1_d
    build_model(data)
    """

    # 2_a, 2_b, 2_c
    x_new = np.array([8, 12, 15.84, 32.05, 64])
    trace_p = build_second_model()
    predictions = predict_sales(trace_p, x_new)

    for i, x in enumerate(x_new):
        pred_samples = predictions[i]
        pred_mean = pred_samples.mean()
        pred_hdi = az.hdi(pred_samples, hdi_prob=0.95)
        print(f"Advertising = {x:.2f}: Predicted sales = {pred_mean:.2f}, 95% HDI = {pred_hdi}")

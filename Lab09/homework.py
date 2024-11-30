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

    ppc = pm.sample_posterior_predictive(trace, model=mpg_model)
    plt.plot(x, y, 'b.')
    plt.plot(x, alpha_m + beta_m * x, c='k',
             label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    az.plot_hdi(x, ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray')
    az.plot_hdi(x, ppc.posterior_predictive['y_pred'], color='gray')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)

    plt.xlabel("Horsepower")
    plt.ylabel("Mpg")
    plt.legend()
    plt.title("Regression lines from posterior sample")
    plt.show()


if __name__ == '__main__':

    # 1_a
    data = import_data()
    # plot_mpg_hp_correlation(data)

    # 1_b, 1_c, 1_d
    build_model(data)

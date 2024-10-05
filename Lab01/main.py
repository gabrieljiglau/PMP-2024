import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Seed pentru reproducibilitate
np.random.seed(12)

def fun():

    # Parametrii reali
    true_mu = 170       # Înălțimea medie reală în cm
    true_sigma = 10     # Deviația standard reală în cm

    # Generăm datele observate (înălțimile studenților)
    # the actual obervations(this simulates real world data)
    observed_heights = np.random.normal(true_mu, true_sigma, size=100)

    plt.hist(observed_heights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribuția Înălțimilor Observate ale Studenților')
    plt.xlabel('Înălțime (cm)')
    plt.ylabel('Număr de Studenți')
    plt.show()

    with pm.Model() as model:
        # Prior pentru înălțimea medie (mu)
        # name, mean, standard deviation
        mu = pm.Normal('mu', mu=160, sigma=15)  # Presupunem inițial că media ar putea fi în jur de 160 cm

        # Prior pentru deviația standard (sigma)
        # sigma has a 'sigma', i.e. a standard deviation,
        # because the standard deviation itself is a probability distribution,
        # the second sigma(sigma=10), signifies our uncertainty or prior beliefs
        sigma = pm.HalfNormal('sigma', sigma=10)  # Deviația standard trebuie să fie pozitivă

        # Verosimilitatea datelor observate
        heights = pm.Normal('heights', mu=mu, sigma=sigma, observed=observed_heights)

        # Efectuăm eșantionarea MCMC
        print("Începem eșantionarea MCMC...")
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)
        print("Eșantionarea MCMC s-a încheiat.")


    # Rezumatul parametrilor posteriori
    print("\nRezumatul estimărilor:")
    summary = az.summary(trace, var_names=['mu', 'sigma'])
    print(summary)


if __name__ == '__main__':
    fun()

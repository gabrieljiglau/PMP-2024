import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from scipy import stats


def solve():
    observed_calls = 180
    hours = 10

    observed_rate = observed_calls / hours

    alpha_prior = 2
    beta_prior = 0.1

    alpha_posterior = alpha_prior + observed_calls
    beta_posterior = beta_prior + hours

    posterior = stats.gamma(a=alpha_posterior, scale=1/beta_posterior)

    x = np.linspace(0, 40, 500)
    plt.plot(x, posterior.pdf(x), label=f"Posterior Gamma({alpha_posterior}, {beta_posterior})")
    plt.xlabel("Call rate (λ)")
    plt.ylabel("Density")
    plt.title("Posterior Distribution of λ")
    plt.legend()

    hdi_limits = az.hdi(posterior.rvs(10000), hdi_prob=0.94)
    print(f"94% HDI: {hdi_limits}")
    plt.axvline(hdi_limits[0], color="red", linestyle="--", label="94% HDI")
    plt.axvline(hdi_limits[1], color="red", linestyle="--")
    plt.legend()
    plt.show()

"""
EXPLICATII:

---------------------
distributia gamma este aleasa ca 'prior' pentru lambda, 
pentru ca gamma este conjugata apriori pentru distributia poisson

asta inseamna ca daca verosimilitatea('likelihood') este Poisson si prior-ul este Gamma, 
rezultantul posterior va fi tot GAMMA (simplificand astfel calculele)

----------------------
folosing formula lui Bayes, ne modificam credintele asupra lui lambda folosind datele observate

distributia aposteriori a lui lambda este proportionala cu : P(λ∣X=k)∝P(X=k∣λ)⋅P(λ),
iar cand substituim P(X=k∣λ) din distributia Poisson si prior-ul P(λ) din distributia Gamma obtinem:

P(λ∣X=k)∝ (λ^k * e^(−λ))/k! * β^α/Γ(α)* λ^(α−1) * e^(−β*λ)

------------------------
cum Gamma prior si Poisson ca verosimilitate se combina si formeaza tot o distributie Gamma,
se pot actualiza α si β direct: 
Posterior α: α_posterior = α_prior + k (k = 180, numarul de apeluri observate)
Posterior β: β_posterior=β_prior + T (T = 10, intervalul de timp in care datele au fost observate)

Rezultatul este tot o distributie Gamma, care are insa parametrul λ actualizat, folosind informatiie observate

---------------------
"""

if __name__ == '__main__':
    solve()

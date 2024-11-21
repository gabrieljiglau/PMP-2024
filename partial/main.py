from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import stats
import matplotlib.pyplot as plt


def ex1(start_probs):
    # Dimensiunea gridului
    dimensiune_grid = (10, 10)

    # Lista de culori predefinite
    culori = [
        "red", "blue", "green", "yellow",
        "purple", "orange", "pink", "cyan",
        "brown", "lime"
    ]

    # Citirea gridului
    df = pd.read_csv('grid_culori.csv', header=None)
    grid_culori = df.to_numpy

    # Generarea secvenței de culori observate
    observatii = ['red', 'red', 'lime', 'yellow', 'blue']

    # Mapare culori -> indecși
    culoare_to_idx = {culoare: idx for idx, culoare in enumerate(culori)}
    idx_to_culoare = {idx: culoare for culoare, idx in culoare_to_idx.items()}

    # Transformăm secvența de observații în indecși
    observatii_idx = [culoare_to_idx[c] for c in observatii]

    # Definim stările ascunse ca fiind toate pozițiile din grid (100 de stări)
    numar_stari = dimensiune_grid[0] * dimensiune_grid[1]
    stari_ascunse = [(i, j) for i in range(dimensiune_grid[0]) for j in range(dimensiune_grid[1])]
    stare_to_idx = {stare: idx for idx, stare in enumerate(stari_ascunse)}
    idx_to_stare = {idx: stare for stare, idx in stare_to_idx.items()}

    # Matrice de tranziție
    transitions = np.zeros((numar_stari, numar_stari))
    for i, j in stari_ascunse:
        vecini = [
            (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)  # sus, jos, stânga, dreapta
        ]
        vecini_valizi = [stare_to_idx[(x, y)] for x, y in vecini if 0 <= x < 10 and 0 <= y < 10]
        ######

    # Matrice de emisie
    emissions = np.zeros((numar_stari, len(culori)))
    ######

    # Modelul HMM
    # stanga(40/10), sus(15/100), jos(15/100), dreapta(15/100), nimic(15/100)
    model = hmm.CategoricalHMM(n_components=numar_stari, n_iter=100, init_params="")
    model.startprob_ = start_probs
    model.transmat_ = None
    ######

    # Rulăm algoritmul Viterbi pentru secvența de observații
    ######

    # Convertim secvența de stări în poziții din grid
    secventa_stari = []
    drum = [idx_to_stare[idx] for idx in secventa_stari]

    # Vizualizăm drumul pe grid
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(dimensiune_grid[0]):
        for j in range(dimensiune_grid[1]):
            culoare = grid_culori[i, j]
            ax.add_patch(plt.Rectangle((j, dimensiune_grid[0] - i - 1), 1, 1, color=culoare))
            ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, culoare,
                    color="white", ha="center", va="center", fontsize=8, fontweight="bold")

    # Evidențiem drumul rezultat
    for idx, (i, j) in enumerate(drum):
        ax.add_patch(plt.Circle((j + 0.5, dimensiune_grid[0] - i - 0.5), 0.3, color="black", alpha=0.7))
        ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, str(idx + 1),
                color="white", ha="center", va="center", fontsize=10, fontweight="bold")

    # Setări axă
    ax.set_xlim(0, dimensiune_grid[1])
    ax.set_ylim(0, dimensiune_grid[0])
    ax.set_xticks(range(dimensiune_grid[1]))
    ax.set_yticks(range(dimensiune_grid[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(visible=True, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    plt.title("Drumul rezultat al stărilor ascunse", fontsize=14)
    plt.show()

def ex2_1(observed_data):
    # 1 for s, 0 for b
    with pm.Model() as model:
        theta = pm.Beta('theta', alpha=1, beta=1)
        y = pm.Bernoulli('y', p=theta, observed=observed_data)
        trace = pm.sample(1000, return_inferencedata=True)

    az.plot_trace(trace, var_names='theta')
    # plt.show()
    print(f"az.summary(trace): {az.summary(trace)}")
    return az.summary(trace)

def ex2_b(observed_data):

    with pm.Model() as model:
        theta = pm.Normal('theta', 0.666, 0.131)
        likelihood = pm.Bernoulli('y', p=theta, observed=observed_data)

        # Sample from the posterior of the second model
        trace = pm.sample(1000, return_inferencedata=True)

    az.plot_trace(trace)
    plt.show()


if __name__ == '__main__':
    # stanga(40/10), sus(15/100), jos(15/100), dreapta(15/100), nimic(15/100)
    # start_probabilities = np.array([40/100, 15/100, 15/100, 15/100, 15/100])
    # ex1(start_probabilities)

    data1 = [1, 1, 0, 0, 1, 1, 1, 1, 0, 1]  # ex2_1
    data2 = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]  # ex2_2a)
    # dist = ex2_1(data1)
    """ 
    # luam media si o vom folosi la nistributia bernoulli
    az.summary(trace):         mean     sd  hdi_3%  hdi_97%  ...  mcse_sd  ess_bulk  ess_tail  r_hat
    theta  0.666  0.131   0.424    0.904  ...    0.004     701.0    1006.0    1.0
    """

    ex2_b(data2)

    """
    Explicatii: 
    distributia de la 3.1 este mai informativa ca distributie apriori decat beta(1,1),
    iar valoarea lui theta este mai apropiata de datele noastre  
    """

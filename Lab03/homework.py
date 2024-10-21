import math

import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def ex1():
    # Defining the network structure
    model = BayesianNetwork([("C", "H"), ("P", "H")])

    # C = the contestant
    # H = the host
    # P = the prize

    # the contestant is choosing one of the 3 doors randomly
    cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.33], [0.33], [0.33]])

    cpd_p = TabularCPD("P", 3, [[0.33], [0.33], [0.33]])  # the prize has been put randomly behind one of the 3 doors
    cpd_h = TabularCPD(
        # Let’s say that the contestant selected door 0
        # and the host opened door 2, we need to find the probability of the prize i.e. P(P|H=2, C=0).
        "H",
        3,
        [  # When C = 0 and P = 0: The car is behind door 0, and the player picked door 0.
            # The host can pick either door 1 or door 2 (both goats), so he picks each with probability 0.5.
            [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
            [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
            [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
        ],
        evidence=["C", "P"],
        evidence_card=[3, 3],
    )

    # Associating the CPDs with the network structure.
    model.add_cpds(cpd_c, cpd_p, cpd_h)

    # Some other methods
    model.get_cpds()
    infer = VariableElimination(model)
    posterior_p = infer.query(["P"], evidence={"C": 0, "H": 2})
    print(posterior_p)

    """
    'Most people conclude that switching does not matter, because there would be a 50% chance of finding the car behind 
    either of the two unopened doors. 
    This would be true if the host selected a door to open at random, but this is not the case.
     
    By opening his door, Monty is saying to the contestant 'There are two doors you did not choose, and the probability 
    that the prize is behind one of them is 2/3. I'll help you by using my knowledge of where the prize is to open 
    one of those two doors to show you that it does not hide the prize. 
    You can now take advantage of this additional information. Your choice of door A has a chance of 1 in 3 of being
    the winner. 
    I have not changed that. But by eliminating door C, I have shown you that the probability that door B hides the 
    prize is 2 in 3
     '
     
     
    another, more intuitive perspective:
    'Savant suggests that the solution will be more intuitive with 1,000,000 doors rather than 3. In this case, 
    there are 999,999 doors with goats behind them and one door with a prize. After the player picks a door, 
    the host opens 999,998 of the remaining doors. On average, in 999,999 times out of 1,000,000, the remaining door
    will contain the prize. 
    Intuitively, the player should ask how likely it is that, given a million doors, they managed to pick the right one
    initially.'
     
    """


def build_model():
    disease_model = BayesianNetwork([('B', 'T'), ('B', 'X'), ('T', 'D'), ('B', 'D')])

    # T, D, B, X - valori binare

    # T = tuseste sau nu
    # D = are dificultate in respiratie
    # B = are boala
    # X = radiografie anormala

    # primul va fi ca nu are (peste tot)

    # how do I model P(T = 1 | B = 1) = 0.8 and P(T = 1| B = 0) ?
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.9], [0.1]])
    cpd_t = TabularCPD(variable='T', variable_card=2,
                       values=[[0.7, 0.2],
                               [0.3, 0.8]],
                       evidence=['B'],
                       # evidence_card = nr de valori pe care le poate lua variabila de care depinde
                       evidence_card=[2])

    cpd_x = TabularCPD(variable='X', variable_card=2,
                       values=[[0.9, 0.1],
                               [0.1, 0.9]],
                       evidence=['B'],
                       evidence_card=[2])

    cpd_d = TabularCPD(variable='D', variable_card=2,
                       values=[[0.9, 0.5, 0.6, 0.1],
                               [0.1, 0.5, 0.4, 0.9]],
                       evidence=['B', 'T'],
                       evidence_card=[2, 2])

    disease_model.add_cpds(cpd_b, cpd_t, cpd_x, cpd_d)
    disease_model.check_model()

    return disease_model


def ex2_b(model):
    infer = VariableElimination(model)
    posterior_p = infer.query(["B"], evidence={"T": 1, "D": 1})
    print(f"{posterior_p} \n")


def ex2_c(model):
    infer = VariableElimination(model)
    posterior_p = infer.query(["X"], evidence={"B":0, "T": 1, "D": 1})
    print(f"{posterior_p} \n")


# heads = 1, tails = 0
def throw_a_coin(p_heads, p_tails):
    return np.random.choice([1, 0], p=[p_heads, p_tails])


def throw_a_die():
    die = [1, 2, 3, 4, 5, 6]
    return np.random.choice(die)


def simulate_one_game():
    first_player = throw_a_coin(1 / 2, 1 / 2)

    number = throw_a_die()
    no_heads = 0
    if first_player == 0:
        for i in range(2 * number):
            # heads is 1
            if throw_a_coin(1 / 2, 1 / 2) == 1:
                no_heads += 1
    else:
        for i in range(2 * number):
            if throw_a_coin(4 / 7, 3 / 7) == 1:
                no_heads += 1

    return first_player, number, no_heads


"""
If first_player is 0, 1 - 0 will give 1.
If first_player is 1, 1 - 1 will give 0.
"""


def designate_winner(tuple_t):
    first_player, number, no_heads = tuple_t
    return first_player if number >= no_heads else 1 - first_player


def ex3a_simulate_game(no_iterations=10000):
    no_wins1 = 0
    no_wins2 = 0

    for i in range(no_iterations):
        t = simulate_one_game()
        winner = designate_winner(t)

        if winner == 1:
            no_wins1 += 1
        else:
            no_wins2 += 1

    print(f"Probability of the first player winning: {no_wins1 / no_iterations}")
    print(f"Probability of the second player winning:  {no_wins2 / no_iterations}")
    return no_wins1 / no_iterations, no_wins2 / no_iterations


def ex3b_build_second_model():
    # M = prima aruncare de moneda (normala)
    # Start = indexul jucatorului care incepe (0 sau 1)
    # N = numarul obtinut dupa aruncarea cu zarul
    # M_prim = number of heads

    game_model = BayesianNetwork([
        ('M', 'Start'),
        ('Start', 'N'),
        ('N', 'M_prim'),
        ('Start', 'M_prim')
    ])

    # (Initial coin toss, 0 = J0 starts, 1 = J1 starts)
    cpd_m = TabularCPD(variable='M', variable_card=2, values=[[0.5], [0.5]])

    # Start (which player starts based on M)
    cpd_start = TabularCPD(
        variable='Start', variable_card=2,
        # first column: the first player starts
        # second column: the second player starts
        values=[[1, 0],
                [0, 1]],
        evidence=['M'], evidence_card=[2]
    )

    # N (Result of the die roll, uniform probabilities for a fair die)
    cpd_n = TabularCPD(
        variable='N', variable_card=6,
        values=[
            [1 / 6, 1 / 6],  # first column: probability of getting 1, 2, 3, ... , 6 for J0
            [1 / 6, 1 / 6],  # second column: probability of getting 1, 2, 3, ... , 6 for J1
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6],
            [1 / 6, 1 / 6]
        ],
        evidence=['Start'], evidence_card=[2]
    )

    # M_prim (number of heads)
    # Since `M_prim` depends on `N` and `Start`, we need to compute the distribution for each case
    heads_probs_j0 = 0.5  # Fair coin for J0
    heads_probs_j1 = 4 / 7  # Biased coin for J1
    max_heads = 12  # Maximum heads we need for M_prim

    # Construct probability values based on biased/unbiased coin
    values = []
    # Loop over each possible die roll outcome, N (from 1 to 6)
    for n in range(1, 7):
        # Calculate probabilities for J0 (who uses a fair coin)
        p_heads_j0 = []
        for k in range(2 * n + 1):  # Possible heads outcomes for J0 from 0 up to 2*N
            probability_k_heads = math.comb(2 * n, k) * (heads_probs_j0 ** k) * ((1 - heads_probs_j0) ** (2 * n - k))
            p_heads_j0.append(probability_k_heads)

        # Calculate probabilities for J1 (who uses a biased coin)
        p_heads_j1 = []
        for k in range(2 * n + 1):  # Possible heads outcomes for J1 from 0 up to 2*N
            probability_k_heads = math.comb(2 * n, k) * (heads_probs_j1 ** k) * ((1 - heads_probs_j1) ** (2 * n - k))
            p_heads_j1.append(probability_k_heads)

        # Pad the probabilities to cover outcomes up to 12 heads
        # This is necessary because `2 * N` can vary, but we need uniform size
        p_heads_j0 += [0] * (max_heads - len(p_heads_j0) + 1)
        p_heads_j1 += [0] * (max_heads - len(p_heads_j1) + 1)

        # Add these distributions to the values list
        values.append(p_heads_j0)  # Append probabilities for J0
        values.append(p_heads_j1)

    # Reshape the list to match expected shape (variable_card, evidence_card[0] * evidence_card[1])
    values = np.array(values).T.tolist()

    cpd_mprim = TabularCPD(
        variable='M_prim', variable_card=max_heads + 1,
        values=values,
        evidence=['Start', 'N'],
        evidence_card=[2, 6])

    # Add CPDs to the model
    game_model.add_cpds(cpd_m, cpd_start, cpd_n, cpd_mprim)

    # Verify the model
    game_model.check_model()
    print("it god damn works!")

    return game_model


def ex3_c(model):
    infer = VariableElimination(model)
    posterior_p = infer.query(["Start"], evidence={'M_prim': 1})
    print(f"Posterior probabilities for who started the game given one head: \n{posterior_p}\n")


if __name__ == '__main__':
    # ex1()
    first_model = build_model()

    """
    print("P(B1) ca sufera")
    ex2_b(first_model)  # P(B=1 | T=1 and D=1) = 0.3478
    """

    # print("P(X1) ca are radiografia anormala")
    # ex2_c(first_model)  # P(X=1 | B = 0 and T = 1 and D=1)

    """
    Probability of the first player winning: 0.4249
    Probability of the second player winning:  0.5751
    """
    ex3a_simulate_game()

    game_model = ex3b_build_second_model()
    ex3_c(game_model)

    """
    Probability of Player1 starting: 0.9633
    Probability of Player2 starting: 0.0367
    """

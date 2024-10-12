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
        [   # When C = 0 and P = 0: The car is behind door 0, and the player picked door 0.
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

    # how do i model P(T = 1 | B = 1) = 0.8 and P(T = 1| B = 0) ?
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
    posterior_p = infer.query(["X"], evidence={"T": 1, "D": 1})
    print(f"{posterior_p} \n")


if __name__ == '__main__':
    # ex1()

    disease_model = build_model()

    print("P(B1) ca sufera")
    ex2_b(disease_model)  # P(B=1 | T=1 and D=1) = 0.3478

    print("P(X1) ca are radiografia anormala")
    ex2_c(disease_model)  # P(X=1 | T = 1 and D=1) = 0.3783

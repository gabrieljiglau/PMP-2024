import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation


def build_and_visualize(edge_list, lambda_arg):

    model = MarkovNetwork()
    model.add_edges_from(edge_list)

    G = nx.Graph()
    G.add_edges_from(edge_list)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue')
    plt.show()

    cliques = list(nx.find_cliques(G))
    print(f"Cliques in graph: {cliques}")

    # define the pairwise factors (based on the received lambda values)
    # lambda represents their interaction strength
    factor_A1_A2 = DiscreteFactor(variables=["A1", "A2"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[0]), np.exp(-lambda_arg[0]),
                                          np.exp(-lambda_arg[0]), np.exp(lambda_arg[0])])
    factor_A1_A3 = DiscreteFactor(variables=["A1", "A3"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[1]), np.exp(-lambda_arg[1]),
                                          np.exp(-lambda_arg[1]), np.exp(lambda_arg[1])])
    factor_A2_A4 = DiscreteFactor(variables=["A2", "A4"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[2]), np.exp(-lambda_arg[2]),
                                          np.exp(-lambda_arg[2]), np.exp(lambda_arg[2])])
    factor_A2_A5 = DiscreteFactor(variables=["A2", "A5"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[3]), np.exp(-lambda_arg[3]),
                                          np.exp(-lambda_arg[3]), np.exp(lambda_arg[3])])
    factor_A3_A4 = DiscreteFactor(variables=["A3", "A4"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[4]), np.exp(-lambda_arg[4]),
                                          np.exp(-lambda_arg[4]), np.exp(lambda_arg[4])])
    factor_A4_A5 = DiscreteFactor(variables=["A4", "A5"], cardinality=[2, 2],
                                  values=[np.exp(lambda_arg[5]), np.exp(-lambda_arg[5]),
                                          np.exp(-lambda_arg[5]), np.exp(lambda_arg[5])])

    # add factors to the model
    model.add_factors(factor_A1_A2, factor_A1_A3, factor_A2_A4, factor_A2_A5, factor_A3_A4, factor_A4_A5)

    # get all factors (all the potentials and probabilities)
    print('Factors: ')
    for factor in model.get_factors():
        print(factor)

    belief_propagation = BeliefPropagation(model)

    joint_prob = belief_propagation.map_query(variables=["A1", "A2", "A3", "A4", "A5"])

    print("Joint Probability (Most Likely States):", joint_prob)


if __name__ == '__main__':
    edges = [("A1", "A2"), ("A1", "A3"),
             ("A2", "A4"), ("A2", "A5"),
             ("A3", "A4"), ("A4", "A5")]

    # lambda values for each pair (A1-A2, A1-A3, A2-A4, A2-A5, A3-A4, A4-A5)
    lambda_param = [1.0, 0.5, 0.8, 0.6, 1.2, 0.7]
    build_and_visualize(edges, lambda_arg=lambda_param)

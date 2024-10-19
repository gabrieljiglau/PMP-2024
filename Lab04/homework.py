from typing import Dict, List
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation


def ex2(weather_stations_num: int, neighbours_memo: Dict[str, List[str]], initial_probabilities: List[float]):
    # Define states for the weather conditions
    states = ["Soare", "Ploaie", "Nori"]  # 0, 1, 2

    # Create a Markov Network
    model = MarkovNetwork()

    # Add nodes to the model
    for station in neighbours_memo.keys():
        model.add_node(station)

    # Add edges (connections) based on neighbours
    for station, neighbours in neighbours_memo.items():
        for neighbour in neighbours:
            model.add_edge(station, neighbour)

    # Define the potential matrix
    potential_matrix = [
        [0.7, 0.2, 0.1],  # From Soare
        [0.2, 0.6, 0.2],  # From Ploaie
        [0.1, 0.3, 0.6]  # From Nori
    ]

    # Ensure the length of initial probabilities matches the number of weather stations
    if len(initial_probabilities) < weather_stations_num:
        raise ValueError(f"Expected at least {weather_stations_num} initial probabilities.")

    # Add factors to the model
    for i, station in enumerate(neighbours_memo.keys()):
        # Define the initial probability for this station
        initial_factor = DiscreteFactor(
            variables=[station],
            cardinality=[3],  # 3 states
            values=[
                initial_probabilities[i],  # Probability for "Soare"
                (1 - initial_probabilities[i]) / 2,  # Probability for "Ploaie"
                (1 - initial_probabilities[i]) / 2  # Probability for "Nori"
            ]
        )
        model.add_factors(initial_factor)

        # Add factors for pairs of neighboring stations
        for neighbour in neighbours_memo[station]:
            factor = DiscreteFactor(
                variables=[station, neighbour],
                cardinality=[3, 3],
                values=[
                    potential_matrix[0][0], potential_matrix[0][1], potential_matrix[0][2],
                    potential_matrix[1][0], potential_matrix[1][1], potential_matrix[1][2],
                    potential_matrix[2][0], potential_matrix[2][1], potential_matrix[2][2]
                ]
            )
            model.add_factors(factor)

    # Perform inference
    bp_infer = BeliefPropagation(model)
    marginals = bp_infer.map_query(variables=list(neighbours_memo.keys()))

    # Print the inferred states
    print("Inferred weather states for the stations:")
    print(marginals)


# Example usage
if __name__ == "__main__":
    weather_stations_num = 7

    # Define neighbors for each weather station
    neighbours_memo = {
        'S1': ['S2', 'S3'],
        'S2': ['S1', 'S4'],
        'S3': ['S1', 'S5'],
        'S4': ['S2', 'S6'],
        'S5': ['S3', 'S7'],
        'S6': ['S4'],
        'S7': ['S5']
    }

    # Define initial probabilities (should be at least equal to the number of stations)
    initial_probabilities = [0.3, 0.5, 0.2] * weather_stations_num

    ex2(weather_stations_num, neighbours_memo, initial_probabilities)

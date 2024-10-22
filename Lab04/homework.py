import numpy as np
from typing import Dict, List
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

potential_matrix = [
    [0.7, 0.2, 0.1],  # Sunny to (Sunny, Rainy, Cloudy)
    [0.2, 0.6, 0.2],  # Rainy to (Sunny, Rainy, Cloudy)
    [0.1, 0.3, 0.6]  # Cloudy to (Sunny, Rainy, Cloudy)
]


def ex2(weather_stations_num: int, neighbours_memo: Dict[str, List[str]], initial_probabilities: List[List[float]]):
    model = MarkovNetwork()

    stations = [f'S{i + 1}' for i in range(weather_stations_num)]
    model.add_nodes_from(stations)

    # add relationship between neighbours
    for station, neighbors in neighbours_memo.items():
        for neighbor in neighbors:
            model.add_edge(station, neighbor)

    # factors for each edge using the potential matrix
    for station, neighbors in neighbours_memo.items():
        for neighbor in neighbors:
            factor = DiscreteFactor(
                variables=[station, neighbor],
                cardinality=[3, 3],  # 3 weather states (Sunny, Rainy, Cloudy)
                values=potential_matrix
            )
            model.add_factors(factor)

    # add initial, prior probabilities for each station
    for station, prob in zip(stations, initial_probabilities):
        initial_factor = DiscreteFactor(
            variables=[station],
            cardinality=[3],  # 3 weather states (Sunny, Rainy, Cloudy)
            values=prob
        )
        model.add_factors(initial_factor)

    model.check_model()

    # get all factors (all the potentials and probabilities)
    print('Factors: ')
    for factor in model.get_factors():
        print(factor)

    # inference
    bp_infer = BeliefPropagation(model)

    # MAP = maximum a posteriori probability;
    # final configuration after the iterative process
    marginals = bp_infer.map_query(variables=stations)

    print("MAP inference result (marginals):")
    print(marginals)


def add_noise_to_image(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    height, width = image.shape

    # the total number of pixels
    num_pixels = height * width
    num_noisy_pixels = int(num_pixels * noise_level)

    # random indices to modify
    indices_to_modify = np.random.choice(num_pixels, num_noisy_pixels, replace=False)

    # create a 2D index array from the flat indices
    row_indices = indices_to_modify // width
    col_indices = indices_to_modify % width

    # create noise
    noise = np.random.randint(-5, 6, size=num_noisy_pixels)  # noise in range [-5, 5]

    # add noise to the selected pixels
    for i in range(num_noisy_pixels):
        image[row_indices[i], col_indices[i]] += noise[i]

    # clip values to be in [0, 255]
    new_image = np.clip(image, 0, 255)

    return new_image


def create_markov_network(height: int, width: int, lambda_arg: float, image: np.ndarray) -> MarkovNetwork:
    model = MarkovNetwork()

    # node names for each pixel in the grid
    nodes = [f'X{i}_{j}' for i in range(height) for j in range(width)]
    model.add_nodes_from(nodes)

    # add connections
    for i in range(height):
        for j in range(width):
            current_node = f'X{i}_{j}'
            # north neighbour
            if i > 0:
                model.add_edge(current_node, f'X{i - 1}_{j}')
            # south neighbour
            if i < height - 1:
                model.add_edge(current_node, f'X{i + 1}_{j}')
            # west neighbour
            if j > 0:
                model.add_edge(current_node, f'X{i}_{j - 1}')
            # east neighbour
            if j < width - 1:
                model.add_edge(current_node, f'X{i}_{j + 1}')

    # factors for each node
    for i in range(height):
        for j in range(width):
            current_node = f'X{i}_{j}'
            observed_value = image[i, j]

            # data term factor from observed noisy image
            data_term_values = np.zeros(256)
            for k in range(256):
                data_term_values[k] = np.exp(-lambda_arg * (k - observed_value) ** 2)

            data_factor = DiscreteFactor(
                variables=[current_node],
                cardinality=[256],
                values=data_term_values
            )
            model.add_factors(data_factor)

            # smoothness as a quadratic penalty on pixel differences
            smoothness_penalty = np.zeros((256, 256))
            for p in range(256):
                for q in range(256):
                    smoothness_penalty[p, q] = np.exp(-(p - q) ** 2)

            if i > 0:  # north neighbor
                neighbor_node = f'X{i - 1}_{j}'
                smoothness_factor = DiscreteFactor(
                    variables=[current_node, neighbor_node],
                    cardinality=[256, 256],
                    values=smoothness_penalty
                )
                model.add_factors(smoothness_factor)

            if i < height - 1:  # south neighbor
                neighbor_node = f'X{i + 1}_{j}'
                smoothness_factor = DiscreteFactor(
                    variables=[current_node, neighbor_node],
                    cardinality=[256, 256],
                    values=smoothness_penalty
                )
                model.add_factors(smoothness_factor)

            if j > 0:  # west neighbor
                neighbor_node = f'X{i}_{j - 1}'
                smoothness_factor = DiscreteFactor(
                    variables=[current_node, neighbor_node],
                    cardinality=[256, 256],
                    values=smoothness_penalty
                )
                model.add_factors(smoothness_factor)

            if j < width - 1:  # east neighbor
                neighbor_node = f'X{i}_{j + 1}'
                smoothness_factor = DiscreteFactor(
                    variables=[current_node, neighbor_node],
                    cardinality=[256, 256],
                    values=smoothness_penalty
                )
                model.add_factors(smoothness_factor)

    model.check_model()
    return model


def estimate_original_image(model, image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape

    try:
        bp_infer = BeliefPropagation(model)
        variables = [f'X{i}_{j}' for i in range(height) for j in range(width)]
        marginals = bp_infer.map_query(variables)

        # extract estimated values for the original image
        estimated_image = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                estimated_image[i, j] = marginals[f'X{i}_{j}']

        print(f"Estimated image: {estimated_image}")
        return estimated_image

    except Exception as e:
        print(f"Error during inference: {e}")
        return None


if __name__ == "__main__":
    # ex2
    """
    num_stations = 3

    neighbours = {
        'S1': ['S2'],
        'S2': ['S1', 'S3'],
        'S3': ['S2']
    }

    # [P(Sunny), P(Rainy), P(Cloudy)]
    probabilities = [
        [0.5, 0.3, 0.2],  # Station S1
        [0.4, 0.55, 0.05],  # Station S2
        [0.2, 0.7, 0.1]  # Station S3
    ]

    ex2(num_stations, neighbours, probabilities)
    """

    # ex3

    np.random.seed(0)  # reproducibility

    original_image = np.array([
        0, 4, 8, 12, 16,
        10, 14, 18, 22, 26,
        20, 24, 28, 32, 36,
        30, 34, 38, 42, 46,
        40, 44, 48, 52, 56
    ]).reshape(5, 5)

    print(f'Original image: {original_image}')

    noisy_image = add_noise_to_image(original_image)
    print(f'Noisy image:\n{noisy_image}')

    lambda_value = 0.5
    # network = create_markov_network(5, 5, lambda_value, noisy_image)
    # returned_image = estimate_original_image(network, noisy_image)

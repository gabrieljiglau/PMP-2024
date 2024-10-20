import random

import numpy as np
from PIL import Image
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


def build_model(original_image):
    # Initialize the Markov Network
    mrf = MarkovNetwork()

    # Define the grid size
    grid_size = original_image.shape[0]

    # Create nodes for each pixel in the image
    nodes = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    mrf.add_nodes_from(nodes)

    # Add edges between neighboring pixels
    for i in range(grid_size):
        for j in range(grid_size):
            if i > 0:  # North neighbor
                mrf.add_edge((i, j), (i - 1, j))
            if i < grid_size - 1:  # South neighbor
                mrf.add_edge((i, j), (i + 1, j))
            if j > 0:  # West neighbor
                mrf.add_edge((i, j), (i, j - 1))
            if j < grid_size - 1:  # East neighbor
                mrf.add_edge((i, j), (i, j + 1))

    print(f"Is the model ok ? {mrf.check_model()}")
    return mrf


def create_factors(mrf, noisy_image, lambd=1):
    """
    Creates and adds factors to the Markov Random Field based on the noisy image.
    """
    grid_size = noisy_image.shape[0]

    # Step 2: Create factors for each pixel
    for i in range(grid_size):
        for j in range(grid_size):
            pixel_var = (i, j)

            # Create the factor for the observed noisy pixel
            factor_noisy = DiscreteFactor(variables=[pixel_var], cardinality=[256], values=np.zeros(256))
            # Set the observed value (noisy pixel)
            factor_noisy.values[noisy_image[i, j]] = lambd  # This keeps the observed noisy value
            # Add the noisy factor to the model
            mrf.add_factors(factor_noisy)

            # Define neighbors
            neighbors = []
            if i > 0:  # North neighbor
                neighbors.append((i - 1, j))
            if i < grid_size - 1:  # South neighbor
                neighbors.append((i + 1, j))
            if j > 0:  # West neighbor
                neighbors.append((i, j - 1))
            if j < grid_size - 1:  # East neighbor
                neighbors.append((i, j + 1))

            # Step 3: Create factors between the current pixel and its neighbors
            for neighbor in neighbors:
                vars_neighbor = [pixel_var, neighbor]  # Current pixel and neighbor

                # Initialize the factor for the current pixel and neighbor
                factor_neighbor = DiscreteFactor(variables=vars_neighbor, cardinality=[256, 256],
                                                 values=np.zeros(256 * 256))  # Start with zeros

                # Set values favoring similar neighboring pixels
                for k in range(256):
                    # Diagonal entry: penalty for same values
                    factor_neighbor.values[(k * 256 + k) % 256] = lambd  # Same values
                    for l in range(256):
                        if k != l:
                            factor_neighbor.values[(k * 256 + l) % 256] = 1.0  # Low value for dissimilar pairs

                # Add the neighbor factor to the model
                mrf.add_factors(factor_neighbor)

    # Optional: Check model for consistency
    print(f"is the model in the second check ok? {mrf.check_model()}")
# Example function to denoise the image
def denoise_image(original_image, noisy_image, lambd=1):
    # Build the Markov Network model
    mrf = build_model(original_image)

    # Create factors based on the noisy image
    create_factors(mrf, noisy_image, lambd)

    # Inference using Belief Propagation
    try:
        bp_infer = BeliefPropagation(mrf)
    except Exception as e:
        print(f"Error during Belief Propagation initialization: {e}")
        return None

    # Map query to get the denoised image
    denoised_image = np.zeros_like(noisy_image)
    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            pixel_var = (i, j)
            try:
                denoised_image[i, j] = bp_infer.map_query(variables=[pixel_var])[pixel_var]
            except Exception as e:
                print(f"Error during map query for pixel ({i}, {j}): {e}")
                denoised_image[i, j] = noisy_image[i, j]  # Fallback to noisy value

    return denoised_image

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

    np.random.seed(0)  # For reproducibility

    # Step 2: Create a noisy version of the image
    original_image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)

    # Create a noisy image by modifying about 10% of the pixels
    noisy_image = original_image.copy()
    num_noisy_pixels = int(0.1 * original_image.size)
    noisy_indices = np.random.choice(original_image.size, num_noisy_pixels, replace=False)
    for idx in noisy_indices:
        x, y = np.unravel_index(idx, original_image.shape)
        noisy_image[x, y] = np.random.randint(0, 256)

    # Denoise the image
    denoised_image = denoise_image(original_image, noisy_image, lambd=1)

    # Print images
    print("Original Image:\n", original_image)
    print("Noisy Image:\n", noisy_image)
    print("Denoised Image:\n", denoised_image)
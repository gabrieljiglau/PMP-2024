import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hmmlearn import hmm


def solve(p_states, p_observations, initial_probabilities):

    n_states = len(p_states)

    transition_probability = np.array([
        [0, 0.5, 0.5],      # Hard to Medium or Easy with equal probability
        [0.5, 0.25, 0.25],  # Medium to Hard, Medium, Easy
        [0.5, 0.25, 0.25]   # Easy to Hard, Medium, Easy
    ])

    emission_probability = np.array([
        [0.1, 0.2, 0.4, 0.3],  # Hard
        [0.15, 0.25, 0.5, 0.1],  # Medium
        [0.2, 0.3, 0.4, 0.1]    # Easy
    ])

    model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, init_params="")
    model.startprob_ = initial_probabilities
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # define the mapping of observation labels to indices
    obs_map = {obs: idx for idx, obs in enumerate(p_observations)}
    obs_sequence = [obs_map[obs] for obs in ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"]]
    obs_sequence = np.array(obs_sequence).reshape(-1, 1)

    # use Viterbi algorithm to find the most probable sequence of hidden states
    logprob, state_sequence = model.decode(obs_sequence, algorithm="viterbi")

    # map state_sequence back to state names
    state_sequence = [p_states[state] for state in state_sequence]

    print("Most likely sequence of difficulties:", state_sequence)
    print("Log Probability of the sequence:", logprob)

    # plotting the state sequence
    sns.set_style("darkgrid")
    plt.plot(state_sequence, '-o', label="Difficulty Level")
    plt.xlabel("Test")
    plt.ylabel("Difficulty Level")
    plt.title("Most Likely Sequence of Test Difficulties")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # parameters
    states = ["Hard", "Medium", "Easy"]
    observations = ["FB", "B", "S", "NS"]
    start_probability = np.array([1/3, 1/3, 1/3])

    solve(states, observations, start_probability)

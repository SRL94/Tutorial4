import numpy as np

def hmm_probability(observations, states, start_probability, transition_probability, emission_probability):
    """
    Calculate the probability of observations given HMM parameters.
    
    Args:
    - observations: List of observed states or observations.
    - states: List of possible states.
    - start_probability: Initial state probabilities (dictionary).
    - transition_probability: Transition probabilities (nested dictionary).
    - emission_probability: Emission probabilities (nested dictionary).
    
    Returns:
    - prob: Probability of observing the sequence of observations.
    """
    num_states = len(states)
    num_observations = len(observations)
    
    # Initialize the forward probabilities matrix
    forward_prob = np.zeros((num_states, num_observations))

    # TO DO
    return prob

def viterbi(observations, states, start_probability, transition_probability, emission_probability):
    """
    Find the optimal sequence of hidden states using the Viterbi algorithm.
    
    Args:
    - observations: List of observed states or observations.
    - states: List of possible states.
    - start_probability: Initial state probabilities (dictionary).
    - transition_probability: Transition probabilities (nested dictionary).
    - emission_probability: Emission probabilities (nested dictionary).
    
    Returns:
    - best_path: Optimal sequence of hidden states.
    """
    # TO DO
    return best_path_states




states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}

observations_sequence = ['normal', 'cold', 'dizzy']

# E1: Given a set of observations X, calculate the occurrence probability of the observations X.
probability_X = hmm_probability(observations_sequence, states, start_probability, transition_probability, emission_probability)
print(f"Probability of observing the sequence: {probability_X}")

# E2: find the optimal sequence of hidden states
optimal_states = viterbi(observations_sequence, states, start_probability, transition_probability, emission_probability)
print(f"Optimal sequence of hidden states: {optimal_states}")





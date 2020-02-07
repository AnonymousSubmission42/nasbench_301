"""
Regularized evolution as described in:
Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
Regularized Evolution for Image Classifier Architecture Search.
In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

The code is based one the original regularized evolution open-source implementation:
https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

NOTE: This script has certain deviations from the original code owing to the search space of the benchmarks used:
1) The fitness function is not accuracy but error and hence the negative error is being maximized.
2) The architecture is a ConfigSpace object that defines the model architecture parameters.

Code adapted from NASBench-101
"""

import collections
import copy
import random
import re

import ConfigSpace
import numpy as np
import seaborn as sns

from nas_benchmark.benchmarks.nasbench_301 import NASBench301_arch_only

sns.set_style('whitegrid')


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(b, config):
    y = b.objective_function(config.get_dictionary())
    # returns negative error (similar to maximizing accuracy)
    return y


def random_architecture(cs):
    config = cs.sample_configuration()
    return config


def mutate_arch(cs, parent_config_instance):
    # Select which cell type to mutate
    cell_type = np.random.choice(['normal', 'reduce'])
    # Choose one of the three mutations
    mutation = np.random.choice(['identity', 'hidden_state_mutation', 'op_mutation'])

    if mutation == 'identity':
        return copy.deepcopy(parent_config_instance)
    elif mutation == 'hidden_state_mutation':
        # Create child architecture with modified link
        child_arch_dict = copy.deepcopy(parent_config_instance.get_dictionary())
        # Get all active hyperparameters which are related to the adjacency matrix of the cells
        hidden_states = list(
            filter(re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                   cs.get_active_hyperparameters(parent_config_instance)))

        # Select one hidden state to modify
        selected_hidden_state = cs.get_hyperparameter(str(np.random.choice(hidden_states)))

        # Choose the parent to change.
        current_parents = [int(parent) for parent in child_arch_dict[selected_hidden_state.name].split('_')]
        removed_parent = np.random.choice(current_parents)
        current_parents.remove(removed_parent)
        [remaining_parent] = current_parents

        # Determine the active intermediate nodes in the cell
        active_intermediate_nodes = []
        for state in hidden_states:
            active_intermediate_nodes.extend([int(intermediate_node) for intermediate_node in
                                              parent_config_instance[cs.get_hyperparameter(state).name].split('_')])

        # Which parent combinations contain the parent_to_stay and which operation edge is affected?
        node_parent_to_edge_num = lambda parent, node: parent + sum(np.arange(2, node + 1)) - node

        # Remove the previous edge
        selected_node = int(selected_hidden_state.name[-1])
        deleted_edge = node_parent_to_edge_num(removed_parent, selected_node)
        op_for_new_edge = child_arch_dict['NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, deleted_edge)]
        child_arch_dict = \
            removekey(child_arch_dict, 'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, deleted_edge))

        # Select the new parent from active intermediate nodes
        possible_parents = [i for i in np.sort(np.unique(active_intermediate_nodes)) if i < selected_node]
        # Remove current parent
        possible_parents.remove(remaining_parent)
        new_parent = np.random.choice(possible_parents)
        new_parents = '_'.join([str(elem) for elem in np.sort([new_parent, remaining_parent])])

        # Add new edge
        new_edge = node_parent_to_edge_num(new_parent, selected_node)
        child_arch_dict['NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, new_edge)] = op_for_new_edge

        # Add new parents
        child_arch_dict[selected_hidden_state.name] = new_parents

        child_config_instance = ConfigSpace.Configuration(cs, values=child_arch_dict)
        return child_config_instance
    else:
        # op mutation
        # Get all active hyperparameters which are related to the operations chosen in the cell
        hidden_state = list(
            filter(re.compile('.*edge_{}*.'.format(cell_type)).match,
                   cs.get_active_hyperparameters(parent_config_instance)))
        selected_hidden_state = cs.get_hyperparameter(str(np.random.choice(hidden_state)))
        choices = list(selected_hidden_state.choices)

        # Drop current value from the list of choices
        choices.remove(parent_config_instance[selected_hidden_state.name])

        # Create child architecture with modified link
        child_arch_dict = copy.deepcopy(parent_config_instance.get_dictionary())

        # Modify the selected link
        child_arch_dict[selected_hidden_state.name] = str(np.random.choice(choices))
        child_config_instance = ConfigSpace.Configuration(cs, values=child_arch_dict)
        return child_config_instance


def random_search(benchmark, configuration_space, cycles):
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(history) < cycles:
        model = Model()
        model.arch = random_architecture(configuration_space)
        model.accuracy = train_and_eval(benchmark, model.arch)
        population.append(model)
        history.append(model)

    return history


def regularized_evolution(benchmark, configuration_space, cycles, population_size, sample_size):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture(configuration_space)
        model.accuracy = train_and_eval(benchmark, model.arch)
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(configuration_space, parent.arch)
        child.accuracy = train_and_eval(benchmark, child.arch)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()

    return history


def regularized_evolution_frontend(surrogate_model_dir, runtime_model_dir, n_repetitions, n_iters, pop_size=50,
                                   sample_size=10):
    results_per_fidelity = {}
    for fidelity in range(7):
        results_per_random_seed = []
        for seed in range(n_repetitions):
            print("##### Seed {} #####".format(seed))
            np.random.seed(seed)

            b = NASBench301_arch_only(surrogate_model_dir=surrogate_model_dir, runtime_model_dir=runtime_model_dir,
                                      fidelity=fidelity)
            cs = b.config_space
            history = regularized_evolution(benchmark=b, configuration_space=cs, cycles=n_iters,
                                            population_size=pop_size, sample_size=sample_size)

            res = b.get_results(ignore_invalid_configs=True)
            results_per_random_seed.append(res)

        results_per_fidelity[fidelity] = results_per_random_seed
    return results_per_fidelity


def random_search_frontend(surrogate_model_dir, runtime_model_dir, n_repetitions, n_iters):
    results_per_fidelity = {}
    for fidelity in range(7):
        results_per_random_seed = []
        for seed in range(n_repetitions):
            print("##### Seed {} #####".format(seed))
            np.random.seed(seed)

            b = NASBench301_arch_only(surrogate_model_dir=surrogate_model_dir, runtime_model_dir=runtime_model_dir,
                                      fidelity=fidelity)
            cs = b.config_space
            history = random_search(benchmark=b, configuration_space=cs, cycles=n_iters)

            res = b.get_results(ignore_invalid_configs=True)
            results_per_random_seed.append(res)

        results_per_fidelity[fidelity] = results_per_random_seed
    return results_per_fidelity

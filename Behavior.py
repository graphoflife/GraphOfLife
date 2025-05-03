import numpy as np
from SimOptions import *
import copy

class Behavior:
    """
    Neural network with a rectified linear activation function that belongs to a particle.
    """

    def __init__(self, sim_options: NewSimOptions, indexer_f, data, parent_behavior=None):

        self.sim_data = data
        self.id = data.get_new_behavior_index()

        self.index = indexer_f()
        self.parent_index = -1
        self.parent_of_parent = -1
        self.parent_of_parent_of_parent = -1
        self.weights = []
        self.biases = []

        if parent_behavior is None:
            # Random weights and biases
            self.rgb = np.random.uniform(0.0, 1.0, (3,))

            sizes = [sim_options.INPUT_AMOUNT] + sim_options.LAYER_SIZES + [sim_options.OUTPUT_AMOUNT]
            for i in range(len(sizes) - 1):
                columns = int(sizes[i])
                rows = int(sizes[i + 1])
                self.weights.append(np.random.normal(0.0, 0.01, (rows, columns)))
                self.biases.append(np.random.normal(0.0, 0.01, (rows, 1)))

            self.meta_options = sim_options.create_meta_dictionary_for_particle()
        else:
            self.parent_index = parent_behavior.index
            self.parent_of_parent = parent_behavior.parent_index
            self.parent_of_parent_of_parent = parent_behavior.parent_of_parent

            self.rgb = parent_behavior.rgb.copy()
            # Inheritance from parent behavior
            for i in range(len(parent_behavior.weights)):
                self.weights.append(parent_behavior.weights[i].copy())
                self.biases.append(parent_behavior.biases[i].copy())
            self.meta_options = copy.deepcopy(parent_behavior.meta_options)

        entry = [self.index,
                 self.parent_index,
                 self.parent_of_parent,
                 self.parent_of_parent_of_parent,
                 self.rgb.copy()]
        data.next_birth_layer.append(entry)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_output(self, input_array: np.array, sim_options: NewSimOptions):
        """
        Feeds input vector through neural network and gets the output.
        Uses rectified linear activation function.

        :param input_array: must have shape input_amount x 1
        :return: output_array with shape output_amount x 1
        """
        output = input_array

        activation_function = sim_options.get(SimOptionsEnum.ACTIVATION_FUNCTION)
        if activation_function == "RELU":
            for i in range(len(self.weights)):
                output = np.maximum(np.matmul(self.weights[i], output) + self.biases[i], 0)
        elif activation_function == "SIGMOID":
            for i in range(len(self.weights)):
                output = self.sigmoid(np.matmul(self.weights[i], output) + self.biases[i])
            output = np.maximum(output, 0)
        else:   # TANH
            for i in range(len(self.weights)):
                output = np.tanh(np.matmul(self.weights[i], output) + self.biases[i])
            output = np.maximum(output, 0)
        return output

    def mutate(self, sim_options, vis_options):
        """
        Mutate Values of weights and biases with probability MUTATION_CHANCE
        :param sim_options: Options of Simulation
        """
        THRESHOLD = 100
        for i in range(len(self.weights)):
            self.weights[i] += np.random.normal(0, 0.01, self.weights[i].shape) \
                * (np.random.uniform(0, 1, self.weights[i].shape) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE))
            self.biases[i] += np.random.normal(0, 0.01, self.biases[i].shape) \
                * (np.random.uniform(0, 1, self.biases[i].shape) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE))
            self.weights[i] = np.maximum(np.minimum(self.weights[i], THRESHOLD), -THRESHOLD)
            self.biases[i] = np.maximum(np.minimum(self.biases[i], THRESHOLD), -THRESHOLD)

        for key in self.meta_options:
            if np.random.uniform(0.0, 1.0) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE_META):
                self.meta_options[key] = sim_options.all_options[key].get_random_option()

        self.rgb += np.random.normal(0.0, vis_options.mutate_rgb, (3, ))
        self.rgb = np.maximum(np.minimum(self.rgb, 1.0), 0.0)

# This module defines a perceptron class for use with the MNIST dataset.

import random
from vectorops import *

class Perceptron:
    """
    A perceptron that can process inputs and be modified.
    """

    def __init__(self, v_len: int, leaky_relu: bool):
        """

        Parameters
        ----------
        v_len: The number of inputs to this Perceptron.
        """
        self.weights = [random.randrange(-5,6) for i in range(v_len)]
        self.bias = random.randrange(-5,6)
        self._is_leaky_relu = leaky_relu

    def evaluate(self, inputs: [float]) -> float:
        """

        Parameters
        ----------
        inputs: A list of inputs to the perceptron.

        Returns
        -------
        The activation value of the perceptron.
        """
        return max(0.0, dot(self.weights, inputs) + self.bias)

    
    def change_weights(self, dw: [float]):
        self.weights = add(self.weights, dw)

    def propagate(self, da : [float]):
        '''Uses information about the desired changes in a, the
        activation values, to determine how to modify weights and bias.'''
        # Move elsewhere?

    def _show_values(self):
        '''Test function to print all weights and the bias.'''
        for i in range(len(self.weights)):
            print("weight", i, "is", self.weights[i])
        print("bias is", self.bias)

class InputPerceptron:
    pass

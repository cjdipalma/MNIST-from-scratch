# This module defines a perceptron class for use with the MNIST dataset.

import random
import vectorops
import numpy


class Perceptron:
    """
    A perceptron that can process inputs and be modified.
    """

    def __init__(self, v_len: int, activation_function):
        """

        Parameters
        ----------
        v_len: The number of inputs to this Perceptron.
        leakage_coeff: A value greater than or equal to zero representing
        how the activation should be modified below zero.
        """
        self.weights = [random.gauss(mu=0, sigma=1) for _ in range(v_len)]
        self.bias = random.gauss(mu=0, sigma=1)
        self._activation = activation_function
        self.z = None

    def evaluate(self, inputs: [float]) -> float:
        """

        Parameters
        ----------
        inputs: A list of inputs to the perceptron.

        Returns
        -------
        The activation value of the perceptron.
        """
        self.z = numpy.dot(self.weights, inputs) + self.bias
        return self._activation(self.z)

    def change_weights_and_bias(self, dw: [float], db: float):
        """

        Parameters
        ----------
        dw: Changes to each weight.
        db: Change to the bias.

        """
        self.weights = numpy.add(self.weights, dw)
        self.bias += db

    def get_activation(self):
        return self._activation

    def show_values(self, prepend: str = ''):
        """
        Test function to print all weights and the bias.
        """
        for i in range(len(self.weights)):
            print(prepend + "weight", i, "is", self.weights[i])
        print(prepend + "bias is", self.bias)

    def size_w(self):
        # may remove this function if self.weights is left public
        return len(self.weights)


class InputPerceptron:
    pass

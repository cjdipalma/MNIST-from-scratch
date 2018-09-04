"""
This file has both training and testing for the network.
"""

from perceptron import Perceptron
from mnist import MNIST
import random
import activation

num_inputs = 784

hiddenlayer1_size = 40
hiddenlayer2_size = 40

num_outputs = 10

batch_size = 100


class Network:

    def __init__(self, is_leaky_relu: bool = False, leakage_coeff: float = 0.2):
        self._relu = activation.ReLU(is_leaky=is_leaky_relu, leakage=leakage_coeff if is_leaky_relu else 0)
        self._sigmoid = activation.Sigmoid()
        self._identity = activation.Identity()

        self._network = [
                            [Perceptron(num_inputs, self._relu) for _ in range(hiddenlayer1_size)],
                            [Perceptron(hiddenlayer1_size, self._relu) for _ in range(hiddenlayer2_size)],
                            [Perceptron(hiddenlayer2_size, self._sigmoid) for _ in range(num_outputs)]
                        ]
        self._mn_data = MNIST('./images')
        self._desired_changes = None
        self._layer_inputs = None

    def feed_forward(self, inputs: [float]) -> [float]:
        """

        Parameters
        ----------
        inputs: a list of length 784 representing the activation levels of each pixel.

        Returns
        -------
        A list of length 10 with activation values corresponding to digits.

        """
        self._layer_inputs = []
        result_after_layer = inputs
        for layer in self._network:
            # print(result_after_layer)
            self._layer_inputs.append(result_after_layer)
            result_after_layer = [perceptron.evaluate(result_after_layer) for perceptron in layer]
        self._layer_inputs.append(result_after_layer)
        return result_after_layer

    def train(self, num_epochs: int):
        """
        Uses stochastic gradient descent to train this network.
        """
        learn_rate = 0.02

        images, labels = self._mn_data.load_training()
        indices = [i for i in range(len(images))]

        for epoch in range(num_epochs):
            random.shuffle(indices)  # Avoids modifying the actual lists
            epoch_cost = 0
            i = 0

            # Go through the training data in batches
            while i < len(indices):
                print(i, "---------------------------------------------------------")

                if i >= 800:
                    break

                start = i
                end = i + batch_size
                batch_indices = indices[start:end]

                dw = [[[0 for _ in range(perceptron.size_w())] for perceptron in layer] for layer in self._network]
                db = [[0 for _ in layer] for layer in self._network]

                # Take a single image from the batch
                for index in batch_indices:
                    # print("ex")
                    result = self.feed_forward(images[index])
                    epoch_cost += self.cost(result, labels[index])  # Creates self._desired_changes

                    # Backpropagate starting from the last (output) layer
                    for j in range(len(self._network)-1, -1, -1):
                        layer = self._network[j]
                        prev_act_values = self._layer_inputs[j]
                        function_name = layer[0].get_activation().name()

                        if j > 0:
                            next_desired_changes = [0.0 for _ in self._network[j-1]]
                        else:
                            next_desired_changes = None

                        if function_name == "relu":
                            leakage = self._relu.get_leakage()

                        # Look at each perceptron
                        for k in range(len(layer)):
                            perceptron = layer[k]
                            dc_da = self._desired_changes[k]

                            if function_name == "sigmoid":
                                dc_da *= self._sigmoid(perceptron.z) * (1 - self._sigmoid(perceptron.z))
                                # print(perceptron.z, sig_delta)
                                # print(dc_da)
                                db[j][k] -= dc_da * learn_rate

                                # For each weight
                                for l in range(len(perceptron.weights)):
                                    dw[j][k][l] -= dc_da * prev_act_values[l] * learn_rate

                                    if next_desired_changes:
                                        next_desired_changes[l] += dc_da * perceptron.weights[l]

                            elif function_name == "relu":
                                dc_da *= leakage if perceptron.z < 0 else 1
                                db[j][k] -= dc_da * learn_rate

                                # For each weight
                                for l in range(len(perceptron.weights)):
                                    dw[j][k][l] -= dc_da * prev_act_values[l] * learn_rate

                                    if next_desired_changes:
                                        next_desired_changes[l] += dc_da * perceptron.weights[l]

                            # print("dcda", dc_da)

                        if next_desired_changes:
                            # print("nd", next_desired_changes)
                            self._desired_changes = next_desired_changes

                    # End of sample image loop
                    # print(dw[1:])
                    # break

                # Update weights and biases
                for j in range(len(self._network)):
                    layer = self._network[j]

                    for k in range(len(layer)):
                        perceptron = layer[k]

                        perceptron.change_weights_and_bias(dw[j][k], db[j][k])

                # print(dw[1:])
                # print(db)

                i += batch_size

            print("Epoch {} completed out of {} with loss {}".format(epoch + 1, num_epochs, epoch_cost))

    def test(self, num_to_test) -> float:
        test_images, test_labels = self._mn_data.load_testing()
        total = min(num_to_test, len(test_images))
        num_correct = 0
        for i in range(min(num_to_test, len(test_images))):
            if i%100 == 0:
                print(i)
            l = self.feed_forward(test_images[i])
            if Network.one_hot_to_digit(l) == test_labels[i]:
                num_correct += 1

        return num_correct / total

    def cost(self, result: [float], label: int) -> float:
        """
        A simple cost function.
        """
        desired_outputs = Network.digit_to_one_hot(label)
        self._desired_changes = [result[i] - desired_outputs[i] for i in range(num_outputs)]
        return sum((result[i] - desired_outputs[i]) ** 2 for i in range(num_outputs))

    def show_network(self):
        if not self._relu.is_leaky():
            print("Network uses a ReLU activation function")
        else:
            print("Network uses a Leaky ReLU activation function with coefficient {}".format(self._relu.get_leakage()))
        print()
        for i in range(len(self._network)):
            print("Layer {}\n".format(i+1))
            for perceptron in self._network[i]:
                perceptron.show_values("\t")
            print()
        print("End display\n")

    @staticmethod
    def digit_to_one_hot(v: int):
        if v < 0 or v > 9:
            raise ValueError("Argument must be a single digit")
        l = [(1 if i == v else 0) for i in range(10)]

        return l

    @staticmethod
    def one_hot_to_digit(l: [float]):
        if len(l) == 0:
            raise ValueError("List is empty")
        max_index = 0
        max_value = l[0]
        for i in range(len(l)):
            if l[i] > max_value:
                max_value = l[i]
                max_index = i

        return max_index

    @staticmethod
    def normalize_input(inputs: [float]) -> [float]:
        """
        Linearly compresses the list "inputs" into the range [0, 1].

        Parameters
        ----------
        inputs: A list of data

        Returns
        -------
        The list, normalized between 0 and 1.
        """

    @staticmethod
    def MNIST_normalize_input(inputs: [int]) -> [float]:
        """
        A simplified version of the above function for the MNIST dataset.

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        return [i/256 for i in inputs]  # simple for MNIST

# print(_cost([0, 500, 0, 0, 0, 0, 0, 1000, 0, 0], 6))




            

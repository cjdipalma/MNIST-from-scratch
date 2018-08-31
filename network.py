'''This file has both training and testing for the network.'''

from perceptron import Perceptron
from perceptron import InputPerceptron
from mnist import MNIST
import random

num_inputs = 784

hiddenlayer1_size = 100
hiddenlayer2_size = 100

outputs = 10
batch_size = 50
desired_activation = 1000

class Network:

    def __init__(self):
        self._network = [
                            [Perceptron(num_inputs) for _ in range(hiddenlayer1_size)],
                            [Perceptron(hiddenlayer1_size) for _ in range(hiddenlayer2_size)],
                            [Perceptron(hiddenlayer2_size) for _ in range(outputs)]
                        ]


    def feed_forward(self, inputs: [float]) -> [float]:
        """

        Parameters
        ----------
        inputs: a list of length 784 representing the activation levels of each pixel.

        Returns
        -------
        A list of length 10 with activation values corresponding to digits.
        """


    def train(self):
        mn_data = MNIST()
        images, labels = mn_data.load_training()

        # Arbitrary large number representing desired confidence


    def train(self):
        """
        Uses stochastic gradient descent to train this network.
        """
        indices = [i for i in range(len(images))]
        random.shuffle(indices)
        index = 0
        for batch_num in range(int(len(images) / batch_size)):  # Batch number
            dw = [0 for i in range(outputs)]  # Used to update output neurons
            for batch_index in range(batch_size):  # Index in current batch
                for p in network[0]:
                    p.evaluate(images[index])

                index += 1

#network[1][0]._show_values()
#print(mn_data.display(images[0]))
#print(images[0])
#print(labels[0])
#print(len(images))
def _cost(self, result: [float], desired: int) -> float:
    """
    A simple cost function
    """
    d_a = tuple((0 if i != desired else desired_activation) for i in range(outputs))
    return sum((d_a[i] - result[i]) ** 2 for i in range(outputs)) / 2


print(_cost([0, 500, 0, 0, 0, 0, 0, 1000, 0, 0], 6))




            

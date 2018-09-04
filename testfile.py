from perceptron import Perceptron
from network import Network
import activation
from mnist import MNIST
images, labels = None, None


def test_perceptron():
    p = Perceptron(5, activation.ReLU(is_leaky=True))
    p.show_values()
    print(p.evaluate([1, 2, 3, 5, 7]))


def test_one_hot_digit_conversions():
    print(Network.one_hot_to_digit([1, 3, 123, -4, 1, 3, -3, 768, 7, 5555]))
    for i in range(9, -1, -1):
        print(Network.digit_to_one_hot(i))


def load_mnist_data():
    global images, labels
    images, labels = MNIST('./images').load_training()


def test_network(index: int = 0, show_network: bool = False):
    # inputs = Network.MNIST_normalize_input(images[index])
    nt = Network(True)
    if show_network:
        nt.show_network()
    # result = nt.feed_forward(inputs)
    # print(result)
    # print(nt.cost(result, labels[index]))
    print(nt.test(700))
    nt.train(5)
    print(nt.test(700))
    # nt.show_network()


def test_sigmoid():
    s = activation.Sigmoid()
    print(s(0))
    print(s(-3))
    print(s(3))
    print(s(500))
    print(s(-1000))
    print(s(-999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999))


if __name__ == '__main__':
    test_network(show_network=False)

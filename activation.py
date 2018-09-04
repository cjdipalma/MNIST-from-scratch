"""
Contains activation functions.
"""

import math


class ActivationFunction:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("call not implemented")

    def name(self):
        raise NotImplementedError("name function not implemented")


class ReLU(ActivationFunction):

    def __init__(self, is_leaky: bool = False, leakage: float = 0.2):
        self._leakage = leakage if is_leaky else 0

    def __call__(self, v: float):
        return v * (self._leakage if v < 0 else 1)

    def name(self) -> str:
        return "relu"

    def is_leaky(self):
        return self._leakage != 0

    def get_leakage(self):
        return self._leakage


class Sigmoid(ActivationFunction):

    def __call__(self, v: float):
        # print(v)
        try:
            return 1/(1 + math.e**-v)
        except OverflowError:
            return 0.0

    def name(self) -> str:
        return "sigmoid"


class Identity(ActivationFunction):

    def __call__(self, v: float):
        return v

    def name(self) -> str:
        return "identity"


def softmax(v: [float]) -> [float]:
    """
    Applies softmax to the given input vector.
    Parameters
    ----------
    v

    Returns
    -------

    """
    l = [math.e ** i for i in v]
    s = sum(l)
    return [i/s for i in l]

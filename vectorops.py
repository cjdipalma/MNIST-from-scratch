'''
A simple collection of vector operations.
Using numpy is preferred.
'''

def dot(v1 : [float], v2 : [float]) -> float:
    """
    Calculates the dot product for two lists of floats of the same length.
    """
    if len(v1) != len(v2):
        raise ValueError("Vector lengths are not the same")
    return sum(v1[i]*v2[i] for i in range(len(v1)))

def mul(v : [float], s : float) -> float:
    """
    Scalar multiplication.
    """
    return sum(v[i]*s for i in range(len(v)))

def add(v1 : [float], v2 : [float]) -> [float]:
    """
    Vector addition.
    """
    if len(v1) != len(v2):
        raise ValueError("Vector lengths are not the same")
    return [v1[i]+v2[i] for i in range(len(v1))]

def sub(v1 : [float], v2 : [float]) -> [float]:
    """
    Vector subtraction.
    """
    if len(v1) != len(v2):
        raise ValueError("Vector lengths are not the same")
    return [v1[i]-v2[i] for i in range(len(v1))]

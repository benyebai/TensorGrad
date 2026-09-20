"""
Similar to Value class in MiniGrad now using numpy arrays
https://github.com/benyebai/MiniGrad
"""
import numpy as np



class Tensor:
    def __init__(self, data, children=(), name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.children = set(children)
        self.name = name
        self._backward = lambda: None

    def __add__(self, other):
        # python is stupid so we use it to do scalars
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.value + other.value, (self, other))

        def _backward():
            # since its addition, my grad is just
            # the deriv of my "parent" * the deriv of myself to my parent which is 1 in the addition case
            self.grad += 1 * res.grad
            other.grad += 1 * res.grad

        # then we install this backward function to our result! So when our result calls backwards he knows how to update us
        res._backward = _backward

        return res

"""
Similar to Value class in MiniGrad now using numpy arrays
https://github.com/benyebai/MiniGrad
"""
import numpy as np


def sum_to_shape(grad, shape):
    # holy crap finally understood
    # start from the back line em up, if its a 1 u sum up that dimension, if its equal u skip
    # if its not 1 and its not equal, ur cooked
    # and then any remaining on the left u just sum up

    # summing from the left till dimension shape equal
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # so for like [3, 4, 5]  [1, 4, 1]
    # you would sum that axis and keep the dimension
    # imagine its a 4x5 sheet and then duplicate across the z axis by 3
    # now same thing with 4x1 and then 1 sheet
    # u collapse the 3 and then skip the 4 then collapse the 5 (but u just keep dims so it matches)
    for axis_index, size in enumerate(shape):
        if size == 1 and grad.shape[axis_index] != 1:
            grad = grad.sum(axis=axis_index, keepdims=True)
        elif grad.shape[axis_index] != size:
            raise ValueError("ye this collapsation is not valid my guy")

    return grad

class Tensor:
    def __init__(self, data, children=(), name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.children = set(children)
        self.name = name
        self._backward = lambda: None

    # should be no different than scalar addition
    # I lied nvm matrix is weird
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        res = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += sum_to_shape(res.grad, self.grad.shape)
            other.grad += sum_to_shape(res.grad, other.grad.shape)

        res._backward = _backward

        return res

    def __radd__(self, other):
        return self + other

    # first create negation so then subtraction is already built in
    def __neg__(self):

        res = Tensor(-self.data, (self,))

        def _backward():
            # ok so it would look like
            # y = -x
            # then dy/dx = -1
            self.grad += -1 * res.grad

        res._backward = _backward

        return res

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        res = Tensor(self.data * other.data, (self, other))

        # now with muiltiplying again,
        # case 1 same shape: if its same shape then same as mini grad, res.grad * other.data vice versa
        # case 2 broadcasting: o its just case 1 mixed with our addition backwards function! nice
        def _backward():
            self.grad += sum_to_shape(res.grad * other.data, self.grad.shape)
            other.grad += sum_to_shape(res.grad * self.data, other.grad.shape)

        res._backward = _backward

        return res

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self

    def backward(self):
        topologialReverseOrder = []
        visited = set()

        # its really not that deep, just make sure u traverse the graph once and add to list after u visit all its children
        def create_top_reverse_order(node):
            if node not in visited:
                visited.add(node)

                for n in node.children:
                    create_top_reverse_order(n)

                topologialReverseOrder.append(node)

        create_top_reverse_order(self)

        # this is important haha oops
        self.grad = np.ones_like(self.data)

        for n in reversed(topologialReverseOrder):
            n._backward()

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

    # should be no different than scalar addition
    def __add__(self, other):
        res = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1 * res.grad
            other.grad += 1 * res.grad

        res._backward = _backward

        return res

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

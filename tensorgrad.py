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

    # only support constant exponents! Gets tricky if log comes into play and dont really need it
    def __pow__(self, exponent):
        # removed the ability for the exponent to be a value, because it screws shi up
        res = Tensor(self.data**exponent, (self,))

        def _backward():
            # ok so it would look like
            # x ** constant = z
            # then dz/dx = constant x x^(constant - 1)
            self.grad += (exponent * (self.data ** (exponent - 1))) * res.grad

        res._backward = _backward

        return res

    def exp(self):
        res = Tensor(np.exp(self.data), (self,))

        def _backward():
            # ok so it would look like
            # y = e^x, uh if i remember correct
            # then dy/dx = e^x
            self.grad += res.data * res.grad

        res._backward = _backward
        return res

    # ezpz
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def sum(self, axis=None, keepdims=False):
        res = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,))

        def _backward():
            grad = res.grad

            # if numpy removed reduced axes, put them back as size-1 axes.
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(
                    # convert negative axes to positive
                    current_axis % self.data.ndim
                    for current_axis in axes
                )

                # added the 1 axis to the right spot
                for current_axis in sorted(axes):
                    grad = np.expand_dims(grad, axis=current_axis)

            # Now that we have the right shape, turn them all into the grad
            self.grad += np.broadcast_to(grad, self.data.shape)

        res._backward = _backward
        return res

    def mean(self, axis=None, keepdims=False):
        res = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,))

        # build our axis a bit earlier since we will need it to get the total
        # amount of elements in our mean
        if axis is None:
            axes = tuple(range(self.data.ndim))
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(current_axis % self.data.ndim for current_axis in axes)

        # the total elements that we added up
        count = 1
        for current_axis in axes:
            count *= self.data.shape[current_axis]

        # same idea as sum, just instead of grad, its just grad/total_elements_added
        def _backward():
            grad = res.grad / count

            if axis is not None and not keepdims:
                for current_axis in sorted(axes):
                    grad = np.expand_dims(grad, axis=current_axis)

            self.grad += np.broadcast_to(grad, self.data.shape)

        res._backward = _backward
        return res

    def reshape(self, *shape):
        res = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self.grad += res.grad.reshape(self.data.shape)

        res._backward = _backward
        return res

    def transpose(self, *axes):
        res = Tensor(self.data.transpose(*axes), (self,))

        def _backward():
            # if there is no axes, then tranpose reversed so just tranpose again
            if len(axes) == 0:
                self.grad += res.grad.transpose()
            else:
                # this one is weird, based on the transpose u have to find out where the og
                # dimension went, so argsort just does that for u
                self.grad += res.grad.transpose(tuple(np.argsort(axes)))

        res._backward = _backward
        return res

    def __getitem__(self, index):
        res = Tensor(self.data[index], (self,))

        def _backward():
            contribution = np.zeros_like(self.data)
            contribution[index] += res.grad
            self.grad += contribution

        res._backward = _backward
        return res

    def __matmul__(self, other):
        res = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self_was_vector = self.data.ndim == 1
            other_was_vector = other.data.ndim == 1

            # turn our first vector into a row of 1, n
            self_matrix = (
                np.expand_dims(self.data, axis=-2) if self_was_vector else self.data
            )
            # turn our second vector into a row of n, 1
            other_matrix = (
                np.expand_dims(other.data, axis=-1) if other_was_vector else other.data
            )

            # restore the matrix axes numpy removed from the forward
            grad = res.grad
            if self_was_vector and other_was_vector:
                grad = grad.reshape(1, 1)
            elif self_was_vector:
                grad = np.expand_dims(grad, axis=-2)
            elif other_was_vector:
                grad = np.expand_dims(grad, axis=-1)

            self_contribution = grad @ np.swapaxes(other_matrix, -1, -2)
            other_contribution = np.swapaxes(self_matrix, -1, -2) @ grad

            # remove the same temporary axes before reducing broadcasted
            # batch dimensions back to each parent's original shape.
            if self_was_vector:
                self_contribution = np.squeeze(self_contribution, axis=-2)
            if other_was_vector:
                other_contribution = np.squeeze(other_contribution, axis=-1)

            self.grad += sum_to_shape(self_contribution, self.data.shape)
            other.grad += sum_to_shape(other_contribution, other.data.shape)

        res._backward = _backward
        return res

    def log(self):
        res = Tensor(np.log(self.data), (self,))

        # dy = log(x) -> 1/x
        def _backward():
            self.grad += res.grad * 1 / self.data

        res._backward = _backward
        return res

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def silu(self):
        return self * self.sigmoid()

    def logsumexp(self, axis=None, keepdims=False):
        # Always retain reduced axes internally for safe broadcasting.
        maximum = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - maximum
        exponentials = np.exp(shifted)
        denominator = np.sum(exponentials, axis=axis, keepdims=True)

        result_with_dims = maximum + np.log(denominator)

        if keepdims:
            result_data = result_with_dims
        elif axis is None:
            result_data = np.squeeze(result_with_dims)
        else:
            result_data = np.squeeze(result_with_dims, axis=axis)

        res = Tensor(result_data, (self,))

        def _backward():
            grad = res.grad

            # this code is so ugly, but this is just restoring removed axis so that when we
            # muitiply it by the res.grad it broadcasts properly
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(current_axis % self.data.ndim for current_axis in axes)

                for current_axis in sorted(axes):
                    grad = np.expand_dims(grad, axis=current_axis)

            # The derivative of logsumexp is softmax.
            # so just e^x / sum(e^xi - m)
            probabilities = exponentials / denominator
            self.grad += np.broadcast_to(grad, self.data.shape) * probabilities

        res._backward = _backward
        return res

    def softmax(self, axis):
        return (self - self.logsumexp(axis=axis, keepdims=True)).exp()

    def detach(self):
        return Tensor(self.data.copy())

    def tanh(self):
        res = Tensor(np.tanh(self.data), (self,))

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - res.data**2) * res.grad

        res._backward = _backward
        return res

    def backward(self, gradient=None):
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
        if gradient is None:
            self.grad = np.ones_like(self.data)
        else:
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.shape != self.data.shape:
                raise ValueError("Gradient shape must match output shape")
            self.grad = gradient.copy()

        for n in reversed(topologialReverseOrder):
            n._backward()

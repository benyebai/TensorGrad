"""
Focused on implementation, the tests were written with the help of AI
"""

import unittest

import numpy as np

from tensorgrad import Tensor


def assert_gradcheck(test_case, function, *input_data, epsilon=1e-6,
                     rtol=1e-5, atol=1e-7):
    """Compare TensorGrad gradients with central finite differences."""
    arrays = [np.asarray(data, dtype=np.float64).copy() for data in input_data]
    tensors = [Tensor(array.copy()) for array in arrays]

    output = function(*tensors)
    if output.data.shape != ():
        raise ValueError("Gradient checks require a scalar Tensor output")
    output.backward()
    analytical_gradients = [tensor.grad.copy() for tensor in tensors]

    numerical_gradients = []
    for input_index, array in enumerate(arrays):
        numerical = np.zeros_like(array)
        for element_index in np.ndindex(array.shape):
            original = array[element_index]

            array[element_index] = original + epsilon
            plus_inputs = [Tensor(item.copy()) for item in arrays]
            plus = function(*plus_inputs).data.item()

            array[element_index] = original - epsilon
            minus_inputs = [Tensor(item.copy()) for item in arrays]
            minus = function(*minus_inputs).data.item()

            array[element_index] = original
            numerical[element_index] = (plus - minus) / (2.0 * epsilon)

        numerical_gradients.append(numerical)

    for input_index, (analytical, numerical) in enumerate(
        zip(analytical_gradients, numerical_gradients)
    ):
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient mismatch for input {input_index}",
        )


class TestTensorFoundation(unittest.TestCase):
    def test_constructor_scalar_vector_matrix(self):
        for data, shape in [
            (3.0, ()),
            ([1.0, 2.0], (2,)),
            ([[1.0, 2.0], [3.0, 4.0]], (2, 2)),
        ]:
            with self.subTest(shape=shape):
                tensor = Tensor(data)
                self.assertIsInstance(tensor.data, np.ndarray)
                self.assertEqual(tensor.data.dtype, np.dtype("float64"))
                self.assertEqual(tensor.data.shape, shape)
                self.assertEqual(tensor.grad.shape, shape)
                np.testing.assert_array_equal(tensor.grad, np.zeros(shape))

    def test_backward_accepts_explicit_upstream_gradient(self):
        value = Tensor([2.0, 3.0, 4.0])
        result = value * 2.0
        result.backward([1.0, 3.0, -2.0])
        np.testing.assert_array_equal(result.grad, [1.0, 3.0, -2.0])
        np.testing.assert_array_equal(value.grad, [2.0, 6.0, -4.0])

    def test_backward_rejects_wrong_upstream_gradient_shape(self):
        result = Tensor([1.0, 2.0]) * 2.0
        with self.assertRaises(ValueError):
            result.backward([1.0])

    def test_backward_without_gradient_seeds_output_with_ones(self):
        value = Tensor([2.0, 3.0])
        result = value * 4.0
        result.backward()
        np.testing.assert_array_equal(result.grad, [1.0, 1.0])
        np.testing.assert_array_equal(value.grad, [4.0, 4.0])

    def test_add_scalar_forward_and_backward(self):
        left = Tensor(2.0)
        right = Tensor(3.0)
        result = left + right

        np.testing.assert_array_equal(result.data, np.array(5.0))
        result.backward()
        np.testing.assert_array_equal(left.grad, np.array(1.0))
        np.testing.assert_array_equal(right.grad, np.array(1.0))

    def test_add_same_shape_matrix_forward_and_backward(self):
        left = Tensor([[1.0, 2.0], [3.0, 4.0]])
        right = Tensor([[10.0, 20.0], [30.0, 40.0]])
        result = left + right
        np.testing.assert_array_equal(result.data, [[11.0, 22.0], [33.0, 44.0]])
        result.backward()  # Implicitly seeds each output with 1.
        np.testing.assert_array_equal(left.grad, np.ones((2, 2)))
        np.testing.assert_array_equal(right.grad, np.ones((2, 2)))

    def test_add_row_vector_broadcast_backward(self):
        matrix = Tensor([[1.0, 2.0], [3.0, 4.0]])
        row = Tensor([10.0, 20.0])
        result = matrix + row
        np.testing.assert_array_equal(result.data, [[11.0, 22.0], [13.0, 24.0]])
        result.backward()
        np.testing.assert_array_equal(matrix.grad, np.ones((2, 2)))
        np.testing.assert_array_equal(row.grad, [2.0, 2.0])
        self.assertEqual(row.grad.shape, (2,))

    def test_add_column_vector_broadcast_backward(self):
        matrix = Tensor([[1.0, 2.0], [3.0, 4.0]])
        column = Tensor([[10.0], [20.0]])
        result = matrix + column
        np.testing.assert_array_equal(result.data, [[11.0, 12.0], [23.0, 24.0]])
        result.backward()
        np.testing.assert_array_equal(matrix.grad, np.ones((2, 2)))
        np.testing.assert_array_equal(column.grad, [[2.0], [2.0]])
        self.assertEqual(column.grad.shape, (2, 1))

    def test_add_scalar_tensor_broadcast_backward(self):
        matrix = Tensor([[1.0, 2.0], [3.0, 4.0]])
        scalar = Tensor(10.0)
        result = matrix + scalar
        np.testing.assert_array_equal(result.data, [[11.0, 12.0], [13.0, 14.0]])
        result.backward()
        np.testing.assert_array_equal(matrix.grad, np.ones((2, 2)))
        np.testing.assert_array_equal(scalar.grad, np.array(4.0))
        self.assertEqual(scalar.grad.shape, ())

    def test_add_three_dimensional_singleton_axes_backward(self):
        cube = Tensor(np.zeros((3, 4, 5)))
        rows = Tensor(np.ones((1, 4, 1)))
        result = cube + rows
        self.assertEqual(result.data.shape, (3, 4, 5))
        result.backward()
        np.testing.assert_array_equal(cube.grad, np.ones((3, 4, 5)))
        np.testing.assert_array_equal(rows.grad, np.full((1, 4, 1), 15.0))

    def test_add_incompatible_shapes_raise(self):
        with self.assertRaises(ValueError):
            Tensor(np.zeros((2, 3))) + Tensor(np.zeros((4,)))

    def test_numeric_addition_both_orders(self):
        value = Tensor([1.0, 2.0])
        np.testing.assert_array_equal((value + 3).data, [4.0, 5.0])
        np.testing.assert_array_equal((3 + value).data, [4.0, 5.0])

    def test_negation_and_subtraction_backward(self):
        left = Tensor([3.0, 5.0])
        right = Tensor([1.0, 2.0])
        result = left - right
        np.testing.assert_array_equal(result.data, [2.0, 3.0])
        result.backward()
        np.testing.assert_array_equal(left.grad, [1.0, 1.0])
        np.testing.assert_array_equal(right.grad, [-1.0, -1.0])

    def test_reverse_subtraction(self):
        value = Tensor([1.0, 2.0])
        result = 5 - value
        np.testing.assert_array_equal(result.data, [4.0, 3.0])
        result.backward()
        np.testing.assert_array_equal(value.grad, [-1.0, -1.0])

    def test_multiply_same_shape_backward(self):
        left = Tensor([2.0, 3.0])
        right = Tensor([4.0, 5.0])
        result = left * right
        np.testing.assert_array_equal(result.data, [8.0, 15.0])
        result.backward()
        np.testing.assert_array_equal(left.grad, [4.0, 5.0])
        np.testing.assert_array_equal(right.grad, [2.0, 3.0])

    def test_multiply_broadcast_backward(self):
        matrix = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        row = Tensor([10.0, 20.0, 30.0])
        result = matrix * row
        np.testing.assert_array_equal(result.data, [[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]])
        result.backward()
        np.testing.assert_array_equal(matrix.grad, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(row.grad, [5.0, 7.0, 9.0])

    def test_numeric_multiplication_both_orders(self):
        value = Tensor([1.0, 2.0])
        np.testing.assert_array_equal((value * 3).data, [3.0, 6.0])
        np.testing.assert_array_equal((3 * value).data, [3.0, 6.0])

    def test_power_backward(self):
        value = Tensor([2.0, 3.0])
        result = value ** 3
        np.testing.assert_array_equal(result.data, [8.0, 27.0])
        result.backward()
        np.testing.assert_array_equal(value.grad, [12.0, 27.0])

    def test_exp_backward(self):
        value = Tensor([0.0, 1.0])
        result = value.exp()
        result.backward()
        np.testing.assert_allclose(result.data, np.exp([0.0, 1.0]))
        np.testing.assert_allclose(value.grad, np.exp([0.0, 1.0]))

    def test_log_forward_and_backward(self):
        value = Tensor([0.5, 1.0, 2.0, 4.0])
        result = value.log()
        weights = Tensor([1.0, 2.0, 3.0, 4.0])
        (result * weights).sum().backward()

        np.testing.assert_allclose(result.data, np.log(value.data))
        np.testing.assert_allclose(value.grad, weights.data / value.data)

    def test_sigmoid_forward_and_backward(self):
        value = Tensor([-2.0, 0.0, 2.0])
        result = value.sigmoid()
        weights = Tensor([1.0, 2.0, 3.0])
        (result * weights).sum().backward()

        expected = 1.0 / (1.0 + np.exp(-value.data))
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(
            value.grad,
            weights.data * expected * (1.0 - expected),
        )

    def test_silu_forward_and_backward(self):
        value = Tensor([-2.0, 0.0, 2.0])
        result = value.silu()
        weights = Tensor([1.0, 2.0, 3.0])
        (result * weights).sum().backward()

        sigmoid = 1.0 / (1.0 + np.exp(-value.data))
        expected = value.data * sigmoid
        expected_derivative = sigmoid + value.data * sigmoid * (1.0 - sigmoid)
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(
            value.grad,
            weights.data * expected_derivative,
        )

    def test_logsumexp_axis_keepdims_forward(self):
        data = np.array([[1.0, 2.0, 3.0],
                         [-2.0, 0.0, 4.0]])
        value = Tensor(data)

        result = value.logsumexp(axis=1, keepdims=True)
        maximum = np.max(data, axis=1, keepdims=True)
        expected = maximum + np.log(
            np.sum(np.exp(data - maximum), axis=1, keepdims=True)
        )

        self.assertEqual(result.data.shape, (2, 1))
        np.testing.assert_allclose(result.data, expected)

    def test_logsumexp_axis_without_keepdims_backward(self):
        data = np.array([[1.0, 2.0, 3.0],
                         [-2.0, 0.0, 4.0]])
        value = Tensor(data)
        weights = Tensor([2.0, 3.0])

        result = value.logsumexp(axis=1, keepdims=False)
        (result * weights).sum().backward()

        shifted = data - np.max(data, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(
            np.exp(shifted), axis=1, keepdims=True
        )
        expected_gradient = probabilities * weights.data[:, None]

        self.assertEqual(result.data.shape, (2,))
        np.testing.assert_allclose(value.grad, expected_gradient)

    def test_logsumexp_extreme_values_are_finite(self):
        value = Tensor([[999.0, 1000.0], [-1000.0, -999.0]])
        result = value.logsumexp(axis=-1)

        self.assertTrue(np.all(np.isfinite(result.data)))
        expected = np.array([
            1000.0 + np.log1p(np.exp(-1.0)),
            -999.0 + np.log1p(np.exp(-1.0)),
        ])
        np.testing.assert_allclose(result.data, expected)

    def test_softmax_last_axis_forward_and_normalization(self):
        data = np.array([[1.0, 2.0, 3.0],
                         [1000.0, 1001.0, 1002.0]])
        value = Tensor(data)
        result = value.softmax(axis=-1)

        shifted = data - np.max(data, axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.sum(
            np.exp(shifted), axis=-1, keepdims=True
        )

        self.assertTrue(np.all(np.isfinite(result.data)))
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(result.data.sum(axis=-1), np.ones(2))

    def test_softmax_backward_with_nonuniform_upstream(self):
        data = np.array([[0.2, -0.4, 1.1],
                         [1.5, 0.3, -0.7]])
        upstream = np.array([[1.0, 2.0, -1.0],
                             [0.5, -2.0, 3.0]])
        value = Tensor(data)
        result = value.softmax(axis=1)
        (result * Tensor(upstream)).sum().backward()

        shifted = data - np.max(data, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(
            np.exp(shifted), axis=1, keepdims=True
        )
        expected_gradient = probabilities * (
            upstream - np.sum(upstream * probabilities, axis=1, keepdims=True)
        )
        np.testing.assert_allclose(value.grad, expected_gradient)

    def test_softmax_passes_finite_difference_gradcheck(self):
        weights = np.array([[1.0, -0.5, 2.0],
                            [-1.0, 3.0, 0.25]])

        def expression(value):
            return (value.softmax(axis=1) * Tensor(weights)).sum()

        assert_gradcheck(
            self,
            expression,
            [[0.2, -0.4, 1.1], [1.5, 0.3, -0.7]],
        )

    def test_division_both_orders_backward(self):
        numerator = Tensor([6.0, 8.0])
        denominator = Tensor([2.0, 4.0])
        result = numerator / denominator
        np.testing.assert_array_equal(result.data, [3.0, 2.0])
        result.backward()
        np.testing.assert_allclose(numerator.grad, [0.5, 0.25])
        np.testing.assert_allclose(denominator.grad, [-1.5, -0.5])

        value = Tensor([2.0, 4.0])
        reverse = 8 / value
        reverse.backward()
        np.testing.assert_array_equal(reverse.data, [4.0, 2.0])
        np.testing.assert_allclose(value.grad, [-2.0, -0.5])

    def test_composed_expression_accumulates_correctly(self):
        value = Tensor([2.0, 3.0])
        result = value * value + value
        result.backward()
        np.testing.assert_array_equal(result.data, [6.0, 12.0])
        np.testing.assert_array_equal(value.grad, [5.0, 7.0])

    def test_composed_expression_passes_finite_difference_gradcheck(self):
        def expression(left, right):
            return ((left * right).silu() + left ** 2).mean()

        assert_gradcheck(
            self,
            expression,
            [[0.2, -0.4, 0.7], [1.1, -0.8, 0.3]],
            [0.5, -1.2, 0.9],
        )

    def test_reductions_pass_finite_difference_gradcheck(self):
        def expression(value):
            return (value.mean(axis=(0, 2), keepdims=True) ** 2).sum()

        assert_gradcheck(
            self,
            expression,
            np.arange(1.0, 25.0).reshape(2, 3, 4) / 10.0,
        )

    def test_shape_operations_pass_finite_difference_gradcheck(self):
        weights = np.arange(1.0, 25.0).reshape(3, 4, 2) / 10.0

        def expression(value):
            transformed = value.reshape(2, 3, 4).transpose(1, 2, 0)
            return (transformed * Tensor(weights)).sum()

        assert_gradcheck(self, expression, np.arange(24.0) / 10.0)

    def test_division_log_exp_pass_finite_difference_gradcheck(self):
        def expression(numerator, denominator):
            return ((numerator / denominator).exp().log()).mean()

        assert_gradcheck(
            self,
            expression,
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
            [1.5, 2.0, 2.5],
        )

    def test_sigmoid_passes_finite_difference_gradcheck(self):
        weights = np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]])

        def expression(value):
            return (value.sigmoid() * Tensor(weights)).sum()

        assert_gradcheck(
            self,
            expression,
            [[-1.5, 0.0, 2.0], [0.7, -0.3, 1.2]],
        )

    def test_slicing_passes_finite_difference_gradcheck(self):
        weights = np.array([[1.0, -2.0], [3.0, 0.5]])

        def expression(value):
            return (value[1:, 1:3] * Tensor(weights)).sum()

        assert_gradcheck(
            self,
            expression,
            np.arange(12.0).reshape(3, 4) / 10.0,
        )

    def test_reshape_forward_and_backward(self):
        value = Tensor(np.arange(6.0))
        weights = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = (value.reshape(2, 3) * weights).sum()
        result.backward()
        np.testing.assert_array_equal(value.grad, np.arange(1.0, 7.0))

    def test_transpose_default_matrix_forward_and_backward(self):
        value = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        transposed = value.transpose()
        np.testing.assert_array_equal(transposed.data, value.data.T)
        (transposed * weights).sum().backward()
        np.testing.assert_array_equal(value.grad, weights.data.T)

    def test_transpose_three_dimensional_inverse_permutation(self):
        value = Tensor(np.arange(24.0).reshape(2, 3, 4))
        weights = Tensor(np.arange(1.0, 25.0).reshape(3, 4, 2))
        transposed = value.transpose(1, 2, 0)
        np.testing.assert_array_equal(transposed.data, np.transpose(value.data, (1, 2, 0)))
        (transposed * weights).sum().backward()
        np.testing.assert_array_equal(value.grad, np.transpose(weights.data, (2, 0, 1)))

    def test_index_single_row_backward(self):
        value = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        selected = value[1]
        (selected * Tensor([2.0, 3.0, 4.0])).sum().backward()
        np.testing.assert_array_equal(selected.data, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(value.grad,
                                      [[0.0, 0.0, 0.0],
                                       [2.0, 3.0, 4.0]])

    def test_index_tuple_scalar_backward(self):
        value = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        selected = value[0, 2]
        (selected * 7.0).backward()
        np.testing.assert_array_equal(selected.data, np.array(3.0))
        np.testing.assert_array_equal(value.grad,
                                      [[0.0, 0.0, 7.0],
                                       [0.0, 0.0, 0.0]])

    def test_index_slice_backward(self):
        value = Tensor(np.arange(12.0).reshape(3, 4))
        selected = value[1:, 1:3]
        weights = Tensor([[1.0, 2.0], [3.0, 4.0]])
        (selected * weights).sum().backward()
        np.testing.assert_array_equal(selected.data, [[5.0, 6.0], [9.0, 10.0]])
        np.testing.assert_array_equal(value.grad,
                                      [[0.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 2.0, 0.0],
                                       [0.0, 3.0, 4.0, 0.0]])

    def test_matrix_multiplication_2d_forward_and_backward(self):
        left = Tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        right = Tensor([[7.0, 8.0],
                        [9.0, 10.0],
                        [11.0, 12.0]])
        upstream = Tensor([[1.0, 2.0],
                           [3.0, 4.0]])

        product = left @ right
        loss = (product * upstream).sum()
        loss.backward()

        np.testing.assert_array_equal(product.data,
                                      [[58.0, 64.0],
                                       [139.0, 154.0]])
        np.testing.assert_array_equal(left.grad,
                                      [[23.0, 29.0, 35.0],
                                       [53.0, 67.0, 81.0]])
        np.testing.assert_array_equal(right.grad,
                                      [[13.0, 18.0],
                                       [17.0, 24.0],
                                       [21.0, 30.0]])

    def test_matrix_multiplication_3d_batched_forward_and_backward(self):
        left_data = np.arange(1.0, 13.0).reshape(2, 2, 3)
        right_data = np.arange(1.0, 13.0).reshape(2, 3, 2)
        upstream_data = np.arange(1.0, 9.0).reshape(2, 2, 2)
        left = Tensor(left_data)
        right = Tensor(right_data)

        product = left @ right
        (product * Tensor(upstream_data)).sum().backward()

        np.testing.assert_array_equal(product.data, np.matmul(left_data, right_data))
        np.testing.assert_array_equal(
            left.grad,
            np.matmul(upstream_data, np.swapaxes(right_data, -1, -2)),
        )
        np.testing.assert_array_equal(
            right.grad,
            np.matmul(np.swapaxes(left_data, -1, -2), upstream_data),
        )

    def test_matrix_multiplication_4d_forward_and_backward(self):
        left_data = np.arange(1.0, 49.0).reshape(2, 2, 3, 4)
        right_data = np.arange(1.0, 81.0).reshape(2, 2, 4, 5)
        upstream_data = np.arange(1.0, 61.0).reshape(2, 2, 3, 5)
        left = Tensor(left_data)
        right = Tensor(right_data)

        product = left @ right
        (product * Tensor(upstream_data)).sum().backward()

        np.testing.assert_array_equal(product.data, np.matmul(left_data, right_data))
        np.testing.assert_array_equal(
            left.grad,
            np.matmul(upstream_data, np.swapaxes(right_data, -1, -2)),
        )
        np.testing.assert_array_equal(
            right.grad,
            np.matmul(np.swapaxes(left_data, -1, -2), upstream_data),
        )

    def test_matrix_multiplication_broadcasted_batch_backward(self):
        left_data = np.arange(1.0, 25.0).reshape(2, 2, 2, 3)
        right_data = np.arange(1.0, 13.0).reshape(1, 2, 3, 2)
        upstream_data = np.arange(1.0, 17.0).reshape(2, 2, 2, 2)
        left = Tensor(left_data)
        right = Tensor(right_data)

        product = left @ right
        (product * Tensor(upstream_data)).sum().backward()

        expected_left_grad = np.matmul(
            upstream_data,
            np.swapaxes(right_data, -1, -2),
        )
        uncollapsed_right_grad = np.matmul(
            np.swapaxes(left_data, -1, -2),
            upstream_data,
        )
        expected_right_grad = uncollapsed_right_grad.sum(axis=0, keepdims=True)

        np.testing.assert_array_equal(product.data, np.matmul(left_data, right_data))
        np.testing.assert_array_equal(left.grad, expected_left_grad)
        np.testing.assert_array_equal(right.grad, expected_right_grad)
        self.assertEqual(right.grad.shape, right_data.shape)

    def test_batched_matrix_multiplication_passes_gradcheck(self):
        def expression(left, right, weights):
            return ((left @ right) * weights).sum()

        assert_gradcheck(
            self,
            expression,
            np.arange(1.0, 13.0).reshape(2, 2, 3) / 10.0,
            np.arange(1.0, 7.0).reshape(1, 3, 2) / 10.0,
            np.arange(1.0, 9.0).reshape(2, 2, 2) / 10.0,
        )

    def test_matrix_multiplication_vector_dot_vector_backward(self):
        left = Tensor([1.0, 2.0, 3.0])
        right = Tensor([4.0, 5.0, 6.0])

        product = left @ right
        loss = product * 3.0
        loss.backward()

        np.testing.assert_array_equal(product.data, np.array(32.0))
        self.assertEqual(product.data.shape, ())
        np.testing.assert_array_equal(left.grad, [12.0, 15.0, 18.0])
        np.testing.assert_array_equal(right.grad, [3.0, 6.0, 9.0])

    def test_matrix_multiplication_matrix_by_vector_backward(self):
        matrix = Tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
        vector = Tensor([7.0, 8.0, 9.0])
        upstream = Tensor([2.0, 3.0])

        product = matrix @ vector
        (product * upstream).sum().backward()

        np.testing.assert_array_equal(product.data, [50.0, 122.0])
        np.testing.assert_array_equal(matrix.grad,
                                      [[14.0, 16.0, 18.0],
                                       [21.0, 24.0, 27.0]])
        np.testing.assert_array_equal(vector.grad, [14.0, 19.0, 24.0])

    def test_matrix_multiplication_vector_by_matrix_backward(self):
        vector = Tensor([1.0, 2.0, 3.0])
        matrix = Tensor([[4.0, 5.0],
                         [6.0, 7.0],
                         [8.0, 9.0]])
        upstream = Tensor([2.0, 3.0])

        product = vector @ matrix
        (product * upstream).sum().backward()

        np.testing.assert_array_equal(product.data, [40.0, 46.0])
        np.testing.assert_array_equal(vector.grad, [23.0, 33.0, 43.0])
        np.testing.assert_array_equal(matrix.grad,
                                      [[2.0, 3.0],
                                       [4.0, 6.0],
                                       [6.0, 9.0]])



if __name__ == "__main__":
    unittest.main()

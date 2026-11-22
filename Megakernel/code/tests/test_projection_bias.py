"""CPU rounding counterexample for the QKV GEMM-to-bias boundary."""
import unittest
import numpy as np

from stage_a.workloads import projection_with_bias_reference


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def to(self, dtype):
        return Tensor(self.values.astype(dtype))

    @property
    def T(self):
        return Tensor(self.values.T)

    def __matmul__(self, other):
        return Tensor(self.values @ other.values)

    def __add__(self, other):
        return Tensor(self.values + other.values)


class ProjectionBiasTests(unittest.TestCase):
    def test_intermediate_fp16_boundary_changes_bias_tie(self):
        x = np.array([[1.0009765625]], dtype=np.float16)
        w = x.copy()
        bias = np.array([.00048828125], dtype=np.float16)
        actual = projection_with_bias_reference(Tensor(x), Tensor(w), Tensor(bias), np.float16).values
        single_round = (x.astype(np.float32) @ w.astype(np.float32).T + bias.astype(np.float32)).astype(np.float16)
        self.assertEqual(float(actual[0, 0]), 1.001953125)
        self.assertEqual(float(single_round[0, 0]), 1.0029296875)
        self.assertNotEqual(float(actual[0, 0]), float(single_round[0, 0]))

    def test_batched_projection_keeps_output_storage_dtype(self):
        x = np.array([[[.25, .75], [-.5, 1.5]]], dtype=np.float16)
        w = np.array([[1., -.25], [-.75, .5]], dtype=np.float16)
        bias = np.array([.125, -.25], dtype=np.float16)
        actual = projection_with_bias_reference(Tensor(x), Tensor(w), Tensor(bias), np.float16).values
        projected = (x.astype(np.float32) @ w.astype(np.float32).T).astype(np.float16)
        expected = (projected.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (1, 2, 2))
        self.assertEqual(actual.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()

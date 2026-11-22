"""CPU regressions for the B0 failures observed before candidate execution."""
import math
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from stage_a.workloads import Workload, stable_softmax_reference


class ArrayTensor:
    """NumPy adapter for the reference's tensor operations; no GPU samples."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def double(self):
        return ArrayTensor(self.values.astype(np.float64))

    def amax(self, dim, keepdim):
        return ArrayTensor(self.values.max(axis=dim, keepdims=keepdim))

    def exp(self):
        return ArrayTensor(np.exp(self.values))

    def sum(self, dim, keepdim):
        return ArrayTensor(self.values.sum(axis=dim, keepdims=keepdim))

    def __sub__(self, other):
        return ArrayTensor(self.values - other.values)

    def __truediv__(self, other):
        return ArrayTensor(self.values / other.values)


def cpu_constructor_backend():
    module = ModuleType("torch")
    module.float16, module.float32 = np.float16, np.float32

    class Generator:
        def __init__(self, device):
            self.rng = None

        def manual_seed(self, seed):
            self.rng = np.random.default_rng(seed)
            return self

    def randn(shape, device, dtype, generator):
        return generator.rng.standard_normal(shape).astype(dtype)

    def empty(shape, device, dtype):
        return np.empty(shape, dtype=dtype)

    module.Generator = Generator
    module.randn = randn
    module.empty = empty
    module.ones = lambda shape, device, dtype: np.ones(shape, dtype=dtype)
    return module


class NumericPreflightTests(unittest.TestCase):
    def test_fp32_logsumexp_subtraction_loses_probability_normalization(self):
        logits = np.array([[1e6, 1e6 - 1, 1e6 - 2]], dtype=np.float32)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        lse = logits.max(axis=-1, keepdims=True) + np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        old = np.exp(logits - lse)
        self.assertGreater(abs(float(old.sum()) - 1.), .01)
        corrected = stable_softmax_reference(ArrayTensor(logits)).values
        expected = np.array([[1., math.exp(-1), math.exp(-2)]])
        expected /= expected.sum()
        np.testing.assert_allclose(corrected, expected, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(float(corrected.sum()), 1., places=14)

    def test_reference_handles_causal_mask_and_low_probability_tail(self):
        logits = np.array([[1000., -np.inf, -np.inf], [1000., 999., -np.inf],
                           [1000., 0., -1000.]], dtype=np.float32)
        probability = stable_softmax_reference(ArrayTensor(logits)).values
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(-1), np.ones(3), atol=1e-14)
        self.assertEqual(probability[0, 0], 1.)
        self.assertEqual(probability[1, 2], 0.)

    def test_qwen_stress_changes_activations_and_preserves_seeded_weights(self):
        case = dict(workload="W7", strategy="S1", params=dict(m=1, batch=1, k=896, n=4864,
                    bn=32, merge=1, dtype="float16"))
        assets = dict(qwen_status="READY", weights="QWEN_SHAPE_SYNTHETIC_WEIGHTS")
        with patch.dict(sys.modules, {"torch": cpu_constructor_backend()}):
            normal = Workload(case, "cpu", 20261119, assets, "normal")
            extreme = Workload(case, "cpu", 20261119, assets, "extreme")
        for name in normal.ws:
            np.testing.assert_array_equal(normal.ws[name], extreme.ws[name])
        self.assertFalse(np.array_equal(normal.args[0], extreme.args[0]))
        self.assertEqual(extreme.args[0].reshape(-1)[17], 4.)
        self.assertEqual(extreme.args[0].reshape(-1)[29], -4.)
        self.assertEqual(extreme.input_contract["version"], "activation_stress_fixed_model_weights_v2")

        x, g, wg, wu, wd = extreme.args[:5]
        z = (x.astype(np.float32) / np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + 1e-6)).astype(np.float16)
        z = (z * g).astype(np.float16).astype(np.float32)

        def evaluate(weights):
            gate = (z @ weights[0].astype(np.float32).T).astype(np.float16)
            up = (z @ weights[1].astype(np.float32).T).astype(np.float16)
            gate32 = gate.astype(np.float32)
            silu = (gate32 / (1. + np.exp(-np.clip(gate32, -80., 80.)))).astype(np.float16)
            activation = (silu * up).astype(np.float16)
            return activation.astype(np.float32) @ weights[2].astype(np.float32).T

        corrected = evaluate((wg, wu, wd))
        self.assertTrue(np.isfinite(corrected).all())
        self.assertLess(float(np.abs(corrected).max()), float(np.finfo(np.float16).max))
        old_weights = [weight.copy() for weight in (wg, wu, wd)]
        for weight in old_weights:
            weight.reshape(-1)[::17] = np.float16(.4)
            weight.reshape(-1)[::29] = np.float16(-.4)
        old = evaluate(old_weights)
        self.assertGreater(float(np.abs(old).max()), float(np.finfo(np.float16).max))


if __name__ == "__main__":
    unittest.main()

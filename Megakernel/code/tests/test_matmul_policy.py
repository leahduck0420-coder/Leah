"""CPU checks for applying the accumulation contract in spawned GPU workers."""
from types import SimpleNamespace
import unittest

from stage_a.numeric import configure_matmul_precision


def torch_backend(with_fp16_accumulation):
    flags = dict(allow_tf32=True, allow_fp16_reduced_precision_reduction=True,
                 allow_bf16_reduced_precision_reduction=True)
    if with_fp16_accumulation:
        flags["allow_fp16_accumulation"] = True
    matmul = SimpleNamespace(**flags)
    return SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))


class MatmulPolicyTests(unittest.TestCase):
    def test_all_exposed_reduced_precision_flags_are_disabled(self):
        torch = torch_backend(True)
        recorded = configure_matmul_precision(torch)
        self.assertEqual(len(recorded), 4)
        self.assertTrue(all(value is False for value in recorded.values()))
        self.assertEqual(configure_matmul_precision(torch), recorded)

    def test_torch_without_optional_fp16_accumulation_api(self):
        torch = torch_backend(False)
        recorded = configure_matmul_precision(torch)
        self.assertEqual(set(recorded), {"allow_tf32", "allow_fp16_reduced_precision_reduction",
                                         "allow_bf16_reduced_precision_reduction"})
        self.assertTrue(all(value is False for value in recorded.values()))

    def test_each_fresh_process_backend_needs_its_own_configuration(self):
        parent = torch_backend(True)
        child = torch_backend(True)
        configure_matmul_precision(parent)
        self.assertTrue(child.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        configure_matmul_precision(child)
        self.assertFalse(child.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)


if __name__ == "__main__":
    unittest.main()

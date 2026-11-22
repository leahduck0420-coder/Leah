"""CPU checks of the launch configuration for the stalled W6 task scheduler."""
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

from stage_a.compiler import plan
from stage_a.workloads import Workload


class DispatchTests(unittest.TestCase):
    def test_fixed_and_ready_use_same_single_warp_and_preserve_grid(self):
        class Buffer:
            def __setitem__(self, key, value):
                pass

        class Torch:
            def zeros_like(self, other):
                return Buffer()

        calls = []

        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **options):
                    calls.append(dict(grid=grid, options=options))
                return launch

        kernels = ModuleType("stage_a.kernels")
        models = ModuleType("stage_a.model_kernels")
        models.dispatch = Kernel()
        modules = {"stage_a.kernels": kernels, "stage_a.model_kernels": models}
        for workers in (1, 4, 16):
            case = dict(workload="W6", strategy="S5", params=dict(dag="fork_join", depth=4,
                        tasks=16, workers=workers, schedule="fixed", dtype="float16"))
            job = object.__new__(Workload)
            job.w, job.p = "W6", case["params"]
            job.t = Torch()
            job.data = job.state = job.count = job.trace = object()
            job.x = job.output = object()
            job.orders = {}
            resets = []
            job.reset = lambda: resets.append(True)
            with patch.dict(sys.modules, modules):
                for active in (False, True):
                    launch, grid = job.bind(plan(case, active))
                    launch()
                    self.assertEqual(grid, (workers,))
            self.assertEqual(len(resets), 2)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call["options"]["num_warps"] == 1 for call in calls))
        self.assertTrue(all(call["options"]["enable_fp_fusion"] is False for call in calls))


if __name__ == "__main__":
    unittest.main()

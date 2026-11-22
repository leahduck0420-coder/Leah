"""Regressions for partial-warp atomic returns in the server's Triton 3.4 PTX."""
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from stage_a.compiler import plan
from stage_a.workloads import Workload


class Buffer:
    def __setitem__(self, key, value):
        pass


class Torch:
    def zeros_like(self, other):
        return Buffer()


def captured_launch(case, enabled, qkv_bias_mode=0):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **options):
                calls.append(dict(grid=grid, arguments=args, options=options))
            return launch

    kernels = ModuleType("stage_a.kernels")
    models = ModuleType("stage_a.model_kernels")
    models.dispatch = Kernel()
    job = object.__new__(Workload)
    job.w, job.p = case["workload"], case["params"]
    job.t, job.orders = Torch(), {}
    job.qkv_bias_mode=qkv_bias_mode
    for name in ("data", "state", "count", "trace", "x", "output", "qkv", "bias", "cos",
                 "sin", "q", "kcache", "vcache", "yhead", "tmp", "stats"):
        setattr(job, name, object())
    job.ws = {"self_attn.o_proj.weight": object()}
    job.L, job.KL = 1, 16
    job.reset = lambda: None
    with patch.dict(sys.modules, {"stage_a.kernels": kernels, "stage_a.model_kernels": models}):
        launch, _ = job.bind(plan(case, enabled))
        launch()
    return calls[0]


class SchedulerPaddingTests(unittest.TestCase):
    def test_server_half_warp_atomic_returns_can_split_ready_branch(self):
        # Observed PTX uses tid&15 for a 16-element vector, predicates atomic
        # reads on tid&16==0, initializes other returns to 0, then reduces each
        # 16-lane group independently before a conditional CTA barrier.
        states = [2] * 16
        required = [1] + [0] * 15
        contributions = []
        for lane in range(32):
            slot = lane % 16
            returned = states[slot] if lane < 16 else 0
            contributions.append(int(required[slot] != 0 and returned != 2))
        old_ready = [sum(contributions[(lane // 16) * 16:(lane // 16 + 1) * 16]) == 0
                     for lane in range(32)]
        self.assertEqual(old_ready, [True] * 16 + [False] * 16)
        # With 32 distinct logical slots, padding has no dependencies. Its
        # masked atomic return cannot contribute to the full-warp reduction.
        contributions = [int(required[lane] != 0 and states[lane] != 2)
                         if lane < 16 else 0 for lane in range(32)]
        self.assertEqual(sum(contributions), 0)

    def test_normal_w6_entry_covers_all_physical_lanes(self):
        for tasks in (16, 64, 256):
            case = dict(workload="W6", strategy="S5", params=dict(dag="fork_join", depth=4,
                        tasks=tasks, workers=4, schedule="fixed", dtype="float16"))
            for enabled in (False, True):
                launch = captured_launch(case, enabled)
                task_count, padded = launch["arguments"][18:20]
                self.assertEqual(task_count, tasks)
                self.assertGreaterEqual(padded, task_count)
                self.assertGreaterEqual(padded, 32 * launch["options"]["num_warps"])
                self.assertEqual(padded & (padded - 1), 0)
                self.assertEqual(launch["grid"], (4,))

    def test_shared_w8_dispatch_also_gets_full_cta_padding(self):
        for batch in (1, 2):
            case = dict(workload="W8", strategy="S5", params=dict(batch=batch, seq=16, phase="prefill",
                        workers=4, cut=32, dtype="float16"))
            for enabled in (False, True):
                launch = captured_launch(case, enabled)
                self.assertEqual(launch["arguments"][18], 16 * batch)
                self.assertEqual(launch["arguments"][19], 128)
                self.assertEqual(launch["options"]["num_warps"], 4)

    def test_both_execution_plans_receive_the_frozen_projection_mode(self):
        case=dict(workload="W8",strategy="S5",params=dict(batch=1,seq=16,phase="prefill",
                  workers=2,cut=32,dtype="float16"))
        for enabled in (False,True):
            launch=captured_launch(case,enabled,qkv_bias_mode=6)
            self.assertEqual(launch["arguments"][-1],6)


if __name__ == "__main__":
    unittest.main()

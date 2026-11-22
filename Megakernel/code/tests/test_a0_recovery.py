"""CPU regressions for the stalled A0 supervisor and missing GPU telemetry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from stage_a.cli import parser
from stage_a.env import driver_cuda_version, resolve_gpu_selector
from stage_a.micro import items, run, run_process
from stage_a.report import generate
from stage_a.util import save


class TelemetryTests(unittest.TestCase):
    def test_torch_bare_uuid_is_valid_nvidia_smi_selector(self):
        bare = "bd9da0e3-1bfe-5016-c453-2eb587c5ca2c"
        self.assertEqual(resolve_gpu_selector(bare, 0, "3"), "GPU-" + bare)
        self.assertEqual(resolve_gpu_selector("GPU-" + bare, 0, "3"), "GPU-" + bare)
        self.assertEqual(resolve_gpu_selector("MIG-GPU-example/1/0", 0, "3"), "MIG-GPU-example/1/0")

    def test_fallback_obeys_visible_device_order(self):
        self.assertEqual(resolve_gpu_selector(None, 0, "3,1"), "3")
        self.assertEqual(resolve_gpu_selector(None, 1, "3,1"), "1")
        self.assertEqual(resolve_gpu_selector(None, 2, ""), "2")
        with self.assertRaises(ValueError):
            resolve_gpu_selector(None, 2, "3,1")

    def test_driver_615_umd_and_legacy_headers(self):
        self.assertEqual(driver_cuda_version("NVIDIA-SMI 615.71.10 KMD Version: 615.71.10 CUDA UMD Version: 13.4"), "13.4")
        self.assertEqual(driver_cuda_version("NVIDIA-SMI 570.0 Driver Version: 570.0 CUDA Version: 12.8"), "12.8")
        self.assertEqual(driver_cuda_version("UNAVAILABLE: missing nvidia-smi"), "UNAVAILABLE")


class MicroRecoveryTests(unittest.TestCase):
    def test_report_cannot_hide_failed_micro_with_other_measured_items(self):
        micro = [dict(family="H" + str(i), name="cpu-test", status="MEASURED") for i in range(1, 7)]
        micro.append(dict(family="H1", name="failed-cpu-test", status="FAILED", reason="regression"))
        with tempfile.TemporaryDirectory() as directory:
            report = generate(Path(directory), dict(errors=[], venv="cpu", prefix="cpu"),
                              dict(weights="QWEN_SHAPE_SYNTHETIC_WEIGHTS", qwen_status="READY"),
                              dict(environment_id="cpu-test", mode="smoke", filtered=False, batches=10),
                              [], [], micro)
        self.assertFalse(report["gates"]["microbench"])
        self.assertFalse(report["gates"]["core_failures_absent"])

    @unittest.skipUnless(os.name == "posix", "POSIX process groups require Linux/macOS")
    def test_timeout_also_kills_worker_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "descendant_must_not_write"
            descendant = "import pathlib,sys,time; time.sleep(1.5); pathlib.Path(sys.argv[1]).write_text('alive')"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(30)"
            result = run_process([sys.executable, "-c", parent, descendant, str(marker)],
                                 Path(directory) / "group.log", .8)
            self.assertEqual(result["status"], "TIMEOUT")
            time.sleep(1.2)
            self.assertFalse(marker.exists())

    def test_real_worker_timeout_terminates_before_future_write(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must_not_exist"
            log = Path(directory) / "worker.log"
            script = "import pathlib,sys,time; print('started',flush=True); time.sleep(1.2); pathlib.Path(sys.argv[1]).write_text('alive')"
            outcome = run_process([sys.executable, "-u", "-c", script, str(marker)], log, .6)
            self.assertEqual(outcome["status"], "TIMEOUT")
            self.assertLess(outcome["elapsed_seconds"], 5.)
            time.sleep(.8)
            self.assertFalse(marker.exists())
            self.assertIn("TIMEOUT", log.read_text(encoding="utf8"))

    def test_real_worker_failure_keeps_exit_code_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            outcome = run_process([sys.executable, "-c", "print('regression failure'); raise SystemExit(7)"], log, 5)
            self.assertEqual(outcome["status"], "FAILED")
            self.assertEqual(outcome["exit_code"], 7)
            self.assertIn("regression failure", log.read_text(encoding="utf8"))

    def test_timeout_is_saved_and_later_gpu_items_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("stage_a.micro.run_process", return_value=dict(status="TIMEOUT", elapsed_seconds=.1)) as worker:
                records = run("cuda:0", 1, True, 24, directory, .1)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(records[0]["status"], "TIMEOUT")
            self.assertTrue(records[0]["fatal_device"])
            self.assertTrue(all(row["status"] == "BLOCKED" for row in records[1:]))
            root = Path(directory)
            self.assertEqual(json.loads((root / "00_environment/microbench.json").read_text()), records)
            self.assertTrue(json.loads((root / "08_logs/micro/progress.json").read_text())["quarantine"])

    def test_completed_worker_without_new_result_cannot_reuse_stale_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "08_logs/micro/H1.gemm.result.json"
            save(target, [dict(family="H1", name="gemm", status="MEASURED")])
            with patch("stage_a.micro.run_process", return_value=dict(status="COMPLETED", elapsed_seconds=.1)):
                records = run("cuda:0", 1, True, 24, directory, only_item="H1.gemm")
            self.assertEqual(records[0]["status"], "FAILED")
            self.assertTrue(records[0]["fatal_device"])

    def test_per_item_results_survive_later_failure(self):
        def worker(command, log, timeout):
            request = json.loads(Path(command[-2]).read_text())
            item = request["item"]
            if item["id"] == "H1.exp":
                return dict(status="TIMEOUT", elapsed_seconds=.1)
            save(command[-1], [dict(family=item["family"], name=item["name"], status="MEASURED", samples_us=[1.])])
            return dict(status="COMPLETED", elapsed_seconds=.1)
        with tempfile.TemporaryDirectory() as directory:
            with patch("stage_a.micro.run_process", side_effect=worker):
                records = run("cuda:0", 1, True, 24, directory)
            self.assertEqual([row["status"] for row in records[:3]], ["MEASURED", "MEASURED", "TIMEOUT"])
            self.assertTrue(all(row["status"] == "BLOCKED" for row in records[3:]))

    def test_diagnostic_mode_and_timeout_options(self):
        args = parser().parse_args(["--mode", "micro", "--micro-item", "H6.ready", "--micro-timeout-seconds", "60"])
        self.assertEqual(args.micro_item, "H6.ready")
        self.assertEqual(args.micro_timeout_seconds, 60)
        self.assertEqual(len(items()), 20)
        self.assertEqual(len({item["id"] for item in items()}), 20)

    def test_invalid_timeout_is_rejected_before_any_worker(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                run("cuda:0", 1, True, 24, timeout_seconds=value)


if __name__ == "__main__":
    unittest.main()

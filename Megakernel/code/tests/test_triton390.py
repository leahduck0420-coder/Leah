import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import sys

from stage_a.assets import ROOT, TUTORIALS, assets
from stage_a.numeric import configure_matmul_precision
from stage_a.optional import tutorial_nodes
from stage_a.toolchain import TRITON_VERSION, UPSTREAM_COMMIT, matches_triton, require_triton, toolkit_release
from stage_a.util import file_hash
from stage_b.importer import audit
from stage_b.runner import admissible_cache


class Triton390Tests(TestCase):
    def test_actual_import_gate(self):
        for version in ("3.9.0", "3.9.0+seam"):
            with patch.dict(sys.modules,triton=SimpleNamespace(__version__=version,__file__="/selected/triton")):
                self.assertEqual(require_triton(),version)
        for version in (None,"3.4.0","3.9.0rc1","3.9.1","3.9.0.post1"):
            self.assertFalse(matches_triton(version))
            with patch.dict(sys.modules,triton=SimpleNamespace(__version__=version,__file__="/selected/triton")):
                with self.assertRaisesRegex(RuntimeError,"Expected Triton"):require_triton()

    def test_nvcc_banner_and_driver_separated(self):
        self.assertEqual(toolkit_release("Cuda compilation tools, release 13.4, V13.4.90"),(13,4))
        self.assertEqual(toolkit_release("Cuda compilation tools, release 12.8, V12.8.93"),(12,8))
        self.assertEqual(toolkit_release("CUDA UMD Version: 13.4"),())

    def test_old_input_rejected_before_search(self):
        from stage_b import importer
        for protocol,env in (({"required_triton":"3.4.0"},{"triton":"3.9.0"}),({"required_triton":"3.9.0"},{"triton":"3.4.0"})):
            with patch.object(importer,"discover",return_value=ROOT),patch.object(importer,"resolve_files",return_value={k:k for k in importer.REQUIRED}),patch.object(importer,"read",side_effect=[{},protocol,env,{}]):
                with self.assertRaisesRegex(ValueError,"collected with Triton 3.9.0"):audit(ROOT)
        self.assertFalse(admissible_cache(dict(identity="cpu",status="NO_OP",triton_version="3.4.0"),"cpu",ROOT,12))

    def test_new_torch_matmul_precision(self):
        backend=SimpleNamespace(fp32_precision="tf32",allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,allow_fp16_accumulation=True)
        torch=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)))
        settings=configure_matmul_precision(torch)
        self.assertEqual(settings["fp32_precision"],"ieee")
        self.assertFalse(settings["allow_fp16_accumulation"])
        self.assertFalse(hasattr(backend,"allow_tf32"))

    def test_optional_tutorial_hashes_and_exports(self):
        pins=json.loads((ROOT/"configs/tutorials_390.json").read_text(encoding="utf8"))
        self.assertEqual((pins["revision"],pins["triton"]),(UPSTREAM_COMMIT,TRITON_VERSION))
        with TemporaryDirectory() as tmp:
            wrong=Path(tmp)/"triton"/TUTORIALS[0]
            wrong.parent.mkdir()
            wrong.write_text("# different release",encoding="utf8")
            paths=assets(tmp)["tutorials"]
            self.assertNotEqual(str(wrong),paths[TUTORIALS[0]])
            for n in TUTORIALS:
                path=Path(paths[n])
                self.assertEqual(file_hash(path),pins["sha256"][n])
                nodes=tutorial_nodes(n,ast.parse(path.read_bytes()))
                compile(ast.Module(nodes,[]),str(path),"exec")
                self.assertFalse(any(isinstance(x,ast.FunctionDef) and x.name.startswith(("test_","bench")) for x in nodes))
                if n.startswith("06"):
                    self.assertEqual(ast.unparse(nodes[-1]),"attention = _attention.apply")
                    self.assertTrue(any(isinstance(x,ast.FunctionDef) and x.name=="prune_invalid_configs" for x in nodes))

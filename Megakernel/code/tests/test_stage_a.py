import ast
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from stage_a import assets as offline
from stage_a.cli import parser,cached
from stage_a.compiler import tile_map,tiled_tensor,plan,graph,levels,order_constraints,check_slices,slices
from stage_a.evidence import statistics_for,features
from stage_a.legality import regression
from stage_a.matrix import anchors as matrix,SUPPORT,TARGET
from stage_a.report import generate,valid
from stage_a.util import Rejected,save,file_hash


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke=matrix(20261119,True)
    def test_w1_to_w8(self):
        self.assertEqual({(x["workload"],x["strategy"]) for x in self.smoke},{(w,s) for w in SUPPORT for s in SUPPORT[w]})
        for c in self.smoke:
            a,b=plan(c,False),plan(c,True)
            self.assertEqual(a.strategy,"none")
            self.assertEqual(b.strategy,c["strategy"])
            self.assertLessEqual(sum([b.local,b.stats,bool(b.cut),b.merge>1,b.scheduling=="ready"]),1)
    def test_all_35_cpu_checks(self):
        checks=regression()
        self.assertEqual({r["strategy"] for r in checks if r["expected_reject"]},set(TARGET))
        self.assertTrue(all(r["passed"] for r in checks))
    def test_masked_multidimensional_ownership(self):
        m=tiled_tensor((19,47),(8,16),"bf16")
        for row in range(19):
            for col in range(47):
                self.assertEqual(m.owners((row,col)),((row//8,col//16),))
        self.assertEqual(len(m.block_of((2,2))),45)
    def test_s1_versions_and_permutation(self):
        c=copy.deepcopy(self.smoke[0]);c["params"]["layout"]="cross_program"
        with self.assertRaises(Rejected):plan(c,True)
        c["params"]["layout"]="lane";self.assertEqual(plan(c,True).proof["local_xor_offset"],1)
    def test_s2_partial_tile(self):
        c=next(copy.deepcopy(c) for c in self.smoke if c["workload"]=="W2" and c["strategy"]=="S2")
        c["params"].update(n=515,merge=4,bn=32)
        self.assertEqual(plan(c,True).ownership["grid"][-1],5)
        c["params"]["merge"]=1
        with self.assertRaisesRegex(Rejected,"occupied"):plan(c,True)
    def test_s3_numerical_identities(self):
        import math,random
        x=[random.Random(j).uniform(-4,4) for j in range(37)];peak=max(x);den=sum(math.exp(v-peak) for v in x)
        s=[math.exp(z-peak)/den for z in x]
        self.assertAlmostEqual(sum(s),1)
        for actual,z in zip(s,x):self.assertAlmostEqual(actual,math.exp(z)/sum(math.exp(t) for t in x))
        for c in self.smoke:
            if c["strategy"]=="S3":self.assertIn("statistics",plan(c,True).proof)
    def test_s4_tail_folding(self):
        for width in (1,2,8,32):
            parts=slices(37,width);check_slices(parts,37)
            self.assertEqual([v for a,b in parts for v in range(a,b)],list(range(37)))
        with self.assertRaises(Rejected):check_slices([(0,3),(2,7)],7)

    def test_online_normalizer_with_causal_tail(self):
        import math
        cfg=next(c for c in self.smoke if c["workload"]=="W5")
        self.assertEqual(cfg["params"]["normalizer"],"online")
        for last in (0,31,32,84,132):
            values=[(.75*i-30 if i<=last else -math.inf) for i in range(133)]
            peak,den=-math.inf,0.
            for k in range(0,len(values),32):
                chunk=values[k:k+32]
                updated=max(peak,max(chunk))
                den=den*math.exp(peak-updated)+sum(math.exp(x-updated) for x in chunk)
                peak=updated
            reference=math.log(sum(math.exp(x) for x in values))
            self.assertAlmostEqual(peak+math.log(den),reference,places=12)
    def test_s5_ready_execution(self):
        for c in self.smoke:
            if c["strategy"]!="S5":continue
            actual=plan(c,True).ddg
            self.assertLess(actual["edge_count"],plan(c,False).ddg["edge_count"])
            done=set();order=[]
            while len(done)<len(actual["data"]):
                ready=[i for i,d in enumerate(actual["data"]) if i not in done and all(k in done for k in d)]
                self.assertTrue(ready);j=ready[-1];done.add(j);order.append(j)
            self.assertEqual(len(set(order)),len(actual["data"]))
    def test_full_stratified(self):
        a=matrix(20261119);self.assertEqual(a,matrix(20261119));self.assertEqual(len({x["id"] for x in a}),len(a))
        self.assertTrue(310<=len(a)<=500)
        counts={k:0 for k in TARGET}
        for c in a:
            try:plan(c,True)
            except Rejected:continue
            counts[c["strategy"]]+=1
        self.assertEqual(counts,TARGET)


class EngineeringTests(unittest.TestCase):
    def test_profiler_guard(self):
        r=subprocess.run([sys.executable,"-c","from stage_a.offline import activate;activate();import subprocess;subprocess.run(['ncu','--version'])"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn("disabled external executable",r.stderr)
    def test_cli(self):
        for mode in ("full","smoke"):
            a=parser().parse_args(["--mode",mode,"--venv-path","/opt/SEAM/.venv","--resume","--only-strategy","S2"])
            self.assertEqual(a.device,"cuda:0");self.assertTrue(a.resume);self.assertEqual(a.venv_path,Path("/opt/SEAM/.venv"))
    def test_wrong_prefix(self):
        r=subprocess.run([sys.executable,"-m","stage_a.cli","--mode","smoke","--venv-path","/invalid/seam-test"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn("venv",r.stderr)
    def test_offline_default_and_missing_real(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(offline.assets(d)["qwen_status"],"READY")
            self.assertEqual(offline.assets(d,real=True)["qwen_status"],"BLOCKED")
            self.assertTrue(all(offline.assets(d)["tutorials"].values()))
    def test_bad_local_config(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"qwen/config/config.json";save(p,{"hidden_size":1})
            self.assertEqual(offline.assets(d)["qwen_status"],"BLOCKED")
    def test_denied_network(self):
        r=subprocess.run([sys.executable,"-c","from stage_a.offline import activate;activate();import socket;socket.getaddrinfo('localhost',0)"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn("offline policy",r.stderr)
    def test_stats(self):
        a=statistics_for([1.,2.,3.,4.],12);self.assertEqual(a["median"],2.5);self.assertEqual(a["batches"],4)
        self.assertLess(a["ci95"][0],a["ci95"][1])
    def test_cost_provenance(self):
        for c in matrix(5,True):
            vals,src=features(c,plan(c,True),dict(ctas=1,registers=None,smem=None,spill=None,theoretical_occupancy="UNAVAILABLE"))
            self.assertEqual(vals["achieved_occupancy"],"UNAVAILABLE")
            self.assertEqual(src["smem"],"UNAVAILABLE")
            self.assertEqual(src["nominal_main_arithmetic_ops"],"STATIC_ESTIMATE")
    def test_resume_signature(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"candidate.json";case=dict(id="unit")
            save(f,dict(id="unit",signature="a",status="NO_OP"))
            self.assertIsNone(cached(case,f,"a",Path(d)))  # Rejections require measured fallback evidence.
            self.assertIsNone(cached(case,f,"b",Path(d)))
            save(f,dict(id="unit",signature="a",status="PASS"))
            self.assertIsNone(cached(case,f,"a",Path(d)))
    def test_blocked_report_has_no_labels(self):
        with tempfile.TemporaryDirectory() as d:
            c=matrix(5,True)[0];root=Path(d)
            res=generate(root,dict(errors=["CPU-only test"],venv="x",prefix="x"),
                         dict(weights="QWEN_SHAPE_SYNTHETIC_WEIGHTS",qwen_status="READY"),
                         dict(environment_id="cpu-test",mode="smoke",filtered=False,batches=10),
                         [dict(**c,status="BLOCKED",reason="test environment")],regression(),[])
            self.assertEqual(res["verdict"],"FAIL");self.assertEqual(res["eligible_measured"],0)
            self.assertTrue((root/"07_dataset/stage_a_dataset_schema.json").exists())
            self.assertEqual(len((root/"02_performance/raw_timings.csv").read_text(encoding="utf-8-sig").splitlines()),1)
    def test_no_legacy_strategy_names(self):
        import re
        for file in (offline.ROOT/"stage_a").glob("*.py"):
            self.assertIsNone(re.search(r'["\']C[0-3]["\']',file.read_text(encoding="utf8")))

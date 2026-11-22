"""Audit the new paired design, lowerings and A-to-B handoff without GPU samples."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_pipeline as pipeline
from stage_a.all_pairs import PROTOCOL
from stage_a.compiler import check_slices, graph, plan
from stage_a.matrix import anchors, matrix, STRATEGIES
from stage_a.ownership import OwnershipMap
from stage_a.util import Rejected, digest, save
from stage_b.importer import read
from stage_b.planner import derive, inventory, original_action
from test_stage_b import act, case


class DesignTests(unittest.TestCase):
    def test_server_anchor_identity_and_all_five(self):
        old = anchors(20261119)
        self.assertEqual(digest(old), "ff185cc1ce67dde6e369443dedb1731ab3a4a44f67cfe679876c0f8443da3f22")
        requests = matrix(20261119)
        self.assertEqual(len(old), 400)
        self.assertEqual(len({r["id"] for r in requests}), 2000)
        for i, anchor in enumerate(old):
            group = requests[5*i:5*i+5]
            self.assertEqual([r["strategy"] for r in group], list(STRATEGIES))
            self.assertEqual({r["anchor_id"] for r in group}, {anchor["id"]})
            for r in group:
                if r["strategy"] == anchor["strategy"]:
                    self.assertEqual(r["id"], anchor["id"])
                    self.assertEqual(r["params"], anchor["params"])
                for k, v in r["params"].items():
                    if k != {"S2":"merge","S4":"cut"}.get(r["strategy"]):
                        self.assertEqual(v, anchor["params"][k])

    def test_all_admitted_actions_have_joint_backend(self):
        counts = defaultdict(Counter)
        for r in matrix(20261119):
            group = counts[r["workload"], r["strategy"]]
            try:
                single = plan(r, True)
            except Rejected as e:
                group[e.status] += 1
                continue
            self.assertEqual(single.strategy, r["strategy"])
            joint = derive(r, (original_action(r),))
            self.assertIn(r["strategy"], joint.proof)
            group["STATIC_ADMITTED"] += 1
        spec = read(Path(__file__).resolve().parents[1]/"validation/default_full_design.json")
        self.assertEqual(len(counts), 40)
        self.assertEqual(sum(bool(v["STATIC_ADMITTED"]) for v in counts.values()), 29)
        for cell in spec["cells"]:
            self.assertEqual(cell["statuses"], dict(counts[cell["workload"], cell["strategy"]]))

    def test_smoke_attempts_each_pair(self):
        m = matrix(11, True)
        self.assertEqual(len(m), 85)
        self.assertEqual(len({(x["workload"], x["strategy"]) for x in m}), 40)

    def test_grouped_dag_exact_original_dependencies_and_tail(self):
        for dag in ("mixed","chain","fork_join","fan_in"):
            c = case("W6", dag=dag, tasks=17)
            q = derive(c, (act(c,"task_groups",4),))
            old = graph(c["params"],"W6")
            expected = [set() for _ in range(5)]
            for child, parents in enumerate(old):
                for parent in parents:
                    self.assertLess(parent,child)
                    if child//4 != parent//4:
                        expected[child//4].add(parent//4)
            self.assertEqual(old, q.ddg["original_data"])
            self.assertEqual(expected, [set(p) for p in q.ddg["data"]])
            self.assertEqual(OwnershipMap.from_json(q.ownership).owners((16,31)), ((4,0),))

    def test_new_sites_and_exclusive_probability_transforms(self):
        for w,site in [("W2","dot_reduction"),("W3","norm_reduction"),("W7","down_reduction")]:
            c = case(w)
            q = derive(c, (act(c,site,64),))
            check_slices(q.proof["S4"]["partition"], c["params"]["n" if w=="W7" else "k"])
        for w in ("W4","W5","W8"):
            c=case(w)
            s1=act(c,"probability_handoff")
            self.assertTrue(derive(c,(s1,)).local)
            with self.assertRaisesRegex(Rejected,"materialization"):
                derive(c,(s1,act(c,"probability_statistics")))

    def test_unavailable_pairs_cannot_become_search_actions(self):
        for w,s in (("W1","S3"),("W1","S4"),("W1","S5"),("W2","S3"),("W2","S5"),("W3","S5"),("W4","S5"),("W5","S5"),("W6","S1"),("W6","S3"),("W7","S5")):
            c=dict(case(w),strategy=s)
            self.assertFalse(any(a.strategy==s for a in inventory(c)))
            with self.assertRaises(Rejected) as error:plan(c,True)
            self.assertEqual(error.exception.status, "NOT_APPLICABLE")
            self.assertEqual(plan(c,False).strategy,"none")


class PipelineTests(unittest.TestCase):
    def test_old_stage_a_is_never_promoted_on_subprocess_failure(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/"stage_a"
            save(a/"pipeline_completion.json", {"invocation":"past"})
            save(a/"06_reports/acceptance_report.json", {"verdict":"PASS"})
            with patch.object(pipeline.subprocess,"run",return_value=SimpleNamespace(returncode=1)) as invoked:
                self.assertEqual(pipeline.stages(root,pipeline.options([]),"signature"),2)
            self.assertEqual(invoked.call_count,1)
            self.assertEqual(read(root/"pipeline.json")["stage_b_verdict"],"BLOCKED")

    def test_partial_stage_a_can_feed_only_its_new_output_to_stage_b(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            opt=pipeline.options(["--mode","full"])
            called=[]
            def run(args,*,cwd,env):
                dest=Path(args[args.index("--output-dir")+1])
                save(dest/"pipeline_completion.json",{"invocation":env["SEAM_AB_INVOCATION"]})
                if "stage_a.cli" in args:
                    save(dest/"00_environment/protocol.json",dict(code="current",mode="full",design=PROTOCOL))
                    save(dest/"00_environment/microbench.json",[])
                    save(dest/"06_reports/acceptance_report.json",dict(verdict="PARTIAL",environment_errors=[],
                         eligible_measured=1,missing_pairs=[],gates=dict(all_40_requested=True,core_failures_absent=True,all_requested_accounted=True,plots=True)))
                    called.append("a")
                    return SimpleNamespace(returncode=1)
                self.assertEqual(Path(args[args.index("--stage-a-results")+1]), root/"stage_a")
                self.assertEqual(dest,root/"stage_b"/("f"*16))
                save(dest/"08_reports/acceptance_report.json",dict(verdict="PASS"))
                called.append("b")
                return SimpleNamespace(returncode=0)
            with patch.object(pipeline.subprocess,"run",side_effect=run), patch.object(pipeline,"handoff",return_value="f"*64),patch.object(pipeline,"fingerprint",return_value="current"):
                self.assertEqual(pipeline.stages(root,opt,"same"),1)
                opt.resume=True
                self.assertEqual(pipeline.stages(root,opt,"same"),1)
            self.assertEqual(called,["a","b","b"])
            self.assertEqual(read(root/"pipeline.json")["overall"],"PARTIAL")

    def test_incomplete_path_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):pipeline.handoff(Path(tmp))

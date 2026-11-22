"""CPU regression suite. Search callbacks use abstract objective values only."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from stage_a.compiler import check_slices, levels
from stage_a.ownership import OwnershipMap
from stage_a.util import Rejected, digest, save
from stage_b.cli import parser
from stage_b.importer import audit, contract, discover, select_seeds
from stage_b.planner import Action, derive, inventory, key, original_action
from stage_b.report import Dirs, Results, COLUMNS, interaction_us
from stage_b.runner import admissible_cache
from stage_b.search import Pool, finite_oracle, select

ROOT = Path(__file__).resolve().parents[1]

def case(w, **params):
    defaults = {
        'W1': ('S1',dict(n=259,tile=128,layout='lane',consumers=2)),
        'W2': ('S1',dict(m=65,n=515,k=512,bm=32,bn=32,bk=32,merge=1,warps=4)),
        'W3': ('S3',dict(m=33,k=259,op='layer',consumer='linear',consumers=2,merge=2)),
        'W4': ('S4',dict(m=3,k=517,cut=128,consumer='weighted',consumers=2)),
        'W5': ('S3',dict(batch=1,seq=133,cut=32,normalizer='online',causal=True,d=64)),
        'W6': ('S5',dict(tasks=16,workers=2,depth=4,dag='fork_join',schedule='fixed')),
        'W7': ('S1',dict(m=33,batch=2,k=896,n=4864,bn=32,merge=2)),
        'W8': ('S3',dict(batch=2,seq=133,phase='decode',cut=32,workers=2,d=64)),
    }
    s,p = defaults[w]
    return dict(id=w+'-CPU',workload=w,strategy=s,params=dict(p,dtype="float16")|params,
                stage_a_run_id="cpu-structural-test",input_seed=7)

def act(c,loc,param=None):
    return next(a for a in inventory(c) if a.location==loc and (param is None or a.parameter==param))

class PlanTests(unittest.TestCase):
    def test_registered_domains(self):
        for i in range(1,9):
            c=case('W'+str(i))
            self.assertTrue(inventory(c))
            self.assertIn(original_action(c), inventory(c))
            self.assertEqual(OwnershipMap.from_json(derive(c,()).ownership).copies(),1)

    def test_nonlocal_handoff_rejected(self):
        c=case('W1',layout='cross_program')
        with self.assertRaisesRegex(Rejected,'nonlocal'):derive(c,(act(c,'bias_to_gelu'),))

    def test_prefix_reproof(self):
        c=case('W3');stats=act(c,'norm_statistics');reb=act(c,'stats_to_rebuild');local=act(c,'norm_to_consumer')
        with self.assertRaisesRegex(Rejected,'no defining'):derive(c,(reb,))
        plain=derive(c,(local,));retarget=derive(c,(local,stats))
        self.assertNotEqual(plain.proof['S1']['value'],retarget.proof['S1']['value'])
        self.assertEqual(key(c,retarget),key(c,derive(c,(stats,reb))))
        with self.assertRaisesRegex(Rejected,'same statistics'):derive(c,(stats,reb,local))

    def test_scope_order_affects_executable(self):
        for w,site in [('W4','weighted_reduction'),('W5','pv_reduction'),('W8','pv_reduction')]:
            c=case(w);s=act(c,'probability_statistics');k=act(c,site,64)
            forward=derive(c,(s,k));reverse=derive(c,(k,s))
            self.assertNotEqual(key(c,forward),key(c,reverse))
            self.assertEqual(forward.ddg["statistic_read_scope"],"row")
            self.assertEqual(reverse.ddg["statistic_read_scope"],"slice")
            check_slices(forward.proof["S4"]["partition"],c['params'].get('k',c['params'].get('seq')))

    def test_partial_tiles(self):
        c=case('W2');p=derive(c,(act(c,'output_columns',4),))
        lm=OwnershipMap.from_json(p.ownership)
        for m in (0,31,32,64):
            for n in (0,31,32,127,128,514):
                self.assertEqual(lm.owners((m,n)),((m//32,n//128),))

    def test_independent_sites_and_alias(self):
        c=case('W7')
        pair=(act(c,'gate_to_swiglu'),act(c,'up_to_swiglu'))
        self.assertEqual(key(c,derive(c,pair)),key(c,derive(c,(act(c,'both_to_swiglu'),))))
        self.assertNotEqual(key(c,derive(c,pair[:1])),key(c,derive(c,pair[1:])))
        exe=derive(c,pair+(act(c,'mlp_rows',2),act(c,'rms_statistics')))
        self.assertTrue(exe.stats and exe.merge==2 and exe.local==3)

    def test_grouped_attention_full_domain(self):
        for phase in ('prefill','decode'):
            c=case('W8',phase=phase);base=derive(c,())
            p=derive(c,(act(c,'head_groups',4),act(c,'pv_reduction',64),act(c,'task_order',4)))
            lm=OwnershipMap.from_json(p.ownership)
            self.assertEqual(tuple(base.ownership["shape"]),lm.shape)
            self.assertEqual(lm.owners((1,13,lm.shape[-1]-1)),((1,3,0),))
            self.assertEqual(len(p.ddg["data"]),12)
            self.assertEqual(p.ddg['data'][-1],[7,8,9,10])
            self.assertEqual(max(levels(p.ddg["execution"])),3)
            for d,x in zip(p.ddg['data'],p.ddg['execution']):self.assertTrue(set(d)<=set(x))

    def test_unavailable_schedule_and_single_slice(self):
        c=case('W5',seq=32)
        self.assertFalse(any(a.strategy=='S5' for a in inventory(c)))
        with self.assertRaisesRegex(Rejected,'single active slice'):derive(c,(act(c,'pv_reduction',32),))
        c=case('W6',dag='chain')
        with self.assertRaisesRegex(Rejected,'no scheduling'):derive(c,(act(c,'task_order'),))

    def test_contract_matches_controls_only(self):
        self.assertEqual(contract(case('W2')),contract(case('W2',bn=128,merge=4)))
        self.assertNotEqual(contract(case('W5')),contract(case('W5',causal=False)))

class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool=Pool(case('W7'),4)
    def objective(self,h):
        # Unit test of traversal and accounting; no timing source or output dataset.
        return dict(status="PASS",latency_us=1+int(h[:4],16),compile_attempts=6,
                    compiled_specializations=2,compile_seconds=0.1,evaluation_seconds=0.2)

    def test_budget_and_reached_proofs(self):
        for method in ('none','single','fixed','greedy','beam'):
            for guided in (True,False):
                paid=[]
                def query(h):
                    self.assertNotIn(h,paid);paid.append(h);return self.objective(h)
                p=self.pool
                s=select(p,method,query,20,4,4,guided=guided)
                self.assertEqual(s['evaluated_candidates'],len(paid))
                self.assertLessEqual(len(paid),20)
                self.assertEqual(s['compiled_candidates'],len(paid))
                self.assertEqual(s['compile_api_calls'],6*len(paid))
                for row in s['trace']:
                    history=tuple(Action(**a) for a in row['sequence'])
                    self.assertEqual(p.hash_path(history),row['plan'])
                if method=='single':self.assertEqual(len(s['selected_sequence']),1)

    def test_order_ablation(self):
        s=select(self.pool,"beam",self.objective,50,8,4,fixed_order=True)
        for row in s["trace"]:
            order=[a["strategy"] for a in row["sequence"]]
            self.assertEqual(order,sorted(order))

    def test_oracle_admission(self):
        p=self.pool
        self.assertIsNone(finite_oracle(p,{}))
        records={h:self.objective(h) for h in p.nodes}
        fastest=min(records,key=lambda h:records[h]['latency_us'])
        self.assertEqual(finite_oracle(p,records),fastest)
        records[fastest]['status']='NO_OP'
        self.assertNotEqual(finite_oracle(p,records),fastest)

    def test_aliases_share_one_plan(self):
        p=self.pool
        self.assertGreater(len(p.paths),len(p.nodes))
        for seq,_,h in p.paths:
            self.assertEqual(p.hash_path(seq),h)
        self.assertTrue(any(len(s)==4 for s,_,_ in p.paths))

class IntegrationTests(unittest.TestCase):
    def test_cli_and_scopes(self):
        o=parser().parse_args(['--mode','full','--only-combination','S1+S2','--resume'])
        self.assertEqual((o.beam_width,o.search_budget,o.max_depth),(16,100,4))
        self.assertTrue(o.resume)
        p=Pool(case('W3'),3,('S1','S3'))
        self.assertFalse(any(a.strategy=='S2' for a in p.actions))

    def test_empty_and_ambiguous_import(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            with self.assertRaises(ValueError):discover(p)
            for i in ('a','b'):save(p/i/'07_dataset/dataset_manifest.json',{})
            with self.assertRaises(ValueError):discover(p)

    def test_offline_guard_in_child(self):
        code="from stage_a.offline import activate; import socket; activate(); socket.getaddrinfo('localhost',80)"
        proc=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn("DNS disabled",proc.stderr)

    def test_factorial_formula(self):
        self.assertAlmostEqual(interaction_us(10,8,5,4),0.)
        self.assertGreater(interaction_us(10,8,5,3),0.)
        self.assertLess(interaction_us(10,8,5,6),0.)
        self.assertEqual(interaction_us(0,8,5,6),"UNAVAILABLE")

    def test_invalid_resume(self):
        self.assertFalse(admissible_cache(dict(status="PASS",identity="x"),"x",ROOT,10))
        self.assertFalse(admissible_cache(dict(status="NO_OP",identity="y"),"x",ROOT,10))
        self.assertFalse(admissible_cache(dict(status="TIMEOUT",identity="x"),"x",ROOT,10))
        self.assertTrue(admissible_cache(dict(status="INVALID_REJECTED",identity="x",triton_version="3.9.0"),"x",ROOT,10))

    def test_missing_measurements_do_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in Dirs:(root/name).mkdir()
            c=case('W4')
            proto=dict(assets={'weights':'QWEN_SHAPE_SYNTHETIC_WEIGHTS'},signature='structural-test',options=dict(mode='smoke',max_depth=3))
            s=Results(root,dict(stage_a_dataset_schema=['case_id'],trusted=0,issues=['test missing inputs']),proto)
            p=Pool(c,2);s.structure(p);s.blocked(p,dict(status='BLOCKED',reason='CPU structural test'))
            verdict=s.finish({'errors':['no target device']},{'qwen_status':'READY'},None)
            self.assertEqual(verdict['verdict'],'FAIL')
            self.assertEqual(verdict['qualified'],0)
            self.assertEqual([f['points'] for f in verdict['figures']],[0]*8,verdict['plot_error'])
            with (root/'09_dataset/stage_b_dataset.csv').open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(p.nodes))
            self.assertTrue(set(COLUMNS)<=rows[0].keys())
            self.assertFalse(any(r['measurement_status']=='MEASURED' for r in rows))
            self.assertTrue((root/'03_multistep/target_coverage.csv').is_file())

    @unittest.skipUnless(os.environ.get('STAGE_A_TEST_RESULTS'),'requires actual archived Stage A observations')
    def test_positive_report_with_recorded_single(self):
        # Exercise the reporting branch with archived single measurements only.
        # No multistrategy time is generated or inferred.
        root=discover(os.environ['STAGE_A_TEST_RESULTS'])
        src=next(r for f in sorted((root/'08_logs/cases').glob('W1-*.json')) for r in
                 [json.loads(f.read_text(encoding='utf8'))] if r['status']=='PASS' and r['strategy']=='S1')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            for name in Dirs:(out/name).mkdir()
            protocol=dict(signature='unit-recorded-stage-a',assets={'weights':'QWEN_SHAPE_SYNTHETIC_WEIGHTS'},options=dict(mode='smoke',max_depth=2))
            st=Results(out,dict(trusted=1,issues=[],stage_a_dataset_schema=['case_id']),protocol)
            seed={**case('W1'),**{k:src[k] for k in ('params','id','input_seed','strategy')},'latency':src['statistics']['B2']['median']}
            pool=Pool(seed,2,("S1",));st.structure(pool)
            recs={}
            for h,seq in pool.nodes.items():
                sub='B2' if seq else 'B1'
                artifacts=[dict(meta=src['metadata']['B1'])]+([dict(meta=src['metadata']['B2'])] if seq else [])
                recs[h]=dict(status='PASS',raw_us=src['raw'],latency_us=src['statistics'][sub]['median'],
                             statistics={**src['statistics'],'candidate':src['statistics'][sub]},prefix_artifacts=artifacts,
                             paired_speedup_ci95=src['paired_speedup_ci95'] if seq else [1,1],
                             cost_by_strategy={'S1':dict(values=src['cost'],sources=src['sources'])} if seq else {})
            subject=next(h for h in pool.nodes if h!=pool.root)
            st.region(pool,recs,[(seed,recs[subject])],[select(pool,'single',recs.__getitem__,10,1,1)])
            self.assertTrue(math.isclose(st.data[-1]['incremental_speedup'],src['incremental_speedup']))
            self.assertEqual(st.data[-1]['register_delta'],src['metadata']['B2']['registers']-src['metadata']['B1']['registers'])
            v=st.finish({'errors':[]},{'qwen_status':'READY'},None)
            self.assertEqual(v['verdict'],'PARTIAL')
            self.assertEqual(v['qualified'],2)
            self.assertIsNone(v['plot_error'])

    @unittest.skipUnless(os.environ.get('STAGE_A_TEST_RESULTS'),'optional external archived Stage A input')
    def test_real_archive(self):
        summary,seeds=audit(Path(os.environ['STAGE_A_TEST_RESULTS']))
        self.assertFalse(summary['issues'])
        self.assertGreater(len(seeds),0)
        self.assertEqual(len(seeds),summary['trusted'])
        self.assertEqual({s['workload'] for s in select_seeds(seeds,True)},{'W'+str(i) for i in range(1,9)})
        for c in seeds:self.assertIn(original_action(c),inventory(c))

if __name__=='__main__':
    unittest.main()

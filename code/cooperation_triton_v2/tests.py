"""CPU-only policy, partition, suite and report-contract tests."""
import json
import tempfile
from pathlib import Path
import unittest
from .specs import Case,CONFIGS,config_id,suite
from .policy import (Dispatcher,Selector,candidates,eligibility,fit,geometry,
                     guard_decision,predict,regions,select_discovery)


class GeometryTests(unittest.TestCase):
    def test_partition(self):
        for m,n in ((1,1),(17,19),(129,1025)):
            for _,bm,bn,_ in CONFIGS:
                self.assertEqual(sum(a*b*c for a,b,c in regions(m,n,bm,bn)),m*n)

    def test_slot_conservation(self):
        for dims in ((1,4096,4096),(17,19,33),(129,1025,1027)):
            for cfg in CONFIGS:
                g=geometry(*dims,cfg)
                self.assertEqual(g['tc_slots']+g['simt_slots']-g['masked_slots'],g['useful_macs'])

    def test_hybrid_k_only_handles_k_tail(self):
        g=geometry(129,131,1025,('hybrid_k',64,128,64))
        self.assertEqual(g['useful_simt_macs'],129*131)
        self.assertEqual(g['tc_slots']+g['simt_slots']-g['masked_slots'],129*131*1025)


class GuardTests(unittest.TestCase):
    def test_narrow_full_simt_is_forbidden(self):
        c=Case('qkt',1,4096,4096,batch=4)
        for cfg in CONFIGS:
            if cfg[0] in ('hybrid','hybrid_k'):
                self.assertFalse(eligibility(c,cfg)[0])

    def test_small_k_tail_allowed(self):
        c=Case('gemm',128,1024,1025)
        self.assertTrue(eligibility(c,('hybrid_k',64,128,64))[0])

    def test_guard_requires_both_baselines(self):
        route,ok=guard_decision({'inductor_default':[100,101,99],
                                 'triton_padding_bound':[95,96,94],
                                 'optimized_dispatch':[80,81,79]})
        self.assertEqual((route,ok),('optimized_dispatch',True))
        route,ok=guard_decision({'inductor_default':[90,91,89],
                                 'triton_padding_bound':[95,96,94],
                                 'optimized_dispatch':[92,93,91]})
        self.assertEqual((route,ok),('inductor_default',False))

    def test_discovery_margin_prefers_padding(self):
        c=Case('gemm',129,129,129);allowed,_=candidates(c)
        values={config_id(x):[100.0] for x in allowed}
        values[config_id(next(x for x in allowed if x[0]=='small_mma'))]=[98.0]
        selected,pad=select_discovery(values,allowed)
        self.assertEqual(selected[0],'padding')


class ModelTests(unittest.TestCase):
    def test_local_model_and_ood(self):
        a=Case('gemm',128,128,129);b=Case('gemm',129,128,129)
        cfg=('small_mma',64,64,32)
        model=fit([(a,cfg,10.0)],'id')
        self.assertIsNotNone(predict(b,cfg,model))
        far=Case('gemm',4096,4096,4097)
        self.assertIsNone(predict(far,cfg,model))
        with self.assertRaises(ValueError):Selector('different',model)

    def test_dispatch_key_is_cached_tuple(self):
        c=Case('linear',129,768,769,ep='bias')
        self.assertIs(c.dispatch_key,c.dispatch_key)
        d=Dispatcher();d.register(c,lambda:42)
        self.assertEqual(d(c.dispatch_key),42)


class SuiteTests(unittest.TestCase):
    def test_calibration_disjoint(self):
        self.assertFalse({x.key for x in suite('calibration')} & {x.key for x in suite('targeted')})

    def test_targeted_has_parent_and_all_tail_kinds(self):
        cases=suite('targeted');kinds={x.kind for x in cases}
        self.assertTrue({'parent_aligned_control','m_tail','n_tail','k_tail_1','k_tail_3','mixed_tail','negative_control'}<=kinds)
        self.assertEqual(len(cases),len({x.key for x in cases}))


class ReportTests(unittest.TestCase):
    def test_server_side_python_plots(self):
        from .report import report
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'cases').mkdir()
            (root/'manifest.json').write_text(json.dumps({'arguments':{'blocks':5}}))
            (root/'failures.json').write_text('[]')
            case=dict(op='gemm',domain='general_tail',kind='k_tail_1',pair='p',m=129,n=128,k=129,
                      batch=1,tokens=[],ep='none')
            methods={
              'inductor_default':{'compiler':'inductor'},
              'triton_padding_autotune':{'audit':{'launch_count':1,'batched':False}},
              'triton_padding_bound':{'audit':{'launch_count':1,'batched':False}},
              'model_bound':{'config':['padding',32,32,32],'source':'model'},
              'optimized_bound':{'config':['small_mma',32,32,32],'source':'table'},
              'optimized_dispatch':{'config':['small_mma',32,32,32],'source':'table'},
              'optimized_guarded':{'route':'optimized_dispatch','selected_config':['small_mma',32,32,32]},
            }
            lat={'inductor_default':100,'triton_padding_autotune':105,'triton_padding_bound':98,
                 'model_bound':91,'optimized_bound':80,'optimized_dispatch':82,'optimized_guarded':83}
            records=[]
            for name,value in lat.items():
                for block in range(5):records.append(dict(method=name,request_us=value+block*.1,
                    device_span_us=value/2,request_samples_us=[value],case_id='x',case=case,block=block))
            (root/'cases'/'x.json').write_text(json.dumps(dict(case=case,eligible_complete=True,methods=methods,records=records)))
            report(root)
            for name in ('comparison.csv','before_after.csv','domain_summary.csv','RESULTS.md',
                         'figures/manifest.json','figures/01_speedup_heatmap.png',
                         'figures/01_speedup_heatmap.svg','figures/01_speedup_heatmap.pdf',
                         'figures/02_latency_gemm.png','figures/04_strategy_selection.png',
                         'figures/05_benefit_regression_summary.png'):
                self.assertTrue((root/name).exists(),name)
            plots=json.loads((root/'figures'/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(plots['backend'],'matplotlib-agg')
            self.assertEqual(plots['formats'],['png','svg','pdf'])
            self.assertGreaterEqual(len(plots['plots']),5)


if __name__=='__main__':unittest.main()

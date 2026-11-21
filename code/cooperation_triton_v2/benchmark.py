"""Remote v2 experiment: exact baselines, guarded optimization, held-out results."""
import argparse
import hashlib
import json
import random
import statistics
import time
import traceback
from functools import partial
from pathlib import Path
from .specs import CONFIGS, config_id, suite
from .policy import (Dispatcher, Selector, candidates, fingerprint, fit,
                     guard_decision, select_discovery)
from .timing import measure


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def request_samples(fn,repeats):
    import torch
    for _ in range(2):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();fn();torch.cuda.synchronize()
        values.append((time.perf_counter_ns()-start)/1000)
    return values


def table_load(path,identity):
    if not path:return {'schema':2,'identity':identity,'entries':{}}
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('schema')!=2 or obj.get('identity')!=identity:
        raise ValueError('table/source/environment fingerprint mismatch')
    return obj


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['plan','calibrate','run'])
    ap.add_argument('--suite',choices=['smoke','targeted','regression','calibration'])
    ap.add_argument('--out',required=True)
    ap.add_argument('--model')
    ap.add_argument('--table')
    ap.add_argument('--ops',nargs='+')
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--repeats',type=int,default=20)
    ap.add_argument('--discovery-blocks',type=int,default=3)
    ap.add_argument('--validation-repeats',type=int,default=9)
    ap.add_argument('--seed',type=int,default=20260917)
    ap.add_argument('--unsafe-hybrid',action='store_true',help='research only: disable v2 hybrid guards')
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--graph',action='store_true',help='also record CUDA graph replay control where supported')
    args=ap.parse_args()
    if min(args.blocks,args.repeats,args.discovery_blocks,args.validation_repeats)<1:ap.error('positive counts required')
    suite_name=args.suite or ('calibration' if args.command=='calibrate' else 'targeted')
    if args.command=='calibrate' and suite_name!='calibration':ap.error('calibrate requires --suite calibration')
    cases=suite(suite_name)
    if args.ops:cases=[c for c in cases if c.op in args.ops]
    if not cases:ap.error('no cases')
    out=Path(args.out)
    if (out/'manifest.json').exists():ap.error('choose a fresh --out directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema=2,arguments=vars(args),suite=suite_name,cases=[c.record() for c in cases],
                  all_configs=CONFIGS,status='planned')
    dump(out/'manifest.json',manifest)
    if args.command=='plan':
        plan=[]
        for c in cases:
            allowed,skipped=candidates(c,args.unsafe_hybrid)
            plan.append(dict(case=c.record(),eligible=list(map(config_id,allowed)),skipped=skipped))
        dump(out/'plan.json',plan)
        print(f'{len(cases)} cases; plan only, no GPU work')
        return

    import torch
    from .runtime import environment,export_profile,inductor_default,make_inputs,Plan,torch_operation,validate
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    env=environment();identity=fingerprint(env)
    dump(out/'environment.json',dict(**env,fingerprint=identity))
    model=json.loads(Path(args.model).read_text(encoding='utf-8')) if args.model else None
    selector=Selector(identity,model)
    if model and set(model.get('training_keys',[])) & {c.key for c in cases}:
        raise ValueError('calibration/evaluation overlap')
    table=table_load(args.table,identity)
    rng=random.Random(args.seed);failures=[];training=[]
    for number,case in enumerate(cases,1):
        cid=hashlib.sha256(case.key.encode()).hexdigest()[:16]
        print(f'[{number}/{len(cases)}] {case.op} {case.m}x{case.n}x{case.k} {case.kind}',flush=True)
        groups=make_inputs(case,args.seed+number)
        reference=torch_operation(case,True)(groups)
        allowed,skipped=candidates(case,args.unsafe_hybrid)
        methods={};meta={};discovery={}
        # Candidate preparation, validation, instruction audit, independent discovery blocks.
        order=list(allowed);rng.shuffle(order)
        for cfg in order:
            name=config_id(cfg);start=time.perf_counter()
            try:
                fn=Plan(case,groups,config=cfg).prepare()
                error=validate(fn(),reference)
                audit=fn.audit(out/'artifacts'/cid/name)
                # Reuse the exact prepared launch so the changed-input check also
                # covers batched head pointer arithmetic and detects stale output.
                with torch.no_grad():
                    for a,_,_,_ in groups:a.neg_()
                try:
                    validate(fn(),torch_operation(case,True)(groups))
                finally:
                    with torch.no_grad():
                        for a,_,_,_ in groups:a.neg_()
                validate(fn(),reference)
                blocks=[measure(fn,args.repeats)['request_us'] for _ in range(args.discovery_blocks)]
                discovery[name]=blocks;methods[name]=fn
                meta[name]=dict(config=cfg,setup_seconds=time.perf_counter()-start,
                    max_abs_error=error,audit=audit,discovery_request_us=blocks)
                training.append((case,cfg,statistics.median(blocks)))
            except Exception:
                failures.append(dict(case=case.record(),method=name,phase='candidate',error=traceback.format_exc()))
                dump(out/'failures.json',failures)
        if args.command=='calibrate':
            dump(out/'calibration'/f'{cid}.json',dict(case=case.record(),eligible=list(map(config_id,allowed)),
                skipped=skipped,methods=meta,complete=set(discovery)==set(map(config_id,allowed))))
            continue

        # Original compiler strategies and a direct call bound to Triton's chosen padding config.
        try:
            ind=inductor_default(case,groups);validate(ind(),reference)
            methods['inductor_default']=ind;meta['inductor_default']={'compiler':'torch.compile/Inductor default'}
            auto=Plan(case,groups,autotune=True).prepare();validate(auto(),reference)
            meta['triton_padding_autotune']=dict(audit=auto.audit(out/'artifacts'/cid/'triton_padding_autotune'),
                configs=auto.actual_configs,wrapper='Triton autotuner remains in call path')
            methods['triton_padding_autotune']=auto
            bound=auto.frozen_padding();validate(bound(),reference)
            meta['triton_padding_bound']=dict(audit=bound.audit(out/'artifacts'/cid/'triton_padding_bound'),
                configs=bound.actual_configs,wrapper='same selected padding configs, direct prepared calls')
            methods['triton_padding_bound']=bound
        except Exception:
            failures.append(dict(case=case.record(),method='compiler_baselines',error=traceback.format_exc()))
            dump(out/'failures.json',failures)
            dump(out/'cases'/f'{cid}.json',dict(case=case.record(),skipped=skipped,methods=meta,records=[]))
            continue

        eligible_complete=set(discovery)==set(map(config_id,allowed))
        if not eligible_complete:
            failures.append(dict(case=case.record(),method='selection',error='eligible candidate set incomplete'))
            dump(out/'failures.json',failures);continue
        learned,pad_discovery=select_discovery(discovery,allowed)
        padding_unique=set(bound.actual_configs)
        model_fallback=next(iter(padding_unique)) if len(padding_unique)==1 and next(iter(padding_unique)) in allowed else pad_discovery
        model_cfg,model_source=selector.choose_model(case,allowed,model_fallback)
        entry=table['entries'].get(case.key)
        if entry:
            selected=tuple(entry['config']);source='frozen_table'
            if selected not in allowed:raise ValueError('frozen table selects ineligible config')
        else:
            selected=learned;source='discovery_table'
        selected_plan=methods[config_id(selected)]
        model_plan=methods[config_id(model_cfg)]
        methods['optimized_bound']=selected_plan
        methods['model_bound']=model_plan
        dispatcher=Dispatcher();dispatcher.register(case,selected_plan)
        dispatch=partial(dispatcher,case.dispatch_key)
        methods['optimized_dispatch']=dispatch
        meta['optimized_bound']=dict(config=selected,source=source)
        meta['model_bound']=dict(config=model_cfg,source=model_source)
        meta['optimized_dispatch']=dict(config=selected,source=source,key_precomputed=True,callable_cached=True)
        # Validation is distinct from discovery and final holdout.
        if entry and 'route' in entry:
            route=entry['route'];qualifies=entry.get('qualified',route=='optimized_dispatch')
            validation={}
        else:
            validation={n:request_samples(methods[n],args.validation_repeats)
                        for n in ('inductor_default','triton_padding_bound','optimized_dispatch')}
            route,qualifies=guard_decision(validation)
            table['entries'][case.key]=dict(config=selected,route=route,qualified=qualifies,
                source=source,validation_request_us=validation,margin=.03)
        methods['optimized_guarded']=methods[route]
        meta['optimized_guarded']=dict(route=route,selected_config=selected,qualified=qualifies,
            validation_request_us=validation,decision_frozen_before_holdout=True)
        records=[]
        # All methods, including aliases, are independently timed in randomized holdout order.
        for block in range(args.blocks):
            names=list(methods);rng.shuffle(names)
            for name in names:
                try:
                    result=measure(methods[name],args.repeats,args.graph)
                    records.append(dict(case_id=cid,case=case.record(),method=name,block=block,**result))
                except Exception:
                    failures.append(dict(case=case.record(),method=name,phase='holdout',block=block,error=traceback.format_exc()))
        if args.profile:
            for name in ('inductor_default','triton_padding_bound','optimized_bound','optimized_dispatch','optimized_guarded'):
                try:export_profile(methods[name],out/'traces'/f'{cid}_{name}.json')
                except Exception:failures.append(dict(case=case.record(),method=name,phase='profile',error=traceback.format_exc()))
        dump(out/'cases'/f'{cid}.json',dict(case=case.record(),eligible=list(map(config_id,allowed)),skipped=skipped,
            eligible_complete=eligible_complete,selected_config=selected,model_config=model_cfg,
            methods=meta,records=records))
        dump(out/'table.json',table);dump(out/'failures.json',failures)
        del methods,groups,reference
        torch._dynamo.reset()
    dump(out/'failures.json',failures)
    if args.command=='calibrate':
        if failures:
            dump(out/'completion.json',dict(complete=False,failures=len(failures)))
            raise SystemExit('calibration incomplete; model not fitted')
        dump(out/'model.json',fit(training,identity))
    else:
        from .report import report
        report(out)
    complete=not failures
    dump(out/'completion.json',dict(complete=complete,failures=len(failures),
        performance_evidence=complete and args.blocks>=5 and suite_name!='smoke'))
    if failures:raise SystemExit('failures recorded; missing results are not winners')


if __name__=='__main__':main()

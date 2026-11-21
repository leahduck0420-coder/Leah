"""Aggregate held-out blocks and generate CSV, Markdown and server-side charts."""
import csv
import json
import math
import statistics
from pathlib import Path
from .specs import config_id

BASELINES=('inductor_default','triton_padding_autotune','triton_padding_bound')
DISPLAY=('inductor_default','triton_padding_autotune','triton_padding_bound','model_bound',
         'optimized_bound','optimized_dispatch','optimized_guarded')


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def gm(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def parse_config(method,meta):
    cfg=meta.get('config') or meta.get('selected_config')
    if not cfg and method.startswith(('padding_','small_mma_','hybrid_')):
        policy,tile=method.rsplit('_',1);cfg=[policy,*map(int,tile.split('x'))]
    return tuple(cfg) if cfg else ()


def report(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=manifest['arguments']['blocks'];rows=[];case_docs={}
    for path in sorted((root/'cases').glob('*.json')):
        doc=json.loads(path.read_text(encoding='utf-8'));case=doc['case'];case_docs[path.stem]=doc
        grouped={}
        for r in doc['records']:grouped.setdefault(r['method'],[]).append(r)
        for method,blocks in grouped.items():
            req=[x['request_us'] for x in blocks];dev=[x['device_span_us'] for x in blocks]
            meta=doc['methods'].get(method,{})
            cfg=parse_config(method,meta);policy=cfg[0] if cfg else '';tile='x'.join(map(str,cfg[1:])) if cfg else ''
            row=dict(case_id=path.stem,op=case['op'],domain=case['domain'],kind=case['kind'],pair=case['pair'],
                m=case['m'],n=case['n'],k=case['k'],batch=case['batch'],tokens=json.dumps(case['tokens']),
                shape=f"{case['m']}x{case['n']}x{case['k']}",method=method,policy=policy,tile=tile,
                route=meta.get('route',''),source=meta.get('source',''),blocks=len(blocks),complete=len(blocks)==expected,
                eligible_complete=doc.get('eligible_complete',False),request_us=statistics.median(req),
                request_min_us=min(req),request_max_us=max(req),request_spread=(max(req)-min(req))/statistics.median(req),
                device_span_us=statistics.median(dev),launch_count=(meta.get('audit') or {}).get('launch_count',''),
                batched=(meta.get('audit') or {}).get('batched',''))
            graph=[x.get('graph_replay_us') for x in blocks if x.get('graph_replay_us') is not None]
            row['graph_replay_us']=statistics.median(graph) if graph else ''
            rows.append(row)
    lookup={(r['case_id'],r['method']):r for r in rows}
    for r in rows:
        for b in BASELINES:
            base=lookup.get((r['case_id'],b));speed=base['request_us']/r['request_us'] if base else ''
            r['speedup_vs_'+b]=speed
            r['clear_win_vs_'+b]=bool(base and r['complete'] and base['complete'] and expected>=5 and
                speed>1.05 and r['request_max_us']<base['request_min_us'] and
                r['request_spread']<.1 and base['request_spread']<.1)
    write_csv(root/'comparison.csv',rows)
    before=[]
    for cid in sorted(case_docs):
        relevant={m:lookup.get((cid,m)) for m in DISPLAY}
        if not all(relevant[x] for x in ('inductor_default','triton_padding_bound','optimized_guarded')):continue
        x=relevant['optimized_guarded']
        before.append(dict(case_id=cid,op=x['op'],domain=x['domain'],kind=x['kind'],pair=x['pair'],shape=x['shape'],
            inductor_default_us=relevant['inductor_default']['request_us'],
            triton_padding_autotune_us=relevant['triton_padding_autotune']['request_us'] if relevant['triton_padding_autotune'] else '',
            triton_padding_bound_us=relevant['triton_padding_bound']['request_us'],
            optimized_bound_us=relevant['optimized_bound']['request_us'] if relevant['optimized_bound'] else '',
            optimized_dispatch_us=relevant['optimized_dispatch']['request_us'] if relevant['optimized_dispatch'] else '',
            final_guarded_us=x['request_us'],final_route=x['route'],optimized_used=x['route']=='optimized_dispatch',
            speedup_vs_inductor=x['speedup_vs_inductor_default'],
            speedup_vs_triton_padding_autotune=x['speedup_vs_triton_padding_autotune'],
            speedup_vs_triton_padding=x['speedup_vs_triton_padding_bound'],
            selected_policy=relevant['optimized_bound']['policy'] if relevant['optimized_bound'] else '',
            selected_tile=relevant['optimized_bound']['tile'] if relevant['optimized_bound'] else ''))
    write_csv(root/'before_after.csv',before)
    summary=[]
    for op,domain in sorted({(r['op'],r['domain']) for r in rows}):
        for method in ('model_bound','optimized_bound','optimized_dispatch','optimized_guarded'):
            for base in BASELINES:
                subset=[r for r in rows if r['op']==op and r['domain']==domain and r['method']==method and r['speedup_vs_'+base] != '']
                if not subset:continue
                vals=[r['speedup_vs_'+base] for r in subset]
                summary.append(dict(op=op,domain=domain,method=method,baseline=base,n=len(vals),
                    geomean_speedup=gm(vals),best_speedup=max(vals),worst_speedup=min(vals),
                    faster_count=sum(v>1 for v in vals),faster_5pct_count=sum(v>1.05 for v in vals),
                    slower_5pct_count=sum(v<1/1.05 for v in vals),clear_win_count=sum(r['clear_win_vs_'+base] for r in subset)))
    write_csv(root/'domain_summary.csv',summary)
    pairs=[]
    bypair={}
    for r in before:
        if r['pair']:bypair.setdefault((r['op'],r['pair']),[]).append(r)
    for (op,pair),group in sorted(bypair.items()):
        control=next((r for r in group if r['kind']=='parent_aligned_control'),None)
        if not control:continue
        for r in group:
            if r is control:continue
            q=dict(r);q['control_shape']=control['shape']
            q['inductor_tail_ratio']=r['inductor_default_us']/control['inductor_default_us']
            q['triton_padding_tail_ratio']=r['triton_padding_bound_us']/control['triton_padding_bound_us']
            q['final_tail_ratio']=r['final_guarded_us']/control['final_guarded_us']
            pairs.append(q)
    write_csv(root/'tail_pairs.csv',pairs)
    from .plot_results import generate_plots
    figures=generate_plots(root)
    failures=json.loads((root/'failures.json').read_text(encoding='utf-8')) if (root/'failures.json').exists() else []
    used=sum(r['optimized_used'] for r in before);positive=sum(r['speedup_vs_inductor']>1.05 for r in before)
    domains=[]
    for s in summary:
        if s['method']=='optimized_guarded' and s['baseline']=='inductor_default':
            domains.append(f"|{s['op']}|{s['domain']}|{s['n']}|{s['geomean_speedup']:.3f}×|{s['best_speedup']:.3f}×|{s['worst_speedup']:.3f}×|{s['faster_count']}/{s['n']}|")
    text=['# TC/SIMT v2 远程实验结果','',
      '本报告同时保留原编译器路径、相同配置直接调用、优化 kernel、含查表调用和验证回退后的最终方案。','',
      f'- case 数：{len(before)}；失败记录：{len(failures)}；最终实际采用优化路径：{used}/{len(before)}。',
      f'- 最终方案相对 Inductor 超过 5% 的 holdout case：{positive}/{len(before)}。这不是独立机器复现数。',
      '- `triton_padding_bound` 是 Triton autotune 选完配置后用相同 Plan 直接调用，用于隔离 autotuner 包装开销。',
      '- `optimized_guarded` 的 route 在 discovery 之后的独立 validation 阶段冻结；holdout 不参与选 route。','',
      '## 按预先声明领域汇总','',
      '|算子|领域|case|最终方案/Inductor 几何平均|最好|最差|快于基线|','|---|---|---:|---:|---:|---:|---:|',*domains,'',
      '## 文件','',
      f'- `figures/`：服务器用 Matplotlib Agg 生成的 {len(figures["plots"])} 组图表，每组同时输出 PNG、SVG、PDF。',
      '- `before_after.csv`：每个 shape 的原编译器、Triton padding 和最终方案性能。',
      '- `comparison.csv`：全部候选和所有报告方法。',
      '- `domain_summary.csv`：各领域几何平均、最好、最差和退化。',
      '- `tail_pairs.csv`：对齐父 shape 与不能整除 Tile 的 shape 成对比较。',
      '- `figures/manifest.json`：图表文件、来源数据和说明。','',
      '加速比 >1 表示更快。request 包含 Python 和同步；device span 包含供给间隙。任何失败或缺失都不会补零。']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

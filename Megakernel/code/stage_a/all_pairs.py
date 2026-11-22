"""Workload-scoped single-rule admission and cross-stage source identity."""
from dataclasses import dataclass
from pathlib import Path

from .compiler import Plan
from .util import Rejected,digest,file_hash

PROTOCOL="all-five-per-original-anchor-triton390-v1"
ROOT=Path(__file__).resolve().parents[1]

def fingerprint():
    return digest({p.relative_to(ROOT).as_posix():file_hash(p)
        for folder in ("stage_a","stage_b","configs") for p in (ROOT/folder).rglob("*")
        if p.is_file() and p.suffix in (".py",".yaml",".json")})

@dataclass(frozen=True)
class SinglePlan(Plan):
    workers:int
    head_group:int

def joint_single(case,enabled):
    from stage_b.planner import derive,inventory,original_action
    w,s=case["workload"],case["strategy"]
    seq=()
    if enabled:
        if not any(a.strategy==s for a in inventory(case)):
            if s=="S3":reason="no sufficient-statistic/reconstruction contract at an intermediate edge"
            elif s=="S5":reason="no independently published logical tasks in the unchanged workload runtime"
            elif w=="W1" and s=="S4":reason="no reduction domain on the elementwise chain"
            else:reason="no established local owner for published predecessor output in the task DAG"
            raise Rejected("NOT_APPLICABLE",reason)
        seq=(original_action(case),)
    ex=derive(case,seq)
    proof=dict(ex.proof)
    if ex.cut:proof["ordered_partition"]=proof["S4"]["partition"]
    return SinglePlan(s if seq else "none",seq[0].location if seq else "baseline",ex.tile,ex.merge,ex.local,
                      ex.stats,ex.cut,ex.scheduling,proof,ex.ownership,ex.ddg,ex.workers,ex.head_group)

def planned_pairs(rows):
    from collections import defaultdict
    import statistics
    from .matrix import STRATEGIES,SUPPORT
    from .report import valid
    groups=defaultdict(list)
    for r in rows:groups[r["workload"],r["strategy"]].append(r)
    result=[]
    for w in SUPPORT:
        for s in STRATEGIES:
            items=groups[w,s]
            applied=[r for r in items if valid(r)]
            fb=[r for r in items if r.get("fallback",{}).get("status")=="MEASURED"]
            result.append(dict(workload=w,strategy=s,requested=len(items),measured_applied=len(applied),
                  measured_fallback=len(fb),status_counts={key:sum(r["status"]==key for r in items) for key in sorted({r["status"] for r in items})},
                  applied_median_B1_over_B2=statistics.median(r["incremental_speedup"] for r in applied) if applied else None,
                  applied_median_B0_over_B2=statistics.median(r["total_speedup"] for r in applied) if applied else None,
                  fallback_median_us=statistics.median(r["fallback"]["statistics"]["fallback"]["median"] for r in fb) if fb else None))
    return result

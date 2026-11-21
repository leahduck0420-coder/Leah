"""Predeclared domains, alignment controls, and immutable cached dispatch keys."""
from dataclasses import dataclass, asdict
from functools import cached_property
import json

TILES = ((32,32,32),(64,64,32),(128,64,32),(64,128,64))
POLICIES = ('padding','small_mma','hybrid','hybrid_k')
CONFIGS = tuple((p,*t) for t in TILES for p in POLICIES)
OPS = ('gemm','linear','linear_post','qkt','pv','moe')

def config_id(config):
    return config[0]+'_'+'x'.join(map(str,config[1:]))

@dataclass(frozen=True)
class Case:
    op: str
    m: int
    n: int
    k: int
    batch: int = 1
    ep: str = 'none'
    tokens: tuple = ()
    domain: str = 'general'
    kind: str = 'tail'
    pair: str = ''

    def __post_init__(self):
        if self.op not in OPS or min(self.m,self.n,self.k,self.batch)<1:
            raise ValueError('invalid operator or dimensions')
        if self.ep not in ('none','bias','bias_relu_residual','bias_gelu_residual'):
            raise ValueError('invalid epilogue')
        if self.op=='moe' and (not self.tokens or min(self.tokens)<0 or not any(self.tokens)):
            raise ValueError('invalid expert token counts')

    @cached_property
    def groups(self):
        return tuple(m for m in self.tokens if m) if self.op=='moe' else (self.m,)*self.batch

    @cached_property
    def transpose_b(self):
        return self.op in ('linear','linear_post','qkt','moe')

    @cached_property
    def dispatch_key(self):
        # Input contract: CUDA FP16, contiguous A, op-defined B layout, FP16 boundary.
        return (self.op,self.m,self.n,self.k,self.batch,self.ep,self.tokens,'fp16',self.transpose_b)

    @cached_property
    def key(self):
        return json.dumps(self.dispatch_key,separators=(',',':'))

    @property
    def flops(self):
        return 2*sum(self.groups)*self.n*self.k

    def record(self):
        return asdict(self)


def build(op,dims,domain,kind='tail',pair=''):
    m,n,k=dims
    return Case(op,m,n,k,batch=4 if op in ('qkt','pv') else 1,
                ep='bias' if op=='linear' else 'bias_gelu_residual' if op=='linear_post' else 'none',
                tokens=(0,1,max(1,m//4),m) if op=='moe' else (),domain=domain,kind=kind,pair=pair)


def suite(name='targeted'):
    out=[]
    if name=='smoke':
        for op in OPS:
            for dims in ((32,32,32),(17,19,33),(7,9,5)):
                out.append(build(op,dims,'smoke'))
    elif name=='regression':
        for op in OPS:
            for dims in ((128,128,128),(127,128,128),(129,128,128),(128,129,128),
                         (128,128,129),(129,129,129),(144,144,144),(512,512,512),
                         (513,513,513),(128,1024,513),(1024,1024,1024),(1,4096,4096)):
                out.append(build(op,dims,'v1_regression','control' if all(x%128==0 for x in dims) else 'tail'))
    elif name=='targeted':
        # Predeclared matrix domains, not shape selection from v2 holdout results.
        domains={
          'gemm':('general_tail',[(128,128,128),(512,512,512),(128,1024,512)]),
          'linear':('ragged_token_projection',[(128,768,768),(256,1024,1024)]),
          'linear_post':('fused_ragged_projection',[(128,768,768),(256,1024,1024)]),
          'qkt':('irregular_attention_sequence',[(128,128,64),(512,512,128)]),
          'pv':('irregular_attention_reduction',[(128,64,128),(512,128,512)]),
          'moe':('ragged_expert_projection',[(128,768,768),(256,1024,1024)]),
        }
        for op,(domain,bases) in domains.items():
            for m,n,k in bases:
                pair=f'{op}_{m}x{n}x{k}'
                out.append(build(op,(m,n,k),domain,'parent_aligned_control',pair))
                variants=[((m+1,n,k),'m_tail'),((m,n+1,k),'n_tail'),
                          ((m,n,k+1),'k_tail_1'),((m,n,k+3),'k_tail_3'),
                          ((m+1,n+1,k+1),'mixed_tail')]
                for dims,kind in variants:
                    out.append(build(op,dims,domain,kind,pair))
        # Negative control where v1 incorrectly chose full-K SIMT.
        for op in OPS:
            out.append(build(op,(1,4096,4096),'narrow_negative_control','negative_control'))
    elif name=='calibration':
        for op in OPS:
            for dims in ((48,80,65),(96,160,127),(192,768,769),(320,1024,1027),
                         (384,384,64),(640,640,129),(768,128,769),(768,768,768),
                         (1536,1536,1537),(2,4096,4096),(8,3072,3073),(48,2048,2049)):
                out.append(build(op,dims,'calibration'))
    else:
        raise ValueError('unknown suite')
    keys=[c.key for c in out]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate workload keys')
    return out

"""Preparation owns metadata work; hot launch owns only bound calls.

Uniform shared-storage heads use grid axis 2. Ragged experts remain separate
launches. All padding controls use the same launch infrastructure.
"""
from functools import partial
from pathlib import Path
import json
import platform
import re
from .specs import CONFIGS
from .policy import geometry


def environment():
    import torch,triton
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError('Linux/NVIDIA CUDA GPU required')
    if torch.cuda.current_device()!=0:
        raise RuntimeError('select target GPU through CUDA_VISIBLE_DEVICES; logical device must be 0')
    p=torch.cuda.get_device_properties(0)
    if p.major<7:raise RuntimeError('Tensor Core GPU required')
    return dict(gpu=p.name,capability=[p.major,p.minor],sm_count=p.multi_processor_count,
                memory=p.total_memory,torch=torch.__version__,triton=triton.__version__,
                cuda=torch.version.cuda,python=platform.python_version(),platform=platform.platform())


def make_inputs(case,seed=20260917):
    import torch
    torch.manual_seed(seed)
    kw=dict(device='cuda',dtype=torch.float16)
    if len(set(case.groups))==1:
        count=len(case.groups);m=case.groups[0]
        a=torch.randn((count,m,case.k),**kw)*.2
        b=torch.randn((count,case.n,case.k) if case.transpose_b else (count,case.k,case.n),**kw)*.2
        bias=torch.randn((count,case.n),**kw)*.1
        r=torch.randn((count,m,case.n),**kw)*.1
        return tuple((a[i],b[i].T if case.transpose_b else b[i],bias[i],r[i]) for i in range(count))
    groups=[]
    for m in case.groups:
        a=torch.randn((m,case.k),**kw)*.2
        b=torch.randn((case.n,case.k) if case.transpose_b else (case.k,case.n),**kw)*.2
        groups.append((a,b.T if case.transpose_b else b,torch.randn((case.n,),**kw)*.1,
                       torch.randn((m,case.n),**kw)*.1))
    return tuple(groups)


def torch_operation(case,reference=False):
    import torch
    def op(groups):
        outputs=[]
        for a,b,bias,residual in groups:
            y=((a.double()@b.double()).half() if reference else a@b).float()
            if case.op=='qkt':y=y*case.k**-.5
            if case.ep!='none':y=y+bias.float()
            if case.ep=='bias_relu_residual':y=torch.relu(y)
            if case.ep=='bias_gelu_residual':
                z=.7978845608028654*(y+.044715*y*y*y)
                y=.5*y*(1+2*torch.sigmoid(2*z)-1)
            if 'residual' in case.ep:y=y+residual.float()
            outputs.append(y.half())
        return tuple(outputs)
    return op


def validate(outputs,reference):
    import torch
    if len(outputs)!=len(reference):raise AssertionError('group count mismatch')
    errors=[]
    for a,b in zip(outputs,reference):
        if not torch.isfinite(a).all():raise AssertionError('nonfinite or unwritten output')
        torch.testing.assert_close(a,b,rtol=.02,atol=.005)
        errors.append(float((a.float()-b.float()).abs().max()))
    return max(errors)


def regular_stride(tensors):
    """Prove common storage and regular offsets before enabling batched pointer math."""
    if len(tensors)==1:return 0
    root=tensors[0].untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr()!=root for t in tensors):return None
    step=tensors[1].storage_offset()-tensors[0].storage_offset()
    if step<=0:return None
    for i,t in enumerate(tensors):
        if t.shape!=tensors[0].shape or t.stride()!=tensors[0].stride():return None
        if t.storage_offset()!=tensors[0].storage_offset()+i*step:return None
    return step


class Plan:
    def __init__(self,case,groups,config=None,autotune=False,configs=None,batch_heads=True):
        import torch
        from .kernels import cooperative,padding_autotuned
        if len(groups)!=len(case.groups):raise ValueError('wrong group count')
        for (a,b,bias,r),m in zip(groups,case.groups):
            for t,shape in zip((a,b,bias,r),((m,case.k),(case.k,case.n),(case.n,),(m,case.n))):
                if t.dtype!=torch.float16 or tuple(t.shape)!=shape or t.device!=groups[0][0].device or t.device.type!='cuda':
                    raise ValueError('shape/device/dtype contract violation')
            if not (a.is_contiguous() and bias.is_contiguous() and r.is_contiguous()):
                raise ValueError('A, bias, residual must be contiguous')
            if not (b.T.is_contiguous() if case.transpose_b else b.is_contiguous()):
                raise ValueError('B layout does not match Case')
        self.case,self.groups,self.autotune=case,groups,autotune
        cfgs=[tuple(x) for x in configs] if configs is not None else [config]*len(groups)
        if len(cfgs)!=len(groups) or (not autotune and any(c not in CONFIGS for c in cfgs)):
            raise ValueError('invalid configurations')
        strides=[regular_stride([g[i] for g in groups]) for i in range(4)]
        self.batched=(batch_heads and len(groups)>1 and len(set(case.groups))==1
                      and all(s is not None for s in strides) and len(set(cfgs))==1)
        if self.batched:
            packed=torch.empty((len(groups),case.m,case.n),device=groups[0][0].device,dtype=torch.float16)
            self.out=tuple(packed[i] for i in range(len(groups)))
            jobs=[(0,len(groups))]
        else:
            self.out=tuple(torch.empty((m,case.n),device=groups[0][0].device,dtype=torch.float16) for m in case.groups)
            jobs=[(i,1) for i in range(len(groups))]
        self.jobs=[]
        for i,count in jobs:
            a,b,bias,residual=groups[i];c=self.out[i];m=a.shape[0]
            kw=dict(M=m,N=case.n,K=case.k,AM=a.stride(0),AK=a.stride(1),
                    BK_STRIDE=b.stride(0),BN_STRIDE=b.stride(1),EP=case.ep,
                    SCALE=case.k**-.5 if case.op=='qkt' else 1.0,BATCH=count,
                    SA=strides[0] if count>1 else 0,SB=strides[1] if count>1 else 0,
                    SBIAS=strides[2] if count>1 else 0,SR=strides[3] if count>1 else 0,
                    SC=case.m*case.n if count>1 else 0)
            if autotune:
                kernel=padding_autotuned
                kw['POLICY']='padding'
                def grid(meta,m=m,n=case.n,count=count):
                    return ((m+meta['BM']-1)//meta['BM'],(n+meta['BN']-1)//meta['BN'],count)
            else:
                kernel=cooperative
                policy,bm,bn,bk=cfgs[i]
                kw.update(POLICY=policy,BM=bm,BN=bn,BK=bk,num_warps=4,num_stages=2)
                grid=((m+bm-1)//bm,(case.n+bn-1)//bn,count)
            # Keyword dictionary and launcher are captured once, not rebuilt each call.
            launch=partial(kernel[grid],a,b,c,bias,residual,**kw)
            self.jobs.append(dict(launch=launch,kernel=kernel,index=i,count=count,config=cfgs[i]))
        self.launches=tuple(j['launch'] for j in self.jobs)
        self.compiled=[]
        self.actual_configs=[]

    def __call__(self):
        for launch in self.launches:launch()
        return self.out

    def prepare(self):
        """Only this method records compilation/config metadata."""
        self.compiled=[];self.actual_configs=[]
        for j in self.jobs:
            obj=j['launch']();self.compiled.append(obj)
            if self.autotune:
                k=j['kernel'].best_config.kwargs
                cfg=('padding',k['BM'],k['BN'],k['BK'])
            else:cfg=j['config']
            self.actual_configs.extend([cfg]*j['count'])
        return self

    def frozen_padding(self):
        if not self.autotune or not self.actual_configs:raise ValueError('prepare autotune first')
        return Plan(self.case,self.groups,configs=self.actual_configs,batch_heads=self.batched).prepare()

    def audit(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        records=[]
        for i,(j,obj) in enumerate(zip(self.jobs,self.compiled)):
            cfg=self.actual_configs[j['index']]
            g=geometry(self.case.groups[j['index']],self.case.n,self.case.k,cfg)
            for ext,code in obj.asm.items():
                if isinstance(code,str):(directory/f'launch{i}.{ext}').write_text(code,encoding='utf-8')
            ptx=obj.asm.get('ptx','')
            mma=bool(re.search(r'\b(?:mma|wgmma|tcgen05\.mma)[.\w]*',ptx))
            fma=bool(re.search(r'\bfma\.[^;\n]*f32',ptx))
            verified=(not g['tc_slots'] or mma) and (not g['useful_simt_macs'] or fma)
            records.append(dict(config=cfg,batch=j['count'],geometry_per_group=g,mma_present=mma,
                fma_present=fma,identity_verified=verified,registers=getattr(obj,'n_regs',None),
                spills=getattr(obj,'n_spills',None),shared_bytes=getattr(obj.metadata,'shared',None),
                convert_layout_count=obj.asm.get('ttgir','').count('convert_layout')))
        (directory/'audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        if not all(r['identity_verified'] for r in records):raise RuntimeError('PTX identity audit failed')
        return dict(launch_count=len(self.launches),batched=self.batched,launches=records)


def inductor_default(case,groups):
    import torch
    fn=torch.compile(torch_operation(case),backend='inductor',fullgraph=True,dynamic=False)
    return partial(fn,groups)


def export_profile(fn,path):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        fn();torch.cuda.synchronize()
    p.export_chrome_trace(str(path))

import triton
import triton.language as tl
from .kernels import _materialize, _att_row


@triton.jit
def mlp(X,G,Wg,Wu,Wd,T,Pair,H,Y,Stats,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
        BN:tl.constexpr,LOCAL:tl.constexpr,REBUILD:tl.constexpr,BM:tl.constexpr=16,CUT:tl.constexpr=0):
    rr=tl.program_id(0)*BM+tl.arange(0,BM)
    kk=tl.arange(0,1024)
    idx=rr[:,None]*K+kk[None,:]
    x=tl.load(X+idx,(rr[:,None]<M)&(kk[None,:]<K),0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,1)/K+1.e-6)
    if REBUILD:
        tl.store(Stats+rr,inv,rr<M)
    else:
        scale=tl.load(G+kk,kk<K,0).to(tl.float32)
        tl.store(T+idx,((x*inv[:,None]).to(X.dtype.element_ty)*scale[None,:]).to(X.dtype.element_ty),(rr[:,None]<M)&(kk[None,:]<K))
    tl.debug_barrier()
    for panel in range(tl.cdiv(N,BN)):
        n=panel*BN+tl.arange(0,BN)
        ag=tl.full((BM,BN),0,tl.float32)
        au=tl.full((BM,BN),0,tl.float32)
        for j in range(tl.cdiv(K,32)):
            k=j*32+tl.arange(0,32)
            mask=(rr[:,None]<M)&(k[None,:]<K)
            if REBUILD:
                raw=tl.load(X+rr[:,None]*K+k[None,:],mask,0).to(tl.float32)
                ri=tl.load(Stats+rr,rr<M,0,volatile=True)
                g=tl.load(G+k,k<K,0).to(tl.float32)
                a=((raw*ri[:,None]).to(X.dtype.element_ty)*g[None,:]).to(X.dtype.element_ty)
            else:
                a=tl.load(T+rr[:,None]*K+k[None,:],mask,0,volatile=True)
            wi=k[:,None]+n[None,:]*K
            wm=(k[:,None]<K)&(n[None,:]<N)
            ag=tl.dot(a,tl.load(Wg+wi,wm,0),ag)
            au=tl.dot(a,tl.load(Wu+wi,wm,0),au)
        pos=rr[:,None]*N+n[None,:]
        active=(rr[:,None]<M)&(n[None,:]<N)
        # Official projection boundary: each linear returns input dtype before SwiGLU.
        gate=ag.to(X.dtype.element_ty)
        up=au.to(X.dtype.element_ty)
        if not (LOCAL & 1):
            gate=_materialize(Pair,pos,gate,active)
        if not (LOCAL & 2):
            up=_materialize(Pair,M*N+pos,up,active)
        silu=(gate.to(tl.float32)*tl.sigmoid(gate.to(tl.float32))).to(X.dtype.element_ty)
        tl.store(H+pos,silu*up,active)
    tl.debug_barrier()
    for panel in range(tl.cdiv(K,32)):
        col=panel*32+tl.arange(0,32)
        acc=tl.full((BM,32),0,tl.float32)
        if CUT:
            for part in range(tl.cdiv(N,CUT)):
                partial=tl.full((BM,32),0,tl.float32)
                for j in tl.static_range(CUT//32):
                    n=part*CUT+j*32+tl.arange(0,32)
                    hs=tl.load(H+rr[:,None]*N+n[None,:],(rr[:,None]<M)&(n[None,:]<N),0,volatile=True)
                    ws=tl.load(Wd+n[:,None]+col[None,:]*N,(n[:,None]<N)&(col[None,:]<K),0)
                    partial=tl.dot(hs,ws,partial)
                acc+=partial
        else:
            for j in range(tl.cdiv(N,32)):
                n=j*32+tl.arange(0,32)
                hs=tl.load(H+rr[:,None]*N+n[None,:],(rr[:,None]<M)&(n[None,:]<N),0,volatile=True)
                ws=tl.load(Wd+n[:,None]+col[None,:]*N,(n[:,None]<N)&(col[None,:]<K),0)
                acc=tl.dot(hs,ws,acc)
        tl.store(Y+rr[:,None]*K+col[None,:],acc,(rr[:,None]<M)&(col[None,:]<K))


@triton.jit
def _qkv(X,W,Bias,Cos,Sin,Q,K,V,batch,L:tl.constexpr,KL:tl.constexpr,BIAS_MODE:tl.constexpr):
    for rb in range(tl.cdiv(L,16)):
        r=rb*16+tl.arange(0,16)
        for head in range(18):
            # One complete head permits half-rotation without inter-CTA visibility.
            d=tl.arange(0,64)
            acc=tl.full((16,64),0,tl.float32)
            for j in range(28):
                k=j*32+tl.arange(0,32)
                x=tl.load(X+(batch*L+r[:,None])*896+k[None,:],r[:,None]<L,0)
                w=tl.load(W+(head*64+d[None,:])*896+k[:,None])
                acc=tl.dot(x,w,acc)
            staged=tl.where(head<14,BIAS_MODE&1,tl.where(head<16,BIAS_MODE&2,BIAS_MODE&4))!=0
            if staged:
                acc=acc.to(X.dtype.element_ty).to(tl.float32)
            projected=(acc+tl.load(Bias+head*64+d)[None,:].to(tl.float32)).to(X.dtype.element_ty)
            if head<16:
                kk=tl.broadcast_to((d^32)[None,:],(16,64))
                half=tl.gather(projected,kk,1)
                angle=(KL-L+r[:,None])*64+d[None,:]
                c=tl.load(Cos+angle,r[:,None]<L,0)
                s=tl.load(Sin+angle,r[:,None]<L,0)
                rot=tl.where(d[None,:]<32,-half,half)
                v1=(projected*c).to(X.dtype.element_ty)
                v2=(rot*s).to(X.dtype.element_ty)
                y=(v1+v2).to(X.dtype.element_ty)
                if head<14:
                    tl.store(Q+((batch*L+r[:,None])*14+head)*64+d[None,:],y,r[:,None]<L)
                else:
                    tl.store(K+((batch*KL+KL-L+r[:,None])*2+head-14)*64+d[None,:],y,r[:,None]<L)
            else:
                tl.store(V+((batch*KL+KL-L+r[:,None])*2+head-16)*64+d[None,:],projected,r[:,None]<L)


@triton.jit
def _out(A,W,Y,batch,L:tl.constexpr):
    for rb in range(tl.cdiv(L,16)):
        r=rb*16+tl.arange(0,16)
        for panel in range(28):
            d=panel*32+tl.arange(0,32)
            acc=tl.full((16,32),0,tl.float32)
            for j in range(28):
                k=j*32+tl.arange(0,32)
                a=tl.load(A+(batch*L+r[:,None])*896+k[None,:],r[:,None]<L,0)
                w=tl.load(W+d[None,:]*896+k[:,None])
                acc=tl.dot(a,w,acc)
            tl.store(Y+(batch*L+r[:,None])*896+d[None,:],acc,r[:,None]<L)


@triton.jit
def _dag_task(X,Y,DataDeps,task,NT:tl.constexpr,WIDTH:tl.constexpr,MIXED:tl.constexpr,SLICED:tl.constexpr,CUT:tl.constexpr):
    j=tl.arange(0,WIDTH)
    val=tl.load(X+task*WIDTH+j).to(tl.float32)
    if SLICED:
        for block in range(tl.cdiv(task,CUT)):
            partial=tl.full((WIDTH,),0,tl.float32)
            for i in tl.static_range(CUT):
                par=block*CUT+i
                if par<task:
                    if tl.load(DataDeps+task*NT+par)!=0:
                        partial+=tl.load(Y+par*WIDTH+j,volatile=True).to(tl.float32)*0.125
            val+=partial
    else:
        for par in range(task):
            if tl.load(DataDeps+task*NT+par)!=0:
                val+=tl.load(Y+par*WIDTH+j,volatile=True).to(tl.float32)*0.125
    iters=1+task%7 if MIXED else 1
    for it in range(iters):
        val=val*0.75+0.03125
    tl.store(Y+task*WIDTH+j,val)


@triton.jit
def dispatch(X,W,Bias,Cos,Sin,Q,K,V,A,Wo,Y,T,Stats,DataDeps,Order,State,Count,Trace,
             TASKS:tl.constexpr,PAD:tl.constexpr,KIND:tl.constexpr,L:tl.constexpr,KL:tl.constexpr,
             PB:tl.constexpr,CUT:tl.constexpr,STATS:tl.constexpr,SLICED:tl.constexpr,
             WIDTH:tl.constexpr,MIXED:tl.constexpr,INSTRUMENT:tl.constexpr,QKV_BIAS_MODE:tl.constexpr=0,
             HEAD_GROUP:tl.constexpr=1,HOIST:tl.constexpr=False,LOCAL:tl.constexpr=False,TASK_GROUP:tl.constexpr=1,LOGICAL_TASKS:tl.constexpr=1):
    # Bounded, non-waiting scheduler. A failed sweep retires the worker. The worker
    # publishing the last running task can claim any newly ready successor.
    ids=tl.arange(0,PAD)
    running=1
    for sweep in range(TASKS):
        if running:
            did=0
            for u in range(TASKS):
                t=(u+tl.program_id(0))%TASKS
                deps=tl.load(Order+t*TASKS+ids,ids<TASKS,0)
                flags=tl.atomic_add(State+ids,tl.full((PAD,),0,tl.int32),mask=ids<TASKS,sem="acquire",scope="gpu")
                ready=tl.sum(((deps!=0)&(flags!=2)).to(tl.int32),0)==0
                if ready:
                    old=tl.atomic_cas(State+t,0,1,sem="acq_rel",scope="gpu")
                    if old==0:
                        if INSTRUMENT:
                            start=tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;","=l",[],dtype=tl.int64,is_pure=False,pack=1)
                            tl.store(Trace+3*t,start)
                        tl.debug_barrier()
                        if KIND==6:
                            if TASK_GROUP==1:
                                _dag_task(X,Y,DataDeps,t,TASKS,WIDTH,MIXED,SLICED,CUT)
                            else:
                                for member in tl.static_range(TASK_GROUP):
                                    task=t*TASK_GROUP+member
                                    if task<LOGICAL_TASKS:
                                        _dag_task(X,Y,DataDeps,task,LOGICAL_TASKS,WIDTH,MIXED,SLICED,CUT)
                                        tl.debug_barrier()
                                        tl.atomic_add(Count+TASKS+task,1,sem="relaxed",scope="gpu")
                        else:
                            groups:tl.constexpr=(14+HEAD_GROUP-1)//HEAD_GROUP
                            b=t//(groups+2)
                            h=t%(groups+2)
                            if h==0:
                                _qkv(X,W,Bias,Cos,Sin,Q,K,V,b,L,KL,QKV_BIAS_MODE)
                            elif h==groups+1:
                                _out(A,Wo,Y,b,L)
                            else:
                                for member in tl.static_range(HEAD_GROUP):
                                    head=(h-1)*HEAD_GROUP+member
                                    if head<14:
                                        for row in range(L):
                                            _att_row(Q,K,V,A,T,Stats,b*L+row,head,head//7,KL-L+row,
                                                     b*KL,L,KL,14,2,64,PB,CUT,True,STATS,SLICED,False,HOIST,LOCAL)
                        tl.debug_barrier()
                        tl.atomic_add(Count+t,1,sem="relaxed",scope="gpu")
                        if INSTRUMENT:
                            end=tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;","=l",[],dtype=tl.int64,is_pure=False,pack=1)
                            tl.store(Trace+3*t+1,end)
                            tl.store(Trace+3*t+2,tl.program_id(0))
                        tl.atomic_xchg(State+t,2,sem="release",scope="gpu")
                        did=1
            if did==0:
                running=0

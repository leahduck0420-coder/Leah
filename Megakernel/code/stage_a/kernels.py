"""Lowerings for the Stage A graph pass, using public Triton language primitives."""
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return 0.5*x*(1.0+tl.erf(x*0.7071067811865476))


@triton.jit
def _materialize(ptr, idx, value, valid):
    tl.store(ptr+idx,value,valid,cache_modifier=".cg")
    tl.debug_barrier()
    return tl.load(ptr+idx,valid,other=0,volatile=True)


@triton.jit
def element(X,B,R,Y,T,N:tl.constexpr,TILE:tl.constexpr,LOCAL:tl.constexpr,
            SHIFT:tl.constexpr,C:tl.constexpr,STRIDE:tl.constexpr):
    j=tl.arange(0,TILE)
    idx=tl.program_id(0)*TILE+j
    perm=j^SHIFT
    target=tl.program_id(0)*TILE+perm
    x=tl.load(X+idx*STRIDE,idx<N,0).to(tl.float32)+tl.load(B+idx,idx<N,0)
    if LOCAL:
        if SHIFT==0:
            y=x
        else:
            y=tl.gather(x,perm,0)
    else:
        tl.store(T+idx,x,idx<N)
        tl.debug_barrier()
        y=tl.load(T+target,target<N,other=0,volatile=True)
    for c in tl.static_range(C):
        r=tl.load(R+c*N+target,target<N,other=0)
        tl.store(Y+c*N+target,_gelu(y)+r,target<N)


@triton.jit
def gemm(X,W,B,Y,T,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,
         BN:tl.constexpr,BK:tl.constexpr,LOCAL:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kr=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for b in range(tl.cdiv(K,BK)):
        k=b*BK+kr
        x=tl.load(X+row[:,None]*K+k[None,:],(row[:,None]<M)&(k[None,:]<K),0)
        w=tl.load(W+k[:,None]*N+col[None,:],(k[:,None]<K)&(col[None,:]<N),0)
        acc+=tl.dot(x,w)
    mask=(row[:,None]<M)&(col[None,:]<N)
    idx=row[:,None]*N+col[None,:]
    if not LOCAL:
        acc=_materialize(T,idx,acc,mask)
    val=acc+tl.load(B+col,col<N,0)[None,:]
    tl.store(Y+idx,_gelu(val),mask)


@triton.jit
def norm(X,G,V,Y,T,Stats,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr,C:tl.constexpr,
         GROUP:tl.constexpr,LAYER:tl.constexpr,LINEAR:tl.constexpr,LOCAL:tl.constexpr,REBUILD:tl.constexpr):
    col=tl.arange(0,B)
    for t in tl.static_range(GROUP):
        row=tl.program_id(0)*GROUP+t
        mask=(row<M)&(col<K)
        idx=row*K+col
        x=tl.load(X+idx,mask,0).to(tl.float32)
        center=tl.sum(x,0)/K if LAYER else 0.
        v=tl.where(col<K,x-center,0.)
        inv=tl.rsqrt(tl.sum(v*v,0)/K+1.e-6)
        g=tl.load(G+col,col<K,0).to(tl.float32)
        if REBUILD:
            tl.store(Stats+row,inv,row<M)
            if LAYER:
                tl.store(Stats+M+row,center,row<M)
            tl.debug_barrier()
        else:
            h=(v*inv*g).to(X.dtype.element_ty)
            if not LOCAL:
                h=_materialize(T,idx,h,mask)
        for c in tl.static_range(C):
            if REBUILD:
                raw=tl.load(X+idx,mask,0,volatile=True).to(tl.float32)
                mu=tl.load(Stats+M+row,row<M,0,volatile=True) if LAYER else 0.
                scale=tl.load(Stats+row,row<M,0,volatile=True)
                value=((raw-mu)*scale*g).to(X.dtype.element_ty).to(tl.float32)
            else:
                value=h.to(tl.float32)
            out=value*tl.load(V+c*K+col,col<K,0).to(tl.float32)
            if LINEAR:
                tl.store(Y+c*M+row,tl.sum(tl.where(mask,out,0),0),row<M)
            else:
                tl.store(Y+c*M*K+idx,out,mask)


@triton.jit
def soft(X,V,Y,T,Stats,K:tl.constexpr,B:tl.constexpr,C:tl.constexpr,WEIGHTED:tl.constexpr,
         REBUILD:tl.constexpr,CUT:tl.constexpr):
    row=tl.program_id(0)
    rows=tl.num_programs(0)
    j=tl.arange(0,B)
    x=tl.load(X+row*K+j,j<K,float("-inf")).to(tl.float32)
    m=tl.max(x,0)
    l=tl.sum(tl.exp(x-m),0)
    if REBUILD:
        tl.store(Stats+row,m)
        tl.store(Stats+rows+row,l)
        tl.debug_barrier()
    elif CUT==0:
        p=_materialize(T,row*K+j,(tl.exp(x-m)/l).to(X.dtype.element_ty),j<K)
    for c in tl.static_range(C):
        if CUT:
            acc=0.
            for i in range(tl.cdiv(K,CUT)):
                k=i*CUT+tl.arange(0,CUT)
                src=tl.load(X+row*K+k,k<K,float("-inf")).to(tl.float32)
                prob=(tl.exp(src-m)/l).to(X.dtype.element_ty)
                prob=_materialize(T,row*K+k,prob,k<K)
                w=tl.load(V+c*K+k,k<K,0).to(tl.float32)
                for unit in tl.static_range(CUT//32):
                    lanes=unit*32+tl.arange(0,32)
                    acc+=tl.sum(tl.gather(prob,lanes,0).to(tl.float32)*tl.gather(w,lanes,0),0)
            tl.store(Y+c*rows+row,acc)
        else:
            if REBUILD:
                raw=tl.load(X+row*K+j,j<K,float("-inf"),volatile=True).to(tl.float32)
                peak=tl.load(Stats+row,volatile=True)
                den=tl.load(Stats+rows+row,volatile=True)
                prob2=(tl.exp(raw-peak)/den).to(X.dtype.element_ty)
            else:
                prob2=p
            v=tl.load(V+c*K+j,j<K,0).to(tl.float32)*prob2.to(tl.float32)
            if WEIGHTED:
                acc2=0.
                for unit in range(tl.cdiv(K,32)):
                    lanes=unit*32+tl.arange(0,32)
                    acc2+=tl.sum(tl.gather(v,lanes,0),0)
                tl.store(Y+c*rows+row,acc2)
            else:
                tl.store(Y+c*rows*K+row*K+j,v,j<K)


@triton.jit
def _att_row(Q,K,V,Y,Scratch,Stats,token,head,kvhead,absolute,base,QL:tl.constexpr,
             KL:tl.constexpr,H:tl.constexpr,KV:tl.constexpr,D:tl.constexpr,
             B:tl.constexpr,CHUNK:tl.constexpr,CAUSAL:tl.constexpr,STATS:tl.constexpr,SLICED:tl.constexpr,
             ONLINE:tl.constexpr=False):
    # Bounded scratch per query, typed FP32 logits and a dtype-rounded probability plane.
    d=tl.arange(0,D)
    qoff=(token*H+head)*D
    q=tl.load(Q+qoff+d).to(tl.float32)
    offset=(token*H+head)*KL
    for ix in range(tl.cdiv(KL,CHUNK)):
        s=ix*CHUNK+tl.arange(0,CHUNK)
        mat=tl.load(K+((base+s[:,None])*KV+kvhead)*D+d[None,:],s[:,None]<KL,0).to(tl.float32)
        logit=tl.sum(mat*q[None,:],1)*(1./tl.sqrt(float(D)))
        if CAUSAL:
            logit=tl.where(s<=absolute,logit,float("-inf"))
        tl.store(Scratch+offset+s,logit,s<KL)
    tl.debug_barrier()
    ss=tl.arange(0,B)
    logits=tl.load(Scratch+offset+ss,ss<KL,float("-inf"),volatile=True)
    if ONLINE:
        # Both paired variants use the same fixed-width, ordered online normalizer.
        peak=float("-inf")
        den=0.
        for block in range(tl.cdiv(KL,32)):
            columns=block*32+tl.arange(0,32)
            z32=tl.load(Scratch+offset+columns,columns<KL,float("-inf"),volatile=True)
            updated=tl.maximum(peak,tl.max(z32,0))
            den=den*tl.exp(peak-updated)+tl.sum(tl.exp(z32-updated),0)
            peak=updated
    else:
        peak=tl.max(logits,0)
        den=tl.sum(tl.exp(logits-peak),0)
    if STATS:
        tl.store(Stats+(token*H+head)*2,peak)
        tl.store(Stats+(token*H+head)*2+1,den)
        tl.debug_barrier()
    elif not SLICED:
        tl.store(Scratch+offset+ss,(tl.exp(logits-peak)/den).to(Q.dtype.element_ty).to(tl.float32),ss<KL)
        tl.debug_barrier()
    acc=tl.full((D,),0,tl.float32)
    for ix in range(tl.cdiv(KL,CHUNK)):
        s=ix*CHUNK+tl.arange(0,CHUNK)
        if STATS:
            m=tl.load(Stats+(token*H+head)*2,volatile=True)
            l=tl.load(Stats+(token*H+head)*2+1,volatile=True)
            z=tl.load(Scratch+offset+s,s<KL,float("-inf"),volatile=True)
            p=(tl.exp(z-m)/l).to(Q.dtype.element_ty).to(tl.float32)
        elif SLICED:
            z=tl.load(Scratch+offset+s,s<KL,float("-inf"),volatile=True)
            p=(tl.exp(z-peak)/den).to(Q.dtype.element_ty).to(tl.float32)
            p=_materialize(Scratch,offset+s,p,s<KL)
        else:
            p=tl.load(Scratch+offset+s,s<KL,0,volatile=True)
        for inner in tl.static_range(CHUNK//32):
            sub=inner*32+tl.arange(0,32)
            pos=ix*CHUNK+sub
            v=tl.load(V+((base+pos[:,None])*KV+kvhead)*D+d[None,:],pos[:,None]<KL,0).to(tl.float32)
            pp=tl.gather(p,sub,0)
            acc+=tl.sum(v*pp[:,None],0)
    tl.store(Y+qoff+d,acc)


@triton.jit
def attention(Q,K,V,Y,T,Stats,L:tl.constexpr,D:tl.constexpr,B:tl.constexpr,CUT:tl.constexpr,
              CAUSAL:tl.constexpr,REBUILD:tl.constexpr,SLICED:tl.constexpr,ONLINE:tl.constexpr):
    row=tl.program_id(0)
    _att_row(Q,K,V,Y,T,Stats,row,0,0,row%L,row//L*L,L,L,1,1,D,B,CUT,CAUSAL,REBUILD,SLICED,ONLINE)

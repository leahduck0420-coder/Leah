"""Independent active-domain checks shared by CPU tests and A1."""
from .compiler import tile_map,handoff,check_slices,slices,levels
from .util import Rejected


def validate_stats(kind,inputs_live,reduction_complete,saved,required):
    expected={"rms":1,"layer":2,"softmax":2}.get(kind)
    if not inputs_live or not reduction_complete or expected!=saved or not required:
        raise Rejected("INVALID_REJECTED","statistic type, lifetime, rounding or completeness")


def regression():
    rows=[]
    def check(strategy,name,func,invalid=False):
        try:
            value=func()
            ok=value is not False and not invalid
            observed="PASS" if ok else "FAIL_CORRECTNESS"
        except Rejected as e:
            ok=invalid and e.status=="INVALID_REJECTED"
            observed=e.status
        rows.append(dict(strategy=strategy,name=name,status=observed,expected_reject=invalid,passed=ok))
    def cover(n,small,factor):
        before,after=tile_map(n,small,"bf16"),tile_map(n,small*factor,"bf16")
        for x in range(n):
            if before.owners((x,))!=((x//small,),) or after.owners((x,))!=((x//(small*factor),),):return False
        return sum(len(after.block_of((p,))) for p in range(after.grid[0]))==n
    for tile in (16,32,64):
        a=tile_map(137,tile,"float16")
        check("S1","same owner tail "+str(tile),lambda:handoff(a,a,a.local_mask) is not None)
        check("S1","value version expires "+str(tile),lambda:handoff(a,a,a.local_mask,False),True)
        check("S1","nonlocal coordinate "+str(tile),lambda:handoff(a,a,tile),True)
        for f in (2,4,8):
            check("S2","tail coverage "+str((tile,f)),lambda:cover(137,tile,f))
    for kind,n in (("rms",1),("layer",2),("softmax",2)):
        check("S3",kind,lambda:validate_stats(kind,True,True,n,True))
        check("S3",kind+" old input overwritten",lambda:validate_stats(kind,False,True,n,True),True)
        check("S3",kind+" partial",lambda:validate_stats(kind,True,False,n,True),True)
    check("S4","ordered tail",lambda:check_slices(slices(137,32),137))
    for parts in ([(0,4),(3,7)],[(0,3),(4,7)],[(3,7),(0,3)]):
        check("S4","reject partition "+str(parts),lambda:check_slices(parts,7),True)
    check("S2","contributions preserved after quotient/remainder regrouping",
          lambda:all(sorted((q%4,q//4) for q in range(n))==sorted((b,g) for g in range((n+3)//4) for b in range(4) if 4*g+b<n) for n in (7,15,33)))
    check("S2","duplicate contribution",lambda:check_slices([(0,3),(2,4)],4),True)
    check("S5","complete fork/join",lambda:levels([[],[0],[0],[1,2]])==[1,2,2,3])
    check("S5","cycle",lambda:levels([[1],[0]]),True)
    return rows

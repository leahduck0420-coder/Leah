"""Remote numerical checks, including batched attention and hybrid_k."""
from .specs import Case
from .policy import candidates


def main():
    from .runtime import environment,make_inputs,Plan,torch_operation,validate
    print(environment())
    cases=[Case('gemm',17,19,33),Case('gemm',129,131,1025),
           Case('linear_post',129,257,1027,ep='bias_gelu_residual'),
           Case('qkt',129,131,129,batch=4),Case('pv',129,131,129,batch=4),
           Case('moe',129,257,769,tokens=(0,1,32,129))]
    for case in cases:
        groups=make_inputs(case);ref=torch_operation(case,True)(groups)
        for cfg in candidates(case)[0]:
            plan=Plan(case,groups,config=cfg).prepare()
            validate(plan(),ref)
        print('PASS',case.key,flush=True)


if __name__=='__main__':main()

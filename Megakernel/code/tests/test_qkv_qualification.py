"""Qualify arithmetic modes per projection without shape or strategy rules."""
from types import SimpleNamespace
import unittest
import numpy as np

from stage_a.numeric import choose_bias_rounding
from stage_a.workloads import Workload,projection_with_bias_reference


class Tensor:
    def __init__(self,values,role=None):
        self.values=np.asarray(values)
        self.role=role

    def float(self):
        return Tensor(self.values.astype(np.float32),self.role)

    def to(self,dtype):
        return Tensor(self.values.astype(dtype),self.role)

    @property
    def T(self):
        return Tensor(self.values.T,self.role)

    def __matmul__(self,other):
        return Tensor(self.values@other.values)

    def __add__(self,other):
        return Tensor(self.values+other.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def job_for_modes(roles):
    def linear(x,weight,bias):
        return projection_with_bias_reference(x,weight,bias,np.float16,roles[weight.role])
    job=object.__new__(Workload)
    job.t=SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)))
    job.dt=np.float16
    job.x=Tensor(np.array([[[1.0009765625]]],dtype=np.float16))
    job.ws={}
    for name in ("q","k","v"):
        job.ws["self_attn."+name+"_proj.weight"]=Tensor(np.array([[1.0009765625]],dtype=np.float16),name)
        job.ws["self_attn."+name+"_proj.bias"]=Tensor(np.array([.00048828125],dtype=np.float16))
    job.input_contract={"profile":"normal"}
    job.qkv_bias_mode=None
    return job


class QKVQualificationTests(unittest.TestCase):
    def test_roles_can_require_different_rounding_boundaries(self):
        job=job_for_modes(dict(q=False,k=True,v=False))
        result=job.qualify_qkv_projection()
        self.assertEqual(result["mode"],2)
        self.assertEqual(job.qkv_bias_mode,2)
        self.assertEqual(result["roles"]["k"]["round_before_bias_mse"],0.)
        self.assertEqual(result["roles"]["q"]["single_round_mse"],0.)

    def test_all_single_and_all_staged_are_qualified_by_values(self):
        for staged,mode in ((False,0),(True,7)):
            job=job_for_modes(dict(q=staged,k=staged,v=staged))
            self.assertEqual(job.qualify_qkv_projection()["mode"],mode)

    def test_tied_modes_prefer_single_round(self):
        result=choose_bias_rounding([1.,2.],[1.,2.],[1.,2.])
        self.assertFalse(result["round_before_bias"])
        self.assertTrue(result["single_round_exact"])

    def test_invalid_probe_cannot_be_qualified(self):
        for native,single,staged in (([float("nan")],[1.],[1.]),([1.,2.],[1.],[1.])):
            with self.assertRaises(ValueError):
                choose_bias_rounding(native,single,staged)


if __name__=="__main__":
    unittest.main()

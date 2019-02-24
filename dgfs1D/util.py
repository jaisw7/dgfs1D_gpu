# -*- coding: utf-8 -*-
import pycuda.driver as cuda
from pycuda import compiler
from gimmik import generate_mm
import numpy as np

np_map = {'float' : np.float32, 'double' : np.float64}
np_rmap = {np.float32 : 'float', np.float64: 'double'}

def get_kernel(module, funcname, args):
    func = module.get_function(funcname)
    func.prepare(args)
    func.set_cache_config(cuda.func_cache.PREFER_L1)
    return func

def filter_tol(mat, tol=1e-15, val=0.):
    mat[abs(mat)<tol] = val
    return mat

def get_mm_kernel(mat, alpha=1., beta=0., tol=1e-15):
    matSrc = generate_mm(filter_tol(mat, tol=tol), dtype=mat.dtype, 
        platform='cuda', alpha=alpha, beta=beta)
    matMod = compiler.SourceModule(matSrc)
    matKern = get_kernel(matMod, "gimmik_mm", 'iPiPi')
    return matKern  

def get_mm_proxycopy_kernel(mat, alpha=1., beta=0., tol=1e-15):
    class generate_mm_proxycopy():
        def prepared_call(*x): 
            cuda.memcpy_dtod(x[5], x[3], x[2]*mat.shape[0]*mat.dtype.itemsize)
    return generate_mm_proxycopy  

def cross(args):
    return it.product(*args)

def get_kernel_op(module, names, pointers):
    return map(lambda v: 
        lambda *args: get_kernel(module, v[0], v[1]).prepared_call(
            grid_Nv, block, *list(map(lambda c: c.ptr, args))
        ), zip(names, pointers)
    )


def check(truth_value, *args):
    if not truth_value: raise ValueError(*args) 
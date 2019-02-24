import numpy as np
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup)
from dgfs1D.bi.initcond import DGFSInitConditionBi
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.util import get_kernel

class DGFSBCBi():
    type = None

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        self._vm = vm
        self._updateBCKern = [lambda *args: None for p in range(vm.nspcs())]
        self._applyBCKern = [lambda *args: None for p in range(vm.nspcs())]

    @property
    def updateBCKern(self): return self._updateBCKern
    
    @property
    def applyBCKern(self): return self._applyBCKern


# enforces the purely diffuse wall boundary condition
class DGFSWallDiffuseBCBi(DGFSBCBi):
    type = 'dgfs-wall-diffuse'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        initcondcls = subclass_where(DGFSInitConditionBi, model='maxwellian')
        bc = initcondcls(cfg, self._vm, cfgsect, wall=True)
        f0 = bc.get_init_vals()
        self._d_bnd_f0 = [gpuarray.to_gpu(f.ravel()) for f in f0]
        unondim = bc.unondim()

        # storage
        self._bc_vals_num = [gpuarray.empty(self._vm.vsize(), 
            self._d_bnd_f0[0].dtype) for p in range(vm.nspcs())]
        self._bc_vals_den = [gpuarray.empty(self._vm.vsize(), 
            self._d_bnd_f0[0].dtype) for p in range(vm.nspcs())]
        self._wall_nden = [gpuarray.empty(1, dtype=self._d_bnd_f0[0].dtype) 
                                for p in range(vm.nspcs())]

        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    ux=unondim[0,0], nl=nl, x=xsol
                )
        kernsrc = DottedTemplateLookup('dgfs1D.bi.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # for applying the boundary condition
        def make_applyBC(p, applyBCFunc):
            def applyBC(ul, ur, t):
                applyBCFunc.prepared_call(
                    grid_Nv, block, 
                    ul.ptr, ur.ptr, self._vm.d_cvx().ptr, 
                    self._d_bnd_f0[p].ptr, self._wall_nden[p].ptr, t
                )
            return applyBC

        applyBCFunc = get_kernel(kernmod, "applyBC", 
                [np.intp]*5+[unondim.dtype])
        for p in range(vm.nspcs()):
            self._applyBCKern[p] = make_applyBC(p, applyBCFunc)
    
        # for extracting left face values
        def make_updateBC(p, updateBCFunc):
            def updateBC(ul, t):
                updateBCFunc.prepared_call(
                    grid_Nv, block, ul.ptr, self._vm.d_cvx().ptr, 
                    self._d_bnd_f0[p].ptr, self._bc_vals_num[p].ptr, 
                    self._bc_vals_den[p].ptr, t
                )
                self._wall_nden[p] = -(gpuarray.sum(self._bc_vals_num[p])
                    /gpuarray.sum(self._bc_vals_den[p])
                )
            return updateBC

        updateBCFunc = get_kernel(kernmod, "updateBC", 
                                    [np.intp]*5+[unondim.dtype])
        for p in range(vm.nspcs()):
            self._updateBCKern[p] = make_updateBC(p, updateBCFunc)            


# enforces the periodic
class DGFSPeriodicBCBi(DGFSBCBi):
    type = 'dgfs-periodic'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    nl=nl, x=xsol
                )
        kernsrc = DottedTemplateLookup('dgfs1D.bi.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # copy the left face values to the right
        applyBCFunc = get_kernel(kernmod, "applyBC", 'PP')
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, 
                                ul.ptr, ur.ptr
                            )  
        self._applyBCKern = [self._applyBCKern]*vm.nspcs() 


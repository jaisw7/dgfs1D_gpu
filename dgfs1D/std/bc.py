import numpy as np
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup)
from dgfs1D.std.initcond import DGFSInitConditionStd
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.util import get_kernel

class DGFSBCStd():
    type = None

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        self._vm = vm
        self._updateBCKern = lambda *args: None
        self._applyBCKern = lambda *args: None

    @property
    def updateBCKern(self): return self._updateBCKern
    
    @property
    def applyBCKern(self): return self._applyBCKern


# enforces the purely diffuse wall boundary condition
class DGFSWallDiffuseBCStd(DGFSBCStd):
    type = 'dgfs-wall-diffuse'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        initcondcls = subclass_where(DGFSInitConditionStd, model='maxwellian')
        bc = initcondcls(cfg, self._vm, cfgsect, wall=True)
        f0 = bc.get_init_vals().reshape(self._vm.vsize(), 1)
        self._d_bnd_f0 = gpuarray.to_gpu(f0)
        unondim = bc.unondim()

        # storage
        self._bc_vals_num = gpuarray.empty(self._vm.vsize(), 
            self._d_bnd_f0.dtype)
        self._bc_vals_den = gpuarray.empty_like(self._bc_vals_num)

        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    ux=unondim[0,0], nl=nl, x=xsol
                )
        kernsrc = DottedTemplateLookup('dgfs1D.std.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # for extracting right face values
        applyBCFunc = get_kernel(kernmod, "applyBC", 
            [np.intp]*5+[unondim.dtype])
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, 
                                ul.ptr, ur.ptr, self._vm.d_cvx().ptr, 
                                self._d_bnd_f0.ptr, self._wall_nden.ptr, t
                            )
        
        # for extracting left face values
        updateBCFunc = get_kernel(kernmod, "updateBC", 
            [np.intp]*5+[unondim.dtype])
        def updateBC(ul, t):
            updateBCFunc.prepared_call(
                grid_Nv, block, ul.ptr, self._vm.d_cvx().ptr, 
                self._d_bnd_f0.ptr, self._bc_vals_num.ptr, 
                self._bc_vals_den.ptr, t
            )
            self._wall_nden = -(gpuarray.sum(self._bc_vals_num)
                /gpuarray.sum(self._bc_vals_den)
            )

        self._updateBCKern = updateBC


# enforces the periodic
class DGFSPeriodicBCStd(DGFSBCStd):
    type = 'dgfs-periodic'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    nl=nl, x=xsol
                )
        kernsrc = DottedTemplateLookup('dgfs1D.std.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # copy the left face values to the right
        applyBCFunc = get_kernel(kernmod, "applyBC", 'PP')
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, 
                                ul.ptr, ur.ptr,  
                            )   

# enforces the purely diffuse wall BC with variable velocity/temperature
class DGFSWallExprDiffuseBCStd(DGFSBCStd):
    type = 'dgfs-wall-expr-diffuse'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        #initcondcls = subclass_where(DGFSInitConditionStd, model='maxwellian')
        #bc = initcondcls(cfg, self._vm, cfgsect, wall=True)
        #f0 = bc.get_init_vals().reshape(self._vm.vsize(), 1)
        #self._d_bnd_f0 = gpuarray.to_gpu(f0)
        #unondim = bc.unondim()
        rhoini = 1.
        ux = cfg.lookupexpr(cfgsect, 'ux')
        uy = cfg.lookupexpr(cfgsect, 'uy')
        uz = cfg.lookupexpr(cfgsect, 'uz')
        T = cfg.lookupexpr(cfgsect, 'T')
        ux = '((' + ux + ')/' + str(self._vm.u0()) + ')'
        uy = '((' + uy + ')/' + str(self._vm.u0()) + ')'
        uz = '((' + uz + ')/' + str(self._vm.u0()) + ')'
        T = '((' + T + ')/' + str(self._vm.T0()) + ')'

        # storage
        self._bc_vals_num = gpuarray.empty(self._vm.vsize(), cfg.dtype)
        self._bc_vals_den = gpuarray.empty_like(self._bc_vals_num)

        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    nl=nl, x=xsol,
                    ux=ux, uy=uy, uz=uz, T=T
                )
        kernsrc = DottedTemplateLookup('dgfs1D.std.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # for extracting right face values
        applyBCFunc = get_kernel(kernmod, "applyBC", 
            [np.intp]*6+[cfg.dtype])
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, 
                                ul.ptr, ur.ptr, self._vm.d_cvx().ptr,
                                self._vm.d_cvy().ptr, self._vm.d_cvz().ptr, 
                                self._wall_nden.ptr, t
                            )
        
        # for extracting left face values
        updateBCFunc = get_kernel(kernmod, "updateBC", 
            [np.intp]*6+[cfg.dtype])
        def updateBC(ul, t):
            updateBCFunc.prepared_call(
                grid_Nv, block, ul.ptr, self._vm.d_cvx().ptr, 
                self._vm.d_cvy().ptr, self._vm.d_cvz().ptr, 
                self._bc_vals_num.ptr, self._bc_vals_den.ptr, t
            )
            self._wall_nden = -(gpuarray.sum(self._bc_vals_num)
                /gpuarray.sum(self._bc_vals_den)
            )

        self._updateBCKern = updateBC


# enforces the inlet boundary condition
class DGFSInletBCStd(DGFSBCStd):
    type = 'dgfs-inlet'

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(xsol, nl, vm, cfg, cfgsect, **kwargs)

        initcondcls = subclass_where(DGFSInitConditionStd, model='maxwellian')
        bc = initcondcls(cfg, self._vm, cfgsect, wall=False)
        f0 = bc.get_init_vals().reshape(self._vm.vsize(), 1)
        self._d_bnd_f0 = gpuarray.to_gpu(f0)
        unondim = bc.unondim()

        # template
        dfltargs = dict(dtype=cfg.dtypename, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    nl=nl, x=xsol
                )
        kernsrc = DottedTemplateLookup('dgfs1D.std.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, self._vm.vsize())

        # for extracting right face values
        applyBCFunc = get_kernel(kernmod, "applyBC", 
            [np.intp]*4+[unondim.dtype])
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, 
                                ul.ptr, ur.ptr, self._vm.d_cvx().ptr, 
                                self._d_bnd_f0.ptr, t
                            )
        
        # no update
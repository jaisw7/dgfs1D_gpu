import numpy as np
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup)
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.util import get_kernel
from dgfs1D.cublas import CUDACUBLASKernels

class DGFSBCStd():
    type = None

    def __init__(self, xsol, nl, vm, cfg, cfgsect, **kwargs):
        self._vm = vm
        self.cfg = cfg
        self._updateBCKern = lambda *args: None
        self._applyBCKern = lambda *args: None

    @property
    def updateBCKern(self): return self._updateBCKern
    
    @property
    def applyBCKern(self): return self._applyBCKern

    def read_common(self, vars):
        # Get the physical location of each solution point
        vars.update(self.cfg.section_values('mesh', np.float64))
        vars.update(self.cfg.section_values('basis', np.float64))
        vars.update(self.cfg.section_values('non-dim', np.float64))
        vars.update(self.cfg.section_values('velocity-mesh', np.float64))

# enforces the purely diffuse wall BC with variable velocity/temperature
class DGFSWallExprDiffuseBCStd(DGFSBCStd):
    type = 'dgfs-wall-expr-diffuse'

    def __init__(self, d_xsol, nl, vm, cfg, cfgsect, **kwargs):
        
        super().__init__(d_xsol, nl, vm, cfg, cfgsect, **kwargs)

        rho = 1.
        ux = cfg.lookupexpr(cfgsect, 'ux')
        uy = cfg.lookupexpr(cfgsect, 'uy')
        uz = cfg.lookupexpr(cfgsect, 'uz')
        T = cfg.lookupexpr(cfgsect, 'T')
        ux = '((' + ux + ')/' + str(self._vm.u0()) + ')'
        uy = '((' + uy + ')/' + str(self._vm.u0()) + ')'
        uz = '((' + uz + ')/' + str(self._vm.u0()) + ')'
        T = '((' + T + ')/' + str(self._vm.T0()) + ')'

        self._vars = {}
        super().read_common(self._vars)
        from mako.template import Template
        ux, uy, uz, T = map(
           lambda v: Template(v).render(**self._vars), 
           (ux, uy, uz, T)
        )

        #Nb, Nqf = map(lambda v: kwargs.get(v), ("Nb", "Nqf"))
        Ndof = kwargs.get("Ndof")

        # storage
        vsize = self._vm.vsize()
        self._bc_vals_num = gpuarray.empty(vsize*Ndof, cfg.dtype)
        self._bc_vals_den = gpuarray.empty_like(self._bc_vals_num)
        self._bc_vals_num_sum = gpuarray.empty(Ndof, cfg.dtype)
        self._bc_vals_den_sum = gpuarray.empty_like(self._bc_vals_num_sum)

        # kerns for summation
        self.mat = gpuarray.to_gpu(np.ones(vsize, cfg.dtype).ravel()) # Nv x nalph
        nalph = 1
        self.blas = CUDACUBLASKernels() # blas kernels for computing moments
        self.sA_mom = (Ndof, vsize)
        self.sB_mom = (nalph, vsize)
        self.sC_mom = (Ndof, nalph)
        
        def sum(f, U):
          self.blas.mul(f, self.sA_mom, self.mat, self.sB_mom, U, self.sC_mom)

        dfltargs = dict(dtype=cfg.dtypename, dim=cfg.dim, 
                    vsize=self._vm.vsize(), cw=self._vm.cw(),
                    nl=nl, ux=ux, uy=uy, uz=uz, T=T
                )
        dfltargs.update(self._vars)
        kernsrc = DottedTemplateLookup('dgfs3D.astd.kernels.bcs', 
                    dfltargs).get_template(self.type).render()
        kernmod = compiler.SourceModule(kernsrc)

        # block size
        block = (128, 1, 1)
        grid_Nv = get_grid_for_block(block, vsize*Ndof)
        dtn = self.cfg.dtypename[0]

        # for extracting right face values
        applyBCFunc = get_kernel(kernmod, "applyBC", 'i'+'P'*8+dtn)
        self._applyBCKern = lambda ul, ur, t: applyBCFunc.prepared_call(
                                grid_Nv, block, vsize*Ndof, d_xsol.ptr, 
                                ul.ptr, ur.ptr, self._vm.d_cvx().ptr,
                                self._vm.d_cvy().ptr, self._vm.d_cvz().ptr, 
                                self._bc_vals_num_sum.ptr, 
                                self._bc_vals_den_sum.ptr, t
                            )
        
        # for extracting left face values
        updateBCFunc = get_kernel(kernmod, "updateBC", "i"+'P'*7+dtn)
        def updateBC(ul, t):
            updateBCFunc.prepared_call(
                grid_Nv, block, vsize*Ndof, d_xsol.ptr,
                ul.ptr, self._vm.d_cvx().ptr, 
                self._vm.d_cvy().ptr, self._vm.d_cvz().ptr, 
                self._bc_vals_num.ptr, self._bc_vals_den.ptr, t
            )
            sum(self._bc_vals_num, self._bc_vals_num_sum)
            sum(self._bc_vals_den, self._bc_vals_den_sum)

        self._updateBCKern = updateBC




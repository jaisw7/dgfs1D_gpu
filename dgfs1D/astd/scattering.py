from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma

# need to fix this (to make things backend independent)
from pycuda import compiler, gpuarray
from dgfs1D.nputil import DottedTemplateLookup
import pycuda.driver as cuda
from dgfs1D.nputil import get_grid_for_block
from dgfs1D.util import get_kernel
from dgfs1D.cublas import CUDACUBLASKernels

class DGFSScatteringModelAstd(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh
        self.block = (256, 1, 1)

        # a utility function for downcasting to raw ptr
        self.ptr = lambda vs: [v.ptr if (hasattr(v, 'ptr')) else v for v in vs]

        # read model parameters
        self.load_parameters()

        # perform any necessary computation
        self.perform_precomputation()
        print('Finished scattering model precomputation')


    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def perform_precomputation(self):
        pass

"""
BGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough;
the error in conservation would be spectrally low
"""
class DGFSBGKDirectGLLScatteringModelAstd(DGFSScatteringModelAstd):
    scattering_model = 'bgk-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if not (basis_kind=='nodal-sem-gll' or basis_kind=='nodal-gll' or basis_kind=='nodal-sem-gll-2'):
            raise RuntimeError("Only tested for nodal basis")
        super().__init__(cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        Pr = 1.
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)

    def nu(self, rho, T):
        return rho*T**(1-self._omega);

    def perform_precomputation(self):
        self.nalph = 5
        vm = self.vm

        # compute mat
        mat = np.vstack(
            (np.ones(vm.vsize()),
                vm.cv(),
                np.einsum('ij,ij->j', vm.cv(), vm.cv())
            )
        )*vm.cw() # 5 x Nv
        self.mat = gpuarray.to_gpu((mat).ravel()) # Nv x 5 flatenned
        self.blas = CUDACUBLASKernels() # blas kernels for computing moments

        # now load the modules
        self.load_modules()


    def load_modules(self):
        """Load modules (this must be called internally)"""

        # number of stages
        nbdf = [1, 2, 3, 4, 5, 6]; nars = [1, 2, 3, 4]

        # extract the template
        dfltargs = dict(vsize=self.vm.vsize(),
            nalph=self.nalph, omega=self._omega, Pr=self._Pr,
            dtype=self.cfg.dtypename, nbdf=nbdf, nars=nars)
        src = DottedTemplateLookup('dgfs1D.astd.kernels.scattering', dfltargs
                    ).get_template(self.scattering_model).render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # data type name prefix
        dtn = self.cfg.dtypename[0]

        # construct maxwellian given (rho, rho*ux, rho*uy, rho*uz, E)
        self.cmaxwellianKern = get_kernel(module, "cmaxwellian", 'iPPPPP')

        # construct the collision operator
        self.collideKern = get_kernel(module, "collide", dtn+'iPPPP')
        self.collideNuKern = get_kernel(module, "collide_nu", dtn+'iPPPP')

        # update the moment
        self.updateMomKernsBDF = tuple(map(
            lambda q: get_kernel(module, "updateMom{0}_BDF".format(q),
                dtn+'i'+dtn+(dtn+'P')*(2*q+1)+dtn), nbdf
        ))
        self.updateMomKernsARS = tuple(map(
            lambda q: get_kernel(module, "updateMom{0}_ARS".format(q),
                dtn+'i'+dtn+dtn*(2*q)+'P'*(2*q+1)), nars
        ))
        self.updateMomKernsLM = tuple(map(
            lambda q: get_kernel(module, "updateMom{0}_LM".format(q),
                dtn+'i'+dtn+dtn*(3*q+1)+'P'*(3*q+1)), nbdf
        ))


        # update the distribution
        self.updateDistKernsBDF = tuple(map(
            lambda q: get_kernel(module, "updateDistribution{0}_BDF".format(q),
                dtn+'i'+dtn+(dtn+'P')*(2*q+2)+'P'), nbdf
        ))
        self.updateDistNuKernsBDF = tuple(map(
            lambda q: get_kernel(module, "updateDistributionNu{0}_BDF".format(q),
                dtn+'i'+dtn+(dtn+'P')*(2*q+2)+'P'), nbdf
        ))
        self.updateDistKernsARS = tuple(map(
            lambda q: get_kernel(module, "updateDistribution{0}_ARS".format(q),
                dtn+'i'+dtn+dtn*(2*q)+'P'*(4*q+2)), nars
        ))
        self.updateDistWeightKernsSSPL = tuple(map(
            lambda q: get_kernel(module, "updateDistributionWeight{0}_SSPL".format(q),
                dtn+'i'+dtn+dtn*(2*q)+'P'*(4*q+3)), nars
        ))
        self.updateDistKernsLM = tuple(map(
            lambda q: get_kernel(module, "updateDistribution{0}_LM".format(q),
                dtn+'i'+dtn+dtn*(3*q+1)+'P'*(4*q+3)), nbdf
        ))
        self.module = module


    def moment(self, t, f, U):
        lda = f.shape[0]//self.vm.vsize()
        assert lda==U.shape[0]//self.nalph, "Some issue"

        sA_mom = (lda, self.vm.vsize())
        sB_mom = (self.nalph, self.vm.vsize())
        sC_mom = (lda, self.nalph)
        self.blas.mul(f, sA_mom, self.mat, sB_mom, U, sC_mom)


    def constructMaxwellian(self, t, U, M):
        lda = M.shape[0]//self.vm.vsize()
        assert lda==U.shape[0]//self.nalph, "Some issue"

        vm = self.vm
        grid = get_grid_for_block(self.block, lda*vm.vsize())
        self.cmaxwellianKern.prepared_call(
                    grid, self.block, lda*vm.vsize(),
                    vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr,
                    M.ptr, U.ptr)

    def collide(self, t, U, M, f, Q):
        lda = M.shape[0]//self.vm.vsize()
        assert lda==U.shape[0]//self.nalph, "Some issue"

        vm = self.vm
        grid = get_grid_for_block(self.block, lda*vm.vsize())
        self.collideKern.prepared_call(
                    grid, self.block, self._prefactor, int(lda*vm.vsize()),
                    M.ptr, U.ptr, f.ptr, Q.ptr)

    def collideNu(self, t, U, M, f, Q):
        lda = M.shape[0]//self.vm.vsize()
        assert lda==U.shape[0]//self.nalph, "Some issue"

        vm = self.vm
        grid = get_grid_for_block(self.block, lda*vm.vsize())
        self.collideNuKern.prepared_call(
                    grid, self.block, self._prefactor, int(lda*vm.vsize()),
                    M.ptr, U.ptr, f.ptr, Q.ptr)

    def updateMomentBDF(self, dt, *args):
        # the size of args should be 4*q+3 for BDF scheme
        q = (len(args) - 3)//4
        assert len(args)==4*q+3, "Inconsistency in number of parameters"

        lda = int(args[1].shape[0])//self.nalph
        grid = get_grid_for_block(self.block, lda)
        self.updateMomKernsBDF[q-1].prepared_call(grid, self.block,
                self._prefactor, lda, dt, *self.ptr(args))


    def updateDistBDF(self, dt, *args):
        # the size of args should be 4*q+5 for BDF scheme
        q = (len(args) - 5)//4
        assert len(args)==4*q+5, "Inconsistency in number of parameters"

        lda = int(args[1].shape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateDistKernsBDF[q-1].prepared_call(grid, self.block,
            self._prefactor, lda, dt, *self.ptr(args))


    def updateDistNuBDF(self, dt, *args):
        # the size of args should be 4*q+5 for BDF scheme
        q = (len(args) - 5)//4
        assert len(args)==4*q+5, "Inconsistency in number of parameters"

        lda = int(args[1].shape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateDistNuKernsBDF[q-1].prepared_call(grid, self.block,
            self._prefactor, lda, dt, *self.ptr(args))


    def updateMomentARS(self, dt, *args):
        # the size of args should be 4*q+1 for ARS scheme
        q = (len(args) - 1)//4
        assert len(args)==4*q+1, "Inconsistency in number of parameters"

        lda = int(args[-1].shape[0])//self.nalph
        grid = get_grid_for_block(self.block, lda)
        self.updateMomKernsARS[q-1].prepared_call(grid, self.block,
                self._prefactor, lda, dt, *self.ptr(args))


    def updateDistARS(self, dt, *args):
        # the size of args should be 6*q+2 for ARS scheme
        q = (len(args) - 2)//6
        assert len(args)==6*q+2, "Inconsistency in number of parameters"

        lda = int(args[-1].shape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateDistKernsARS[q-1].prepared_call(grid, self.block,
            self._prefactor, lda, dt, *self.ptr(args))


    def updateDistWeightSSPL(self, dt, *args):
        # the size of args should be 6*q+5 for ARS scheme
        q = (len(args) - 3)//6
        assert len(args)==6*q+3, "Inconsistency in number of parameters"

        lda = int(args[-1].shape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateDistWeightKernsSSPL[q-1].prepared_call(grid, self.block,
            self._prefactor, lda, dt, *self.ptr(args))


    def updateMomentLM(self, dt, A, B, C, LU, LM, U, nstages):
        lda = int(U[0].shape[0])//self.nalph
        grid = get_grid_for_block(self.block, lda)
        coeffs = A + B + C
        args = LU + LM + U
        #print(len(coeffs)+len(args)); exit()
        self.updateMomKernsLM[nstages-1].prepared_call(grid, self.block,
                self._prefactor, lda, dt, *coeffs, *self.ptr(args))


    def updateDistLM(self, dt, A, B, C, L, M, F, fnew, U, nstages):
        lda = int(F[0].shape[0])
        grid = get_grid_for_block(self.block, lda)
        coeffs = A + B + C
        args = L + M + F + [fnew] + U
        #print(len(coeffs)+len(args)); exit()
        self.updateDistKernsLM[nstages-1].prepared_call(grid, self.block,
            self._prefactor, lda, dt, *coeffs, *self.ptr(args))


"""
ESBGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough;
the error in conservation would be spectrally low
"""
class DGFSESBGKDirectGLLScatteringModelAstd(DGFSBGKDirectGLLScatteringModelAstd):
    scattering_model = 'esbgk-direct-gll'

    def load_parameters(self):
        Pr = self.cfg.lookupordefault('scattering-model', 'Pr', 2./3.);
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)


    def perform_precomputation(self):
        self.nalph = 11
        vm = self.vm

        # compute mat
        cv = vm.cv()
        mat = np.vstack(
            (np.ones(vm.vsize()), # mass
            cv, # momentum
            np.einsum('ij,ij->j', vm.cv(), vm.cv()), # energy
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:] # off-diag
        ))*vm.cw() # 11 x Nv
        self.mat = gpuarray.to_gpu((mat).ravel()) # Nv x 11 flatenned
        self.blas = CUDACUBLASKernels() # blas kernels for computing moments

        # now load the modules
        self.load_modules()

        # construct (rho, u, T, P) given (rho, rho*u, E, P)
        self.momentNormKern = get_kernel(self.module, "momentNorm", 'iP')
        self.normU = None


    def constructMaxwellian(self, t, U, M):
        lda = M.shape[0]//self.vm.vsize()
        assert lda==U.shape[0]//self.nalph, "Some issue"

        vm = self.vm

        if not self.normU: self.normU = gpuarray.empty_like(U)
        cuda.memcpy_dtod(self.normU.ptr, U.ptr, U.nbytes)

        grid = get_grid_for_block(self.block, lda)
        self.momentNormKern.prepared_call(grid, self.block, lda, self.normU.ptr)

        grid = get_grid_for_block(self.block, lda*vm.vsize())
        self.cmaxwellianKern.prepared_call(
                    grid, self.block, lda*vm.vsize(),
                    vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr,
                    M.ptr, self.normU.ptr)



"""
Shakov "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough;
the error in conservation would be spectrally low
"""
class DGFSShakovDirectGLLScatteringModelAstd(DGFSESBGKDirectGLLScatteringModelAstd):
    scattering_model = 'shakov-direct-gll'

    def load_parameters(self):
        Pr = self.cfg.lookupordefault('scattering-model', 'Pr', 2./3.);
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)


    def perform_precomputation(self):
        self.nalph = 14
        vm = self.vm

        # compute mat
        cv = vm.cv()
        mat = np.vstack(
            (np.ones(vm.vsize()), # mass
            cv, # momentum
            np.einsum('ij,ij->j', vm.cv(), vm.cv()), # energy
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:], # off-diag
            np.einsum('ij,ij->j', cv, cv)*cv[0,:], # x-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[1,:], # y-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[2,:] # z-heat-flux
        ))*vm.cw() # 14 x Nv
        self.mat = gpuarray.to_gpu((mat).ravel()) # Nv x 11 flatenned
        self.blas = CUDACUBLASKernels() # blas kernels for computing moments

        # now load the modules
        self.load_modules()

        # construct (rho, u, T, P) given (rho, rho*u, E, P)
        self.momentNormKern = get_kernel(self.module, "momentNorm", 'iP')
        self.normU = None



class DGFSBoltzBGKDirectGLLScatteringModelAstd(DGFSBGKDirectGLLScatteringModelAstd):
    scattering_model = 'boltz-bgk-direct-gll-v1'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._eps = kwargs.get('eps')
        super().__init__(cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        # now the penalization operator
        Pr = 1.
        omega = 0.5;
        #self._pprefactor = (2**2.5)/np.sqrt(np.pi)*(self.vm.dev()**4)
        self._pprefactor = 16*np.pi #*self.vm.dev()

        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');
        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press
        self._prefactor = (t0*Pr*p0/visc)*4*np.pi

        self._omega = omega
        self._Pr = Pr
        #self._prefactor = self._eps*self._pprefactor
        print("penalized-prefactor:", self._prefactor)



class DGFSBoltzBGKDirectGLLScatteringModelAstd(DGFSBGKDirectGLLScatteringModelAstd):
    scattering_model = 'boltz-bgk-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._eps = kwargs.get('eps')
        super().__init__(cfg, velocitymesh, **kwargs)


    def load_parameters(self):
        alpha = 1.0
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        self._gamma = 2.0*(1-omega)

        dRef = self.cfg.lookupfloat('scattering-model', 'dRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        invKn = self.vm.H0()*np.sqrt(2.0)*np.pi*self.vm.n0()*dRef*dRef*pow(
            Tref/self.vm.T0(), omega-0.5);

        self._prefactor = 100*invKn*alpha/(
            pow(2.0, 2-omega+alpha)*gamma(2.5-omega)*np.pi);
        self._omega = omega
        self._Pr = 1
        #super().load_parameters()
        print("penalized-prefactor:", self._prefactor)


    def load_parameters__(self):
        # now the penalization operator
        Pr = 1.
        omega = 0.5;
        #self._pprefactor = (2**2.5)/np.sqrt(np.pi)*(self.vm.dev()**4)
        self._pprefactor = 16*np.pi #*self.vm.dev()

        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');
        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press
        #self._prefactor = (t0*Pr*p0/visc)*4*np.pi
        self._prefactor = (t0*Pr*p0/visc)

        self._omega = omega
        self._Pr = Pr
        #self._prefactor = self._eps*self._pprefactor
        print("penalized-prefactor:", self._prefactor)



class DGFSBoltzESBGKDirectGLLScatteringModelAstd(DGFSESBGKDirectGLLScatteringModelAstd):
    scattering_model = 'boltz-esbgk-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._eps = kwargs.get('eps')
        super().__init__(cfg, velocitymesh, **kwargs)

    def _load_parameters(self):
        # now the penalization operator
        Pr = 2./3.
        #omega = 0.5;
        #self._pprefactor = 8*np.pi #(2**2.5)/np.sqrt(np.pi)*(self.vm.dev()**4)

        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');
        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press
        self._prefactor = (t0*Pr*p0/visc)

        #self._pprefactor = 4*np.pi
        self._omega = omega
        self._Pr = Pr
        #self._prefactor = self._eps*self._pprefactor
        print("penalized-prefactor:", self._prefactor)



class DGFSBoltzShakovDirectGLLScatteringModelAstd(DGFSShakovDirectGLLScatteringModelAstd):
    scattering_model = 'boltz-shakov-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._eps = kwargs.get('eps')
        super().__init__(cfg, velocitymesh, **kwargs)

    def _load_parameters(self):
        # now the penalization operator
        Pr = 2./3.
        omega = -1;
        self._pprefactor = (2**2.5)/np.sqrt(np.pi)*(self.vm.dev()**4)

        self._omega = omega
        self._Pr = Pr
        self._prefactor = self._eps*143.2601955378592 #self._pprefactor
        print("penalized-prefactor:", self._prefactor)




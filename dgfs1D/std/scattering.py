from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma, isnan, ceil

# need to fix this (to make things backend independent)
from pycuda import compiler, gpuarray
from dgfs1D.nputil import DottedTemplateLookup
from dgfs1D.cufft import (cufftPlan3d, cufftPlanMany, 
                            cufftExecD2Z, cufftExecZ2Z, cufftExecC2C,
                            CUFFT_D2Z, CUFFT_Z2Z, CUFFT_C2C, 
                            CUFFT_FORWARD, CUFFT_INVERSE
                        )
import pycuda.driver as cuda
from dgfs1D.nputil import get_grid_for_block
from dgfs1D.util import get_kernel
from dgfs1D.cublas import CUDACUBLASKernels
from dgfs1D.cusolver import CUDACUSOLVERKernels

class DGFSScatteringModelStd(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh

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

    @abstractmethod 
    def fs(self, d_arr_in, d_arr_out, elem, upt):
        pass

# Simplified VHS model for GLL based nodal collocation schemes
class DGFSVHSGLLScatteringModelStd(DGFSScatteringModelStd):
    scattering_model = 'vhs-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if basis_kind!='nodal-sem-gll':
            raise RuntimeError("Only tested for nodal basis")
        super().__init__(cfg, velocitymesh, **kwargs)
        
    def load_parameters(self):
        alpha = 1.0
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        self._gamma = 2.0*(1-omega)

        dRef = self.cfg.lookupfloat('scattering-model', 'dRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        invKn = self.vm.H0()*np.sqrt(2.0)*np.pi*self.vm.n0()*dRef*dRef*pow(
            Tref/self.vm.T0(), omega-0.5);

        self._prefactor = invKn*alpha/(
            pow(2.0, 2-omega+alpha)*gamma(2.5-omega)*np.pi);

        print("Kn:", 1.0/invKn)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        N = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()

        l0 = np.concatenate((np.arange(0,N/2), np.arange(-N/2, 0)))
        #l = l0[np.mgrid[0:N, 0:N, 0:N]]
        #l = l.reshape((3,vsize))
        l = np.zeros((3,vsize))
        for idv in range(vsize):
            I = int(idv/(N*N))
            J = int((idv%(N*N))/N)
            K = int((idv%(N*N))%N)
            l[0,idv] = l0[I];
            l[1,idv] = l0[J];
            l[2,idv] = l0[K];
        d_lx = gpuarray.to_gpu(np.ascontiguousarray(l[0,:]))
        d_ly = gpuarray.to_gpu(np.ascontiguousarray(l[1,:]))
        d_lz = gpuarray.to_gpu(np.ascontiguousarray(l[2,:]))
        
        dtype = self.cfg.dtype
        cdtype = np.complex128
        CUFFT_T2T = CUFFT_Z2Z
        self.cufftExecT2T = cufftExecZ2Z

        if dtype==np.float32: 
            cdtype = np.complex64
            CUFFT_T2T = CUFFT_C2C
            self.cufftExecT2T = cufftExecC2C

        # define scratch  spaces
        self.d_FTf = gpuarray.empty(vsize, dtype=cdtype)
        self.d_fC = gpuarray.empty_like(self.d_FTf)
        self.d_QG = gpuarray.empty_like(self.d_FTf)
        self.d_t1 = gpuarray.empty(M*Nrho*vsize, dtype=cdtype)
        self.d_t2 = gpuarray.empty_like(self.d_t1)
        self.d_t3 = gpuarray.empty_like(self.d_t1)
        self.d_t4 = gpuarray.empty_like(self.d_t1)

        self.block = (128, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)
        self.gridNrhoNv = get_grid_for_block(self.block, Nrho*vsize)
        self.gridNrhoMNv = get_grid_for_block(self.block, Nrho*M*vsize)

        # define complex to complex plan
        rank = 3
        n = np.array([N, N, N], dtype=np.int32)

        #planD2Z = cufftPlan3d(N, N, N, CUFFT_D2Z)
        self.planT2T_MNrho = cufftPlanMany(rank, n.ctypes.data,
            None, 1, vsize, 
            None, 1, vsize, 
            CUFFT_T2T, M*Nrho)
        self.planT2T = cufftPlan3d(N, N, N, CUFFT_T2T)

        dfltargs = dict(dtype=self.cfg.dtypename, 
            Nrho=Nrho, M=M, 
            vsize=vsize, sw=self.vm.sw(), prefac=self._prefactor, 
            qw=qw, sz=sz, gamma=self._gamma, 
            L=L, qz=qz, Ne=self._Ne
        )
        src = DottedTemplateLookup(
            'dgfs1D.std.kernels.scattering', dfltargs
        ).get_template('vhs-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        self.d_aa = gpuarray.empty(Nrho*M*vsize, dtype=dtype)
        precompute_aa = get_kernel(module, "precompute_aa", 'PPPP')
        precompute_aa.prepared_call(self.grid, self.block, d_lx.ptr, d_ly.ptr, 
            d_lz.ptr, self.d_aa.ptr)

        self.d_bb1 = gpuarray.empty(Nrho*vsize, dtype=dtype)
        self.d_bb2 = gpuarray.empty(vsize, dtype=dtype)
        precompute_bb = get_kernel(module, "precompute_bb", 'PPPPP')
        precompute_bb.prepared_call(self.grid, self.block, d_lx.ptr, d_ly.ptr, 
            d_lz.ptr, self.d_bb1.ptr, self.d_bb2.ptr)

        # transform scalar to complex
        self.r2zKern = get_kernel(module, "r2z_", 'iiPP')

        # Prepare the cosSinMul kernel for execution
        self.cosSinMultKern = get_kernel(module, "cosSinMul", 'PPPP')

        # Prepare the magSqrKern kernel for execution
        self.magSqrKern = get_kernel(module, "magSqr", 'PPP')

        # Prepare the computeQG kernel for execution
        self.computeQGKern = get_kernel(module, "computeQG", 'PPP')

        # Prepare the ax kernel for execution
        self.axKern = get_kernel(module, "ax", 'PP')

        # Prepare the output append kernel for execution
        self.outAppendKern = get_kernel(module, "output_append_", 'iiiPPPP')

        # Prepare the scale kernel for execution
        #self.scaleKern = get_kernel(module, "scale", 'P')

        # Prepare the scale kernel for execution
        #self.scaleMNKern = get_kernel(module, "scale_MN", 'P')

        # required by the child class (may be deleted by the child)
        self.module = module

        
    #def fs(self, d_arr_in, d_arr_out, elem, modein, modeout):
    def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr
            
        # construct d_fC from d_f0
        self.r2zKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f0, self.d_fC.ptr)

        # compute forward FFT of f | Ftf = fft(f)
        self.cufftExecT2T(self.planT2T, 
            self.d_fC.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTf.ptr)
        
        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTf_r
        # scales d_FTf
        self.cosSinMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t2 = t3^2 + t4^2
        self.magSqrKern.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t2.ptr)

        # compute t1 = fft(t2)
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t2.ptr, self.d_t1.ptr, CUFFT_FORWARD)
        # scaling factor is multiplied in the computeQGKern 
        # note: t1 is not modified in computeQGKern
        #self.scaleMNKern.prepared_call(self.grid, self.block, 
        #    self.d_t1.ptr)

        # compute fC_r = 2*wrho_p*ws*b1_p*t1_r
        self.computeQGKern.prepared_call(self.grid, self.block, 
            self.d_bb1.ptr, self.d_t1.ptr, self.d_fC.ptr)

        # inverse fft| QG = iff(fC)  [Gain computed]
        self.cufftExecT2T(self.planT2T, 
            self.d_fC.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTf_r = b2_r*FTf_r
        self.axKern.prepared_call(self.grid, self.block, 
            self.d_bb2.ptr, self.d_FTf.ptr)

        # inverse fft| fC = iff(FTf)
        self.cufftExecT2T(self.planT2T, 
            self.d_FTf.ptr, self.d_fC.ptr, CUFFT_INVERSE)
        
        # outKern
        self.outAppendKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, 
            self.d_QG.ptr, self.d_fC.ptr, d_f0, d_Q)
        

# for full nodal version
class DGFSVHSScatteringModelStd(DGFSVHSGLLScatteringModelStd):
    scattering_model = 'vhs'

    def __init__(self, cfg, velocitymesh, **kwargs):
        super().__init__(cfg, velocitymesh, **kwargs)


    def perform_precomputation(self):
        super().perform_precomputation()

        self.d_FTg = gpuarray.empty_like(self.d_FTf)
        self.d_gC = gpuarray.empty_like(self.d_FTf)
        self.d_t5 = gpuarray.empty_like(self.d_t1)

        # Prepare the cosMul kernel for execution
        self.cosMultKern = get_kernel(self.module, "cosMul", 'PPPPP')

        # Prepare the cosMul kernel for execution
        self.sinMultKern = get_kernel(self.module, "sinMul", 'PPPPP')

        # Prepare the cplxMul kernel for execution
        self.cplxMul = get_kernel(self.module, "cplxMul", 'PPP')

        # Prepare the cplxMulAdd kernel for execution
        self.cplxMulAdd = get_kernel(self.module, "cplxMulAdd", 'PPP')

        del self.module


    def fs(self, d_arr_in1, d_arr_in2, d_arr_out, elem, modein, modeout):
        d_f = d_arr_in1.ptr
        d_g = d_arr_in2.ptr
        d_Q = d_arr_out.ptr
            
        # construct d_fC from d_f
        self.r2zKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f, self.d_fC.ptr)

        # construct d_gC from d_g
        self.r2zKern.prepared_call(self.grid, self.block, 
            elem, modein, d_g, self.d_gC.ptr)

        # compute forward FFT of f | FTf = fft(f)
        self.cufftExecT2T(self.planT2T, 
            self.d_fC.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTf.ptr)

        # compute forward FFT of g | FTg = fft(g)
        self.cufftExecT2T(self.planT2T, 
            self.d_gC.ptr, self.d_FTg.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTg.ptr)
        
        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = cos(a_{pqr})*FTg_r
        # scales d_FTf, d_FTg
        self.cosMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_FTg.ptr, 
            self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t5 = t3*t4
        self.cplxMul.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t5.ptr)

        # compute t1_{pqr} = sin(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTg_r
        # "does not" scale d_FTf, d_FTg
        self.sinMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_FTg.ptr, 
            self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t5 += t3*t4
        self.cplxMulAdd.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t5.ptr)

        # compute t1 = fft(t5)
        self.cufftExecT2T(self.planT2T_MNrho, 
            self.d_t5.ptr, self.d_t1.ptr, CUFFT_FORWARD)
        # scaling factor is multiplied in the computeQGKern 
        # note: t1 is not modified in computeQGKern
        #self.scaleMNKern.prepared_call(self.grid, self.block, 
        #    self.d_t1.ptr)

        # compute fC_r = 2*wrho_p*ws*b1_p*t1_r
        self.computeQGKern.prepared_call(self.grid, self.block, 
            self.d_bb1.ptr, self.d_t1.ptr, self.d_fC.ptr)

        # inverse fft| QG = iff(fC)  [Gain computed]
        self.cufftExecT2T(self.planT2T, 
            self.d_fC.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTg_r = b2_r*FTg_r
        self.axKern.prepared_call(self.grid, self.block, 
            self.d_bb2.ptr, self.d_FTg.ptr)

        # inverse fft| fC = iff(FTg)
        self.cufftExecT2T(self.planT2T, 
            self.d_FTg.ptr, self.d_fC.ptr, CUFFT_INVERSE)
        
        # outKern
        self.outAppendKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, self.d_QG.ptr, self.d_fC.ptr, d_f, d_Q)




# BGK "conservative" scattering model
"""
References:
Mieussens, Luc. Journal of Computational Physics 162.2 (2000): 429-466.
"Discrete-velocity models and numerical schemes for the Boltzmann-BGK 
equation in plane and axisymmetric geometries." 

Sruti Chigullapalli, PhD thesis, 2011, Purdue University
"Deterministic Approach for Unsteady Rarefied Flow Simulations in Complex
Geometries and its Application to Gas Flows in Microsystems.""
"""
class DGFSBGKGLLScatteringModelStd(DGFSScatteringModelStd):
    scattering_model = 'bgk-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if basis_kind!='nodal-sem-gll':
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

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = self.cfg.dtype
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = self.vm.d_cvx()
        self.d_cvy = self.vm.d_cvy()
        self.d_cvz = self.vm.d_cvz()
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]
        self.d_equiMoms = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # compute mBGK
        mBGK = np.vstack(
            (np.ones(vsize), cv, np.einsum('ij,ij->j', cv, cv))
        ) # 5 x vsize
        self.d_mBGK = gpuarray.to_gpu((mBGK).ravel()) # vsize x 5 flatenned

        # storage for expM
        self.d_expM = gpuarray.empty_like(self.d_mBGK)

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha
        self.d_alpha = gpuarray.empty(nalph, dtype=dtype)

        # residual 
        self.d_res = gpuarray.empty(1, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            nalph=self.nalph, omega=self._omega,
            block_size=self.block[0], Ne=self._Ne, dtype=self.cfg.dtypename)
        src = DottedTemplateLookup(
            'dgfs1D.std.kernels.scattering', dfltargs
        ).get_template('bgk-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = get_kernel(module, "sum_", 'PPII')
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = get_kernel(module, "flocKern", 'IIPP')

        # compute the first moment of local distribution
        self.mom1Kern = get_kernel(module, "mom1", 'PPP'+'P'+'PPP')

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = get_kernel(module, "mom01Norm", 'PPPP')

        # compute the second moment of local distribution
        self.mom2Kern = get_kernel(module, "mom2", 'PPP'+'PP'+'P'*4)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = get_kernel(module, 
            "equiDistInit", 'P'+'P'*nalph)

        # compute the moments of equilibrium distribution
        self.equiDistMomKern = get_kernel(module, 
            "equiDistMom", 'PPPP'+'P'+'PPPP')

        # Prepare the equiDistCompute kernel for execution
        self.equiDistComputeKern = get_kernel(module, 
            "equiDistCompute", 'PPPP'+'PPP'+'P'*3)

        # Prepare the gaussElim kernel for execution
        self.gaussElimKern = get_kernel(module, 
            "gaussElim", 'PP'+'P'*(nalph*2+1))

        # Prepare the out kernel for execution
        self.outKern = get_kernel(module, "output", 'III'+'P'*2+'PPP')

        # required by the child class (may be deleted by the child)
        self.module = module

        # define a blas handle
        self.blas = CUDACUBLASKernels()

        # multiplication kernel "specifically" for computing jacobian
        sA = (nalph, vsize)
        sB = (nalph, vsize)
        sC = (nalph, nalph)        
        self.jacMulFunc = lambda A, B, C: self.blas.mul(A, sA, B, sB, C, sC)
        self.d_J = gpuarray.empty(nalph*nalph, dtype=dtype)
        

    def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
        # Asumption: d_arr_in1 == d_arr_in2

        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        # alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            self.d_alpha.ptr, *self.ptr(self.d_moms))

        # initialize the residual and the tolerances
        res, initRes, iterTol = 1.0, 1.0, 1e-10
        nIter, maxIters = 0, 20

        # start the iteration        
        while res>iterTol:

            self.equiDistComputeKern.prepared_call(self.grid, self.block,
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
                self.d_alpha.ptr,
                self.d_floc.ptr, self.d_fe.ptr, self.d_expM.ptr,
                self.d_moms[1].ptr, self.d_moms[2].ptr, self.d_moms[3].ptr)

            # form the jacobian 
            self.jacMulFunc(self.d_mBGK, self.d_expM, self.d_J)

            # compute moments (gemv/gemm is slower)
            self.equiDistMomKern.prepared_call(self.grid, self.block, 
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, 
                self.d_cSqr.ptr, self.d_fe.ptr, 
                self.d_mom10.ptr, self.d_mom11.ptr, 
                self.d_mom12.ptr, self.d_mom2.ptr)
            self.sumFunc(self.d_fe, self.d_equiMoms[0])
            self.sumFunc(self.d_mom10, self.d_equiMoms[1])
            self.sumFunc(self.d_mom11, self.d_equiMoms[2])
            self.sumFunc(self.d_mom12, self.d_equiMoms[3])
            self.sumFunc(self.d_mom2, self.d_equiMoms[4])
            
            # gaussian elimination
            self.gaussElimKern.prepared_call((1,1), (1,1,1),
                self.d_res.ptr, self.d_alpha.ptr,
                *self.ptr(list(self.d_moms + self.d_equiMoms + [self.d_J]))
            )

            res = self.d_res.get()[0]
            if nIter==0: initRes = res
            if(isnan(res)): raise RuntimeError("NaN encountered")

            # increment iterations
            nIter += 1

            # break if the number of iterations are greater
            if nIter>maxIters: break

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_fe.ptr, self.d_floc.ptr, d_Q)




# ESBGK "conservative" scattering model
"""
References:
Sruti Chigullapalli, PhD thesis, 2011, Purdue University
"Deterministic Approach for Unsteady Rarefied Flow Simulations in Complex
Geometries and its Application to Gas Flows in Microsystems.""
"""
class DGFSESBGKGLLScatteringModelStd(DGFSScatteringModelStd):
    scattering_model = 'esbgk-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if basis_kind!='nodal-sem-gll':
            raise RuntimeError("Only tested for nodal basis")
        super().__init__(cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        Pr = self.cfg.lookupordefault('scattering-model', 'Pr', 2./3.)
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("Pr:", self._Pr)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = self.cfg.dtype
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # number of variables for ESBGK
        self.nalphES = 10
        nalphES = self.nalphES        

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = self.vm.d_cvx()
        self.d_cvy = self.vm.d_cvy()
        self.d_cvz = self.vm.d_cvz()
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function (BGK)
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # cell local equilibrium distribution function (ESBGK)
        self.d_feES = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # additional variables for ESBGK
        self.d_mom2es_xx = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_xy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zx = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]
        self.d_equiMoms = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # storage for reduced moments for ESBGK
        self.d_momsES = [gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]
        self.d_equiMomsES = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]

        # compute mBGK
        mBGK = np.vstack(
            (np.ones(vsize), cv, np.einsum('ij,ij->j', cv, cv))
        ) # 5 x vsize
        self.d_mBGK = gpuarray.to_gpu((mBGK).ravel()) # vsize x 5 flatenned

        # storage for expM
        self.d_expM = gpuarray.empty_like(self.d_mBGK)

        # compute mBGKES (This gets updated in fs)
        mBGKES = np.vstack(
            (np.ones(vsize), # mass
            cv, # momentum
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:]  # off-diag 
        )) # 10 x vsize
        self.d_mBGKES = gpuarray.to_gpu((mBGKES).ravel()) # vsize x 10 flat

        # storage for expMES
        self.d_expMES = gpuarray.empty_like(self.d_mBGKES)

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha
        self.d_alpha = gpuarray.empty(nalph, dtype=dtype)

        # storage for alpha (ESBGK)
        self.d_alphaES = gpuarray.empty(nalphES, dtype=dtype)

        # storage for F
        self.d_F = gpuarray.empty(nalph, dtype=dtype)

        # storage for F (ESBGK)
        self.d_FES = gpuarray.empty(nalphES, dtype=dtype)

        # residual 
        self.d_res = gpuarray.empty(1, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            nalph=self.nalph, omega=self._omega, Pr=self._Pr,
            block_size=self.block[0], Ne=self._Ne, dtype=self.cfg.dtypename, 
            nalphES=self.nalphES)
        src = DottedTemplateLookup(
            'dgfs1D.std.kernels.scattering', dfltargs
        ).get_template('esbgk-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = get_kernel(module, "sum_", 'PPII')
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = get_kernel(module, "flocKern", 'IIPP')

        # compute the first moment of local distribution
        self.mom1Kern = get_kernel(module, "mom1", 'PPP'+'P'+'PPP')

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = get_kernel(module, "mom01Norm", 'PPPP')

        # compute the second moment of local distribution
        self.mom2Kern = get_kernel(module, "mom2", 'PPP'+'PP'+'P'*4)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = get_kernel(module, 
            "equiDistInit", 'P'+'P'*nalph)

        # computes the equilibrium BGK distribution, and constructs expM
        self.equiDistComputeKern = get_kernel(module, 
            "equiDistCompute", 'PPPP'+'PPP'+'P'*3)

        # compute the moments of equilibrium BGK distribution
        self.equiDistMomKern = get_kernel(module, 
            "equiDistMom", 'PPPP'+'P'+'PPPP')

        # Prepare the gaussElim kernel for execution
        self.gaussElimKern = get_kernel(module, 
            "gaussElim", 'PP'+'P'*(nalph*2+1))

        # compute the second moments for ESBGK
        self.mom2ESKern = get_kernel(module, 
            "mom2ES", 'PPP'+'PP'+'P'*4+'P'*6)

        # compute the second moments for ESBGK
        self.mom2ESNormKern = get_kernel(module, 
            "mom2ESNorm", 'P'+'P'*nalph+'P'*nalphES)

        # computes the equilibrium BGK distribution, and constructs expM
        self.equiESDistComputeKern = get_kernel(module, 
            "equiESDistCompute", 'PPP'+'PPP'+'P'*3)

        # Assemble the ES-BGK system 
        self.assembleESKern = get_kernel(module, "assembleES", 
            'PP'+'P'*(nalphES+1))

        # Update alpha for the ES-BGK system 
        self.updateAlphaESKern = get_kernel(module, "updateAlphaES", 'PP')

        # Prepare the out kernel for execution
        self.outKern = get_kernel(module, "output", 'III'+'P'*2+'PPP')

        # required by the child class (may be deleted by the child)
        self.module = module

        # define a blas handle
        self.blas = CUDACUBLASKernels()

        # multiplication kernel "specifically" for computing BGK jacobian
        sA = (nalph, vsize)
        sB = (nalph, vsize)
        sC = (nalph, nalph)        
        self.jacMulFunc = lambda A, B, C: self.blas.mul(A, sA, B, sB, C, sC)
        self.d_J = gpuarray.empty(nalph*nalph, dtype=dtype)
       
        # multiplication kernel "specifically" for computing ES jacobian
        sAES = (nalphES, vsize)
        sBES = (nalphES, vsize)
        sCES = (nalphES, nalphES)        
        self.jacESMulFunc = lambda A, B, C: self.blas.mul(
            A, sAES, B, sBES, C, sCES)
        self.d_JES = gpuarray.empty(nalphES*nalphES, dtype=dtype)

        # multiplication kernel "specifically" for computing ES dist moments
        sA_eMom = (1, vsize)
        sB_eMom = (nalphES, vsize)
        sC_eMom = (1, nalphES)        
        self.equiESDistMulFunc = lambda A, B, C: self.blas.mul(A, sA_eMom, 
            B, sB_eMom, C, sC_eMom)
        self.d_equiMomsES = gpuarray.empty(nalphES, dtype=dtype)

        #sA_eMom = (nalphES, vsize)
        #self.equiESDistMulFunc = lambda A, B, C: self.blas.gemvO(A, sA_eMom, 
        #    B, vsize, C)
        #self.d_equiMomsES = gpuarray.empty(nalphES, dtype=dtype)

        # define a cusolver handle
        self.cusp = CUDACUSOLVERKernels()
        self.luSolveES = lambda A, B: self.cusp.solveLU(A, 
            (nalphES, nalphES), B, nalphES)

    def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
        # Asumption: d_arr_in1 == d_arr_in2

        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        # alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            self.d_alpha.ptr, *self.ptr(self.d_moms))

        # initialize the residual and the tolerances
        res, initRes, iterTol = 1.0, 1.0, 1e-10
        nIter, maxIters = 0, 20

        # start the iteration        
        while res>iterTol:

            self.equiDistComputeKern.prepared_call(self.grid, self.block,
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
                self.d_alpha.ptr,
                self.d_floc.ptr, self.d_fe.ptr, self.d_expM.ptr,
                self.d_moms[1].ptr, self.d_moms[2].ptr, self.d_moms[3].ptr)

            # form the jacobian 
            self.jacMulFunc(self.d_mBGK, self.d_expM, self.d_J)

            # compute moments (gemv/gemm is slower)
            self.equiDistMomKern.prepared_call(self.grid, self.block, 
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, 
                self.d_cSqr.ptr, self.d_fe.ptr, 
                self.d_mom10.ptr, self.d_mom11.ptr, 
                self.d_mom12.ptr, self.d_mom2.ptr)
            self.sumFunc(self.d_fe, self.d_equiMoms[0])
            self.sumFunc(self.d_mom10, self.d_equiMoms[1])
            self.sumFunc(self.d_mom11, self.d_equiMoms[2])
            self.sumFunc(self.d_mom12, self.d_equiMoms[3])
            self.sumFunc(self.d_mom2, self.d_equiMoms[4])
            
            # gaussian elimination
            self.gaussElimKern.prepared_call((1,1), (1,1,1),
                self.d_res.ptr, self.d_alpha.ptr,
                *self.ptr(list(self.d_moms + self.d_equiMoms + [self.d_J]))
            )

            res = self.d_res.get()[0]
            if nIter==0: initRes = res
            if(isnan(res)): raise RuntimeError("NaN encountered")

            # increment iterations
            nIter += 1

            # break if the number of iterations are greater
            if nIter>maxIters: break

        
        # Now the ESBGK loop

        # compute the second moments for ESBGK
        self.mom2ESKern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, 
            self.d_fe.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr, 
            self.d_mom2es_xx.ptr, self.d_mom2es_yy.ptr, 
            self.d_mom2es_zz.ptr, self.d_mom2es_xy.ptr, 
            self.d_mom2es_yz.ptr, self.d_mom2es_zx.ptr
        )

        # reduce the moments 
        self.sumFunc(self.d_mom2es_xx, self.d_momsES[4])  # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yy, self.d_momsES[5]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zz, self.d_momsES[6]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_xy, self.d_momsES[7]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yz, self.d_momsES[8]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zx, self.d_momsES[9]) # missing: ${cw}/d_moms[0]

        # normalize the moments (incorporates the missing factor)
        # (Also transfers first four entries of d_moms to d_momsES)
        # initializes alphaES
        self.mom2ESNormKern.prepared_call((1,1), (1,1,1), 
            self.d_alphaES.ptr,
            *self.ptr(list(self.d_moms + self.d_momsES))
        )

        # initialize the residual and the tolerances
        res, initRes, iterTol = 1.0, 1.0, 1e-10
        nIter, maxIters = 0, 40

        # start the ESBGK iteration        
        while res>iterTol:

            self.equiESDistComputeKern.prepared_call(self.grid, self.block,
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
                self.d_alphaES.ptr,
                self.d_feES.ptr, self.d_expMES.ptr,
                self.d_momsES[1].ptr, self.d_momsES[2].ptr, self.d_momsES[3].ptr)

            self.jacESMulFunc(self.d_mBGKES, self.d_expMES, self.d_JES)

            # This is faster
            self.equiESDistMulFunc(self.d_feES, self.d_mBGKES, self.d_equiMomsES)

            # generates the vector of conserved variables
            self.assembleESKern.prepared_call((1,1), (1,1,1),
                self.d_res.ptr, self.d_FES.ptr,
                *self.ptr(list(self.d_momsES + [self.d_equiMomsES]))
            )

            # Solve the system
            self.luSolveES(self.d_JES, self.d_FES)

            # update alphaES
            self.updateAlphaESKern.prepared_call((1,1), (self.nalphES,1,1),
                self.d_FES.ptr, self.d_alphaES.ptr
            )

            res = self.d_res.get()[0] #sum(abs(F))
            if nIter==0: initRes = res
            if(isnan(res)): raise RuntimeError("NaN encountered")
            #print(nIter, res)

            # increment iterations
            nIter += 1

            # break if the number of iterations are greater
            if nIter>maxIters: break

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_feES.ptr, self.d_floc.ptr, d_Q)










"""
BGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class DGFSBGKDirectGLLScatteringModelStd(DGFSScatteringModelStd):
    scattering_model = 'bgk-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if basis_kind!='nodal-sem-gll':
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

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = self.cfg.dtype
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = self.vm.d_cvx()
        self.d_cvy = self.vm.d_cvy()
        self.d_cvz = self.vm.d_cvz()
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            nalph=self.nalph, omega=self._omega,
            block_size=self.block[0], Ne=self._Ne, dtype=self.cfg.dtypename)
        src = DottedTemplateLookup(
            'dgfs1D.std.kernels.scattering', dfltargs
        ).get_template('bgk-direct-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = get_kernel(module, "sum_", 'PPII')
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = get_kernel(module, "flocKern", 'IIPP')

        # compute the first moment of local distribution
        self.mom1Kern = get_kernel(module, "mom1", 'PPP'+'P'+'PPP')

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = get_kernel(module, "mom01Norm", 'PPPP')

        # compute the second moment of local distribution
        self.mom2Kern = get_kernel(module, "mom2", 'PPP'+'PP'+'P'*4)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = get_kernel(module, 
            "equiDistInit", 'P'*nalph)

        # Prepare the out kernel for execution
        self.outKern = get_kernel(module, "output", 
            'III'+'P'*nalph+'PPP'+'PPP')

        # required by the child class (may be deleted by the child)
        self.module = module
        

    def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
        # Asumption: d_arr_in1 == d_arr_in2

        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            self.d_moms[0].ptr, self.d_moms[1].ptr, self.d_moms[2].ptr, 
            self.d_moms[3].ptr, self.d_moms[4].ptr)

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, 
            self.d_moms[0].ptr, 
            self.d_moms[1].ptr, self.d_moms[2].ptr, self.d_moms[3].ptr,
            self.d_moms[4].ptr, 
            self.d_fe.ptr, self.d_floc.ptr, d_Q, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr)






"""
ESBGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class DGFSESBGKDirectGLLScatteringModelStd(DGFSScatteringModelStd):
    scattering_model = 'esbgk-direct-gll'

    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        basis_kind = cfg.lookupordefault('basis', 'kind', 'nodal-sem-gll')
        if basis_kind!='nodal-sem-gll':
            raise RuntimeError("Only tested for nodal basis")
        super().__init__(cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        Pr = self.cfg.lookupordefault('scattering-model', 'Pr', 2./3.)
        omega = self.cfg.lookupfloat('scattering-model', 'omega');
        muRef = self.cfg.lookupfloat('scattering-model', 'muRef');
        Tref = self.cfg.lookupfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("Pr:", self._Pr)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = self.cfg.dtype
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # number of variables for ESBGK
        self.nalphES = 10
        nalphES = self.nalphES        

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = self.vm.d_cvx()
        self.d_cvy = self.vm.d_cvy()
        self.d_cvz = self.vm.d_cvz()
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function (BGK)
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # cell local equilibrium distribution function (ESBGK)
        self.d_feES = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # additional variables for ESBGK
        self.d_mom2es_xx = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_xy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zx = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # storage for reduced moments for ESBGK
        self.d_momsES = [gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]
        self.d_equiMomsES = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha (ESBGK)
        self.d_alphaES = gpuarray.empty(nalphES, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            nalph=self.nalph, omega=self._omega, Pr=self._Pr,
            block_size=self.block[0], Ne=self._Ne, dtype=self.cfg.dtypename, 
            nalphES=self.nalphES)
        src = DottedTemplateLookup(
            'dgfs1D.std.kernels.scattering', dfltargs
        ).get_template('esbgk-direct-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = get_kernel(module, "sum_", 'PPII')
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = get_kernel(module, "flocKern", 'IIPP')

        # compute the first moment of local distribution
        self.mom1Kern = get_kernel(module, "mom1", 'PPP'+'P'+'PPP')

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = get_kernel(module, "mom01Norm", 'PPPP')

        # compute the second moment of local distribution
        self.mom2Kern = get_kernel(module, "mom2", 'PPP'+'PP'+'P'*4)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = get_kernel(module, 
            "equiDistInit", 'P'*nalph)

        # Prepare the equiDistCompute kernel for execution
        self.equiDistComputeKern = get_kernel(module, 
            "equiDistCompute", 'PPPP'+'P'*nalph)

        # compute the second moments for ESBGK
        self.mom2ESKern = get_kernel(module, 
            "mom2ES", 'PPP'+'PP'+'P'*4+'P'*6)

        # compute the second moments for ESBGK
        self.mom2ESNormKern = get_kernel(module, 
            "mom2ESNorm", 'P'*nalph+'P'*nalphES)

        # computes the equilibrium BGK distribution, and constructs expM
        self.equiESDistComputeKern = get_kernel(module, 
            "equiESDistCompute", 'PPP'+'P'+'P'*nalphES)

        # Prepare the out kernel for execution
        self.outKern = get_kernel(module, "output", 'III'+'P'*2+'PPP')

        # required by the child class (may be deleted by the child)
        self.module = module

    def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
        # Asumption: d_arr_in1 == d_arr_in2

        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            elem, modein, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        # alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            *self.ptr(self.d_moms))

        # compute the equilibrium BGK distribution
        self.equiDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_fe.ptr, *self.ptr(self.d_moms)
        )
        
        # Now the ESBGK relaxation

        # compute the second moments for ESBGK
        self.mom2ESKern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, 
            self.d_fe.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr, 
            self.d_mom2es_xx.ptr, self.d_mom2es_yy.ptr, 
            self.d_mom2es_zz.ptr, self.d_mom2es_xy.ptr, 
            self.d_mom2es_yz.ptr, self.d_mom2es_zx.ptr
        )

        # reduce the moments 
        self.sumFunc(self.d_mom2es_xx, self.d_momsES[4])  # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yy, self.d_momsES[5]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zz, self.d_momsES[6]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_xy, self.d_momsES[7]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yz, self.d_momsES[8]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zx, self.d_momsES[9]) # missing: ${cw}/d_moms[0]

        # normalize the moments (incorporates the missing factor)
        # (Also transfers first four entries of d_moms to d_momsES)
        # initializes alphaES
        self.mom2ESNormKern.prepared_call((1,1), (1,1,1), 
            *self.ptr(list(self.d_moms + self.d_momsES))
        )

        # compute the ESBGK distribution
        self.equiESDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_feES.ptr, *self.ptr(self.d_momsES)
        )

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            elem, modein, modeout, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_feES.ptr, self.d_floc.ptr, d_Q)


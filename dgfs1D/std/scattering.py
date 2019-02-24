from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma

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

        
    def fs(self, d_arr_in, d_arr_out, elem, modein, modeout):
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


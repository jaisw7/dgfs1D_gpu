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
from dgfs1D.sphericaldesign import get_sphquadrule
import itertools as it
from dgfs1D.util import get_kernel, check

class DGFSScatteringModelBi(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh
        self._nspcs = self.vm.nspcs()

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


# for gll
class DGFSBiVSSScatteringModel(DGFSScatteringModelBi):
    scattering_model = 'vss'
    
    def __init__(self, cfg, velocitymesh, **kwargs):
        self._Ne = kwargs.get('Ne')
        super().__init__(cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        # pre-computation (full sphere)
        ssrulepre = self.cfg.lookup('scattering-model', 'ssrulepre')
        self._Mpre = self.cfg.lookupint('scattering-model', 'Mpre')
        print("Mpre:", self._Mpre)
        srule = get_sphquadrule('symmetric', rule=ssrulepre, npts=self._Mpre)
        self._szpre = srule.pts
        self._swpre = 4*np.pi/self._Mpre

        self._cases=list(it.product(np.arange(self._nspcs), repeat=2))

        self._gamma = {}
        self._eta = {}
        self._prefactor = {}

        masses = self.vm.masses()

        for p, q in self._cases:
            pq = str(p)+str(q)

            alpha = self.cfg.lookupfloat('scattering-model', 'alpha'+pq)
            self._eta[pq] = (alpha-1.)

            omega = self.cfg.lookupfloat('scattering-model', 'omega'+pq)
            self._gamma[pq] = 2.0*(1.-omega)

            dRef = self.cfg.lookupfloat('scattering-model', 'dRef'+pq)
            Tref = self.cfg.lookupfloat('scattering-model', 'Tref'+pq)

            invKn = (np.sqrt(1.+masses[p]/masses[q])*np.pi*self.vm.n0()
                    *dRef*dRef*pow(Tref/self.vm.T0(),omega-0.5)*self.vm.H0())

            mu = masses[p]*masses[q]/(masses[p]+masses[q])

            self._prefactor[pq] = invKn*alpha/(
                np.sqrt(1.+masses[p]/masses[q])*pow(mu, omega-0.5)
                *pow(2., 1+alpha)*gamma(2.5-omega)*np.pi);

            print("Kn%s: %s "%(pq, 1/invKn)),
            print("prefactor%s: %s "%(pq, self._prefactor[pq]))

    
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
        sw = self.vm.sw()
        vsize = self.vm.vsize()
        szpre = self._szpre
        swpre = self._swpre

        check(self.cfg.dtype==np.float64, "Need to extend for single precision")

        # precision control
        dint = np.int32
        dfloat = np.float64
        dcplx = np.complex128

        if self.cfg.dtype == np.float32:
            dfloat = np.float32
            dcplx = np.complex64

        l0 = np.concatenate((np.arange(0,Nv/2, dtype=dint), 
            np.arange(-Nv/2, 0, dtype=dint)))
        l = np.zeros((3,vsize), dtype=dint)
        for idv in range(vsize):
            I = int(idv/(Nv*Nv))
            J = int((idv%(Nv*Nv))/Nv)
            K = int((idv%(Nv*Nv))%Nv)
            l[0,idv] = l0[I];
            l[1,idv] = l0[J];
            l[2,idv] = l0[K];
        d_lx = gpuarray.to_gpu(np.ascontiguousarray(l[0,:]))
        d_ly = gpuarray.to_gpu(np.ascontiguousarray(l[1,:]))
        d_lz = gpuarray.to_gpu(np.ascontiguousarray(l[2,:]))

        # transfer sphere points to gpu
        d_sz_x = gpuarray.to_gpu(np.ascontiguousarray(sz[:,0]))
        d_sz_y = gpuarray.to_gpu(np.ascontiguousarray(sz[:,1]))
        d_sz_z = gpuarray.to_gpu(np.ascontiguousarray(sz[:,2]))
        
        # define complex to complex plan
        rank = 3
        n = np.array([Nv, Nv, Nv], dtype=np.int32)

        #planD2Z = cufftPlan3d(Nv, Nv, Nv, CUFFT_D2Z)
        self.planZ2Z_MNrho = cufftPlanMany(rank, n.ctypes.data,
            None, 1, vsize, 
            None, 1, vsize, 
            CUFFT_Z2Z, M*Nrho)
        self.planZ2Z = cufftPlan3d(Nv, Nv, Nv, CUFFT_Z2Z)

        dfltargs = dict(dtype=self.cfg.dtypename, 
            Nrho=Nrho, M=M, 
            vsize=vsize, sw=sw, prefac=self._prefactor, 
            cases=self._cases, masses=self.vm.masses(),
            qw=qw, qz=qz, 
            L=L, sz=sz, 
            gamma=self._gamma, eta=self._eta,
            Mpre=self._Mpre, szpre=szpre, swpre=swpre #, Ne=self._Ne
        )
        src = DottedTemplateLookup(
            'dgfs1D.bi.kernels.scattering', dfltargs
        ).get_template(self.scattering_model).render()

        # Compile the source code and retrieve the kernel
        print("\nCompiling scattering kernels, this may take some time ...")
        module = compiler.SourceModule(src)

        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        print("Starting precomputation, this may take some time ...")
        start, end = cuda.Event(), cuda.Event()
        cuda.Context.synchronize()
        start.record()
        start.synchronize()

        self.d_aa = gpuarray.empty(Nrho*M*vsize, dtype=dfloat)
        precompute_aa = get_kernel(module, "precompute_a", 'PPPP')
        precompute_aa.prepared_call(self.grid, self.block, 
            d_lx.ptr, d_ly.ptr, d_lz.ptr, self.d_aa.ptr)

        self.d_bb1 = {}; self.d_bb2 = {}
        precompute_bb = {}
        for cp, cq in self._cases:
            cpcq = str(cp)+str(cq)
            self.d_bb1[cpcq] = gpuarray.empty(Nrho*M*vsize, dtype=dcplx)
            self.d_bb2[cpcq] = gpuarray.zeros(vsize, dtype=dcplx)
            precompute_bb[cpcq] = module.get_function("precompute_bc_"+cpcq)
            precompute_bb[cpcq].prepare('IIdddPPPPPPPP')
            precompute_bb[cpcq].set_cache_config(cuda.func_cache.PREFER_L1)

            for p in range(Nrho):
                fac = np.pi/L*qz[p]
                fac_b = swpre*pow(qz[p], self._gamma[cpcq]+2)
                fac_c = qw[p]*sw*fac_b
                for q in range(M):
                    precompute_bb[cpcq].prepared_call(self.grid, self.block,
                        dint(p), dint(q), dfloat(fac), 
                        dfloat(fac_b), dfloat(fac_c),
                        d_lx.ptr, d_ly.ptr, d_lz.ptr, 
                        d_sz_x.ptr, d_sz_y.ptr, d_sz_z.ptr, 
                        self.d_bb1[cpcq].ptr, self.d_bb2[cpcq].ptr
                    )

        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        print("Finished precomputation in: %fs" % (secs))

        # transform scalar to complex
        self.r2zKern = module.get_function("r2z")
        self.r2zKern.prepare('IIIPP')
        self.r2zKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the cosSinMul kernel for execution
        self.cosSinMultKern = {}
        #self.computeQGKern = {}
        self.outKern = {}
        for cp, cq in self._cases:
            idx = str(cp) + str(cq)
            self.cosSinMultKern[idx] = module.get_function("cosSinMul_"+idx)
            self.cosSinMultKern[idx].prepare('PPPPP')
            self.cosSinMultKern[idx].set_cache_config(
                cuda.func_cache.PREFER_L1)

            #self.computeQGKern[idx] = module.get_function("computeQG_"+idx)
            #self.computeQGKern[idx].prepare('PPP')
            #self.computeQGKern[idx].set_cache_config(
            #    cuda.func_cache.PREFER_L1)

            self.outKern[idx] = module.get_function("output_"+idx)
            self.outKern[idx].prepare('IIIIPPPP')
            self.outKern[idx].set_cache_config(
                cuda.func_cache.PREFER_L1)

        # prepare the computeQG kernel
        self.computeQGKern = module.get_function("computeQG")
        self.computeQGKern.prepare('PPP')
        self.computeQGKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the prodKern kernel for execution
        self.prodKern = module.get_function("prod")
        self.prodKern.prepare('PPP')
        self.prodKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the ax kernel for execution
        self.ax2Kern = module.get_function("ax2")
        self.ax2Kern.prepare('PPP')
        self.ax2Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # define scratch  spaces
        self.d_FTf = gpuarray.empty(vsize, dtype=dcplx)
        self.d_FTg = gpuarray.empty(vsize, dtype=dcplx)
        self.d_f1C = gpuarray.empty_like(self.d_FTf)
        self.d_f2C = gpuarray.empty_like(self.d_FTf)
        self.d_QG = gpuarray.empty_like(self.d_FTf)
        self.d_t1 = gpuarray.empty(M*Nrho*vsize, dtype=dcplx)
        self.d_t2 = gpuarray.empty_like(self.d_t1)
        self.d_t3 = gpuarray.empty_like(self.d_t1)

        
    def fs(self, idx, d_arr_in1, d_arr_in2, d_arr_out, elem, upt, uptout):
        d_f1 = d_arr_in1.ptr
        d_f2 = d_arr_in2.ptr
        d_Q = d_arr_out.ptr
            
        # construct d_fC from d_f0
        self.r2zKern.prepared_call(self.grid, self.block, self._Ne,
            elem, upt, d_f1, self.d_f1C.ptr)
        self.r2zKern.prepared_call(self.grid, self.block, self._Ne,
            elem, upt, d_f2, self.d_f2C.ptr)

        # compute forward FFT of f | Ftf = fft(f)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_f1C.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_f2C.ptr, self.d_FTg.ptr, CUFFT_FORWARD)

        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTf_r
        self.cosSinMultKern[idx].prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_FTg.ptr, 
            self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t1.ptr, CUFFT_INVERSE)

        # compute t2 = t3*t1
        self.prodKern.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t1.ptr, self.d_t2.ptr)

        # compute t1 = fft(t2)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t1.ptr, CUFFT_FORWARD)

        # compute f1C_r = wrho_p*ws*b1_p*t1_r
        # scaling for forward FFT is included here
        #self.computeQGKern[idx].prepared_call(self.grid, self.block, 
        #    self.d_bb1[idx].ptr, self.d_t1.ptr, self.d_f1C.ptr)
        self.computeQGKern.prepared_call(self.grid, self.block, 
            self.d_bb1[idx].ptr, self.d_t1.ptr, self.d_f1C.ptr)

        # inverse fft| QG = iff(f1C)  [Gain computed]
        cufftExecZ2Z(self.planZ2Z, 
            self.d_f1C.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTg_r = b2_r*FTg_r
        self.ax2Kern.prepared_call(self.grid, self.block, 
            self.d_bb2[idx].ptr, self.d_FTg.ptr, self.d_FTf.ptr)

        # inverse fft| fC = iff(FTf)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_FTf.ptr, self.d_f2C.ptr, CUFFT_INVERSE)
        
        # outKern
        self.outKern[idx].prepared_call(self.grid, self.block, 
            self._Ne, elem, upt, uptout,
            self.d_QG.ptr, self.d_f2C.ptr, d_f1, d_Q)


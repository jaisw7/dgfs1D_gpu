import numpy as np
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.axnpby import get_axnpby_kerns
from dgfs1D.nputil import get_grid_for_block
from dgfs1D.util import check

""" 
Integration schemes for multi-species systems 
"""

class DGFSIntegratorBi():
    intg_kind = None

    """time integration base class"""
    def __init__(self, rhs, shape, dtype, nspcs, **kwargs):
        self._rhs = rhs
        self._nspcs = nspcs

        K, Ne, Nv = shape
        block = (128, 1, 1)
        grid_NeNv = get_grid_for_block(block, Ne*Nv)

        # axnpby kernel generator
        self._addGen = lambda nx: get_axnpby_kerns(nx, range(K), Ne*Nv, dtype)
        self._add = lambda kern: lambda *x: kern.prepared_call(
                            grid_NeNv, block, 
                            *list(list(map(lambda c: c.ptr, x[1::2])) 
                                + list(x[::2]))
                        )

        self.add2 = self._add(self._addGen(2)) # axnpby on 2 vectors
        self.add3 = self._add(self._addGen(3)) # axnpby on 3 vectors

        # allocate scratch storage 
        self.scratch = [gpuarray.empty(K*Ne*Nv, dtype) 
                        for r in range(self._nscratch)]

    def integrate(self, t, dt, d_ucoeffs):
        pass

    def copy(self, olds, news):
        for old, new in zip(olds, news):
            cuda.memcpy_dtod(new.ptr, old.ptr, old.nbytes)


"""1st order euler time integration scheme"""
class EulerIntegratorBi(DGFSIntegratorBi):
    intg_kind = "euler"

    def __init__(self, rhs, shape, dtype, nspcs, **kwargs):
        super().__init__(rhs, shape, dtype, nspcs, **kwargs)

    @property
    def _nscratch(self):
        return self._nspcs

    def integrate(self, t, dt, d_ucoeffs):
        add2, rhs = self.add2, self._rhs
        r0 = self.scratch

        # First stage; r0[p] = rhs(p, d_ucoeffs); 
        #              r0[p] = d_ucoeffs[p] + dt*r0[p]
        # (unchanged) d_ucoeffs is needed for evaluating collision kernels
        for p in range(self._nspcs):
            rhs(p, t, d_ucoeffs, r0[p])
            add2(dt, r0[p], 1.0, d_ucoeffs[p])

        # finally update d_ucoeffs; d_ucoeffs[p] = r0[p]
        self.copy(r0, d_ucoeffs)



"""2nd order Strong stability preserving RK scheme"""
class SSPRK2IntegratorBi(DGFSIntegratorBi):
    intg_kind = "ssp-rk2"

    def __init__(self, rhs, shape, dtype, nspcs, **kwargs):
        super().__init__(rhs, shape, dtype, nspcs, **kwargs)

    @property
    def _nscratch(self):
        return self._nspcs + 1

    def integrate(self, t, dt, d_ucoeffs):
        nspcs = self._nspcs
        add2, add3, rhs = self.add2, self.add3, self._rhs
        r0, r1 = self.scratch[:nspcs], self.scratch[nspcs:]

        # First stage; r0[p] = rhs(p, d_ucoeffs); 
        #              r0[p] = d_ucoeffs[p] + dt*r0[p]
        for p in range(self._nspcs):
            rhs(p, t, d_ucoeffs, r0[p])
            add2(dt, r0[p], 1.0, d_ucoeffs[p])

        # Second stage; r1[0] = rhs(p, r0); 
        #           d_ucoeffs[p] = 0.5*d_ucoeffs[p] + 0.5*r0[p] + 0.5*dt*r1[0]
        for p in range(self._nspcs):
            rhs(p, t + dt, r0, r1[0])
            add3(0.5, d_ucoeffs[p], 0.5, r0[p], 0.5*dt, r1[0])


"""3rd order Strong stability preserving RK scheme"""
class SSPRK3IntegratorBi(DGFSIntegratorBi):
    intg_kind = "ssp-rk3"

    def __init__(self, rhs, shape, dtype, nspcs, **kwargs):
        super().__init__(rhs, shape, dtype, nspcs, **kwargs)

    @property
    def _nscratch(self):
        return 2*self._nspcs

    def integrate(self, t, dt, d_ucoeffs):
        nspcs = self._nspcs
        add2, add3, rhs = self.add2, self.add3, self._rhs
        r0, r1 = self.scratch[:nspcs], self.scratch[nspcs:]

        # First stage; r0[p] = rhs(p, d_ucoeffs); 
        #              r0[p] = d_ucoeffs[p] + dt*r0[p]
        for p in range(self._nspcs):
            rhs(p, t, d_ucoeffs, r1[p])
            add3(0.0, r0[p], 1.0, d_ucoeffs[p], dt, r1[p])

        # Second stage; r1[p] = rhs(p, r0); 
        #             r1[p] = 0.75*d_ucoeffs[p] + 0.25*r0[p] + 0.25*dt*r1[p]
        for p in range(self._nspcs):
            rhs(p, t+dt, r0, r1[p])
            add3(0.25*dt, r1[p], 0.75, d_ucoeffs[p], dt, r0[p])

        # Third stage; r0[0] = rhs(p, r1);
        # d_ucoeffs[p] = 1.0/3.0*d_ucoeffs[p] + 2.0/3.0*r1[p] + 2.0/3.0*dt*r0[0]
        for p in range(self._nspcs):
            rhs(p, t+0.5*dt, r1, r0[0])
            add3(1.0/3.0, d_ucoeffs[p], 2.0/3.0, r1[p], 2.0/3.0*dt, r0[0])



"""4th order, 5 stages Low staorage RK scheme"""
class LSERK45IntegratorBi(DGFSIntegratorBi):
    intg_kind = "lse-rk45"

    rk45a = [
        0.0,
        -567301805773.0/1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/842570457699.0
    ]

    rk45b = [
        1432997174477.0/9575080441755.0,
        5161836677717.0/13612068292357.0,
        1720146321549.0/2090206949498.0,
        3134564353537.0/4481467310338.0,
        2277821191437.0/14882151754819.0
    ]

    rk45c = [
        0.0,
        1432997174477.0/9575080441755.0,
        2526269341429.0/6820363962896.0,
        2006345519317.0/3224310063776.0,
        2802321613138.0/2924317926251.0
    ]

    def __init__(self, rhs, shape, dtype, nspcs, **kwargs):
        super().__init__(rhs, shape, dtype, nspcs, **kwargs)

    @property
    def _nscratch(self):
        return 2*self._nspcs

    def integrate(self, t, dt, d_ucoeffs):
        nspcs = self._nspcs
        add2, add3, rhs = self.add2, self.add3, self._rhs
        rk45a, rk45b, rk45c = self.rk45a, self.rk45b, self.rk45c
        r0, r1 = self.scratch[:nspcs], self.scratch[nspcs:]


        # s^{th} stage; r0[p] = rhs(p, d_ucoeffs);
        #               r1[p] = rk45a[istp]*r1[p] + dt*r0[0]
        #               r0[p] = d_ucoeffs[p] + rk45b[istp]*r1[p]
        for istp in range(5):
            for p in range(self._nspcs):
                rhs(p, t + rk45c[istp]*dt, d_ucoeffs, r0[p])
                add2(rk45a[istp], r1[p], dt, r0[p])
                add3(0., r0[p], 1., d_ucoeffs[p], rk45b[istp], r1[p])

            # finally update d_ucoeffs; d_ucoeffs[p] = r0[p]
            self.copy(r0, d_ucoeffs)



"""1st order euler time integration scheme for multi-species systems"""
"""class EulerIntegratorBi(EulerIntegratorStd):
    intg_kind = "bi-euler"

    def __init__(self, rhs, shape, dtype, **kwargs):
        self._rhsPtrs = rhs
        super().__init__(rhs, shape, dtype, **kwargs)

    def integrate(self, t, dt, d_ucoeffs):
        for d_ucoeff, rhs in zip(d_ucoeffs, self._rhsPtrs):
            self._rhs = rhs
            super().integrate(t, dt, d_ucoeff)
"""
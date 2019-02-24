import numpy as np
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.axnpby import get_axnpby_kerns
from dgfs1D.nputil import get_grid_for_block

class DGFSIntegratorStd():
    intg_kind = None

    """time integration base class"""
    def __init__(self, rhs, shape, dtype, **kwargs):
        self._rhs = rhs

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

    def integrate(self, t, dt, d_ucoeff):
        pass

"""1st order euler time integration scheme"""
class EulerIntegratorStd(DGFSIntegratorStd):
    intg_kind = "euler"

    def __init__(self, rhs, shape, dtype, **kwargs):
        super().__init__(rhs, shape, dtype, **kwargs)

    @property
    def _nscratch(self):
        return 1

    def integrate(self, t, dt, d_ucoeff):
        add2, rhs = self.add2, self._rhs
        r0,  = self.scratch

        # First stage; r0 = -∇·f(d_ucoeff); d_ucoeff = d_ucoeff + dt*r0
        rhs(t, d_ucoeff, r0)
        add2(1.0, d_ucoeff, dt, r0)


"""2nd order Strong stability preserving RK scheme"""
class SSPRK2IntegratorStd(DGFSIntegratorStd):
    intg_kind = "ssp-rk2"

    def __init__(self, rhs, shape, dtype, **kwargs):
        super().__init__(rhs, shape, dtype, **kwargs)

    @property
    def _nscratch(self):
        return 2

    def integrate(self, t, dt, d_ucoeff):
        add2, add3, rhs = self.add2, self.add3, self._rhs
        r0, r1 = self.scratch

        # First stage; r0 = -∇·f(d_ucoeff); r0 = d_ucoeff + dt*r0
        rhs(t, d_ucoeff, r0)
        add2(dt, r0, 1.0, d_ucoeff)

        # Second stage; r1 = -∇·f(r0); 
        #               d_ucoeff = 0.5*d_ucoeff + 0.5*r0 + 0.5*dt*r1
        rhs(t + dt, r0, r1)
        add3(0.5, d_ucoeff, 0.5, r0, 0.5*dt, r1)


"""3rd order Strong stability preserving RK scheme"""
class SSPRK3IntegratorStd(DGFSIntegratorStd):
    intg_kind = "ssp-rk3"

    def __init__(self, rhs, shape, dtype, **kwargs):
        super().__init__(rhs, shape, dtype, **kwargs)

    @property
    def _nscratch(self):
        return 2

    def integrate(self, t, dt, d_ucoeff):
        add2, add3, rhs = self.add2, self.add3, self._rhs
        r0, r1 = self.scratch

        # First stage; r1 = -∇·f(d_ucoeff); r0 = d_ucoeff + dt*r1
        rhs(t, d_ucoeff, r1)
        add3(0.0, r0, 1.0, d_ucoeff, dt, r1)

        # Second stage; r1 = -∇·f(r0); 
        #               r0 = 0.75*d_ucoeff + 0.25*r0 + 0.25*dt*r1
        rhs(t + dt, r0, r1)
        add3(0.25, r0, 0.75, d_ucoeff, 0.25*dt, r1)

        # Third stage; r1 = -∇·f(r0);
        #              d_ucoeff = 1.0/3.0*d_ucoeff + 2.0/3.0*r0 + 2.0/3.0*dt*r1
        rhs(t + 0.5*dt, r0, r1)
        add3(1.0/3.0, d_ucoeff, 2.0/3.0, r0, 2.0/3.0*dt, r1)


"""4th order, 5 stages Low staorage RK scheme"""
class LSERK45IntegratorStd(DGFSIntegratorStd):
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

    def __init__(self, rhs, shape, dtype, **kwargs):
        super().__init__(rhs, shape, dtype, **kwargs)

    @property
    def _nscratch(self):
        return 2

    def integrate(self, t, dt, d_ucoeff):
        add2, add3, rhs = self.add2, self.add3, self._rhs
        rk45a, rk45b, rk45c = self.rk45a, self.rk45b, self.rk45c
        r0, r1 = self.scratch

        # s^{th} stage; r0 = -∇·f(d_ucoeff);
        #               r1 = rk45a[istp]*r1 + dt*r0
        #               d_ucoeff += rk45b[istp]*r1
        for istp in range(5):
            rhs(t + rk45c[istp]*dt, d_ucoeff, r0)
            add2(rk45a[istp], r1, dt, r0)
            add2(1., d_ucoeff, rk45b[istp], r1)

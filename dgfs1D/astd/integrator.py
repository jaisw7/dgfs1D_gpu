import numpy as np
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
from dgfs1D.axnpby import get_axnpby_kerns, copy
from dgfs1D.nputil import get_grid_for_block

pex = lambda *args: print(*args) + exit(-1)
np.set_printoptions(linewidth=1000)

class DGFSIntegratorAstd():
    intg_kind = None

    def init(self, explicit, sm, shape, dtype, **kwargs):
        self._explicit = explicit
        self._sm = sm
        self._shape = shape
        self._dtype = dtype
        self._explicitQ = kwargs.get('explicitQ')

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

    def allocate(self, explicit, sm, shape, dtype, **kwargs):
        K, Ne, Nv = self._shape
        
        # allocate scratch storage 
        self.scratch = [gpuarray.zeros(K*Ne*Nv, dtype) 
                        for r in range(self._nscratch)]

        # allocate scratch storage for moments
        self.scratch_moms = [gpuarray.zeros(K*Ne*sm.nalph, dtype) 
                        for r in range(self._nscratch_moms)]

    """time integration base class"""
    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)
        self.allocate(*args, **kwargs)

    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        pass


"""(a,b,c) order ARS time integration scheme"""
class ARSabcIntegratorAstd(DGFSIntegratorAstd):
    intg_kind = None

    @staticmethod
    def _integrate(sm, explicit, t, dt, nacptsteps, d_ucoeff, 
        scratch, scratch_moms, _order, _A, A, _nstages, boltz=None, add2=None):    
        #sm, explicit = self._sm, self._explicit
        moment, updateMoment = sm.moment, sm.updateMomentARS
        updateDist, consMaxwellian = sm.updateDistARS, sm.constructMaxwellian

        L, M, F = scratch[::3], scratch[1::3], scratch[2::3]
        U, LU = scratch_moms[::2], scratch_moms[1::2]

        # the first and the last registers are the initial data
        F = [d_ucoeff] + F + [d_ucoeff]

        # Compute the moment of the initial distribution
        moment(t, F[0], U[0])

        # loop over the stages
        for i in range(_nstages):

            _Aj, Aj = _A[i+1][0:i+1], A[i+1][1:i+2]

            # Compute the explicit part; L[i] = -∇·f(d_ucoeff);
            explicit(t, F[i], L[i])
            if boltz is not None:
                boltz(t, F[i], M[i])
                print(">>>", gpuarray.sum(M[i]), end=" ")
                #add2(1., L[i], 1., M[i])
                consMaxwellian(t, U[i], M[i])
                print(gpuarray.sum(M[i]))
                sm.collide(t, U[i], M[i], F[i], M[i])
                print(gpuarray.sum(M[i]))
                add2(1., L[i], -1., M[i])
                
            # Compute the moment of the explicit part
            moment(t, L[i], LU[i])

            # update the moments
            updateMoment(dt, *[*_Aj, *Aj, *LU[:i+1], *U[:i+2]])

            # implictly construct the Maxwellian/Gaussian given moments
            consMaxwellian(t, U[i+1], M[i])

            # update the distribution
            updateDist(dt, *[*_Aj, *Aj, *L[:i+1], *U[:i+2], 
                *M[:i+1], *F[:i+2]])


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        base_integrate = getattr(ARSabcIntegratorAstd, '_integrate')
        base_integrate(self._sm, self._explicit, 
            t, dt, nacptsteps, d_ucoeff, 
            self.scratch, self.scratch_moms,
            self._order, self._A, self.A, self._nstages, 
            boltz=self._explicitQ, add2=self.add2)



"""1st order ARS time integration scheme"""
class ARS111IntegratorAstd(ARSabcIntegratorAstd):
    intg_kind = "ars-111"

    _A = [[0., 0.], [1., 0.]]
    A = [[0., 0.], [0., 1.]]

    _nscratch = 2
    _nscratch_moms = 3
    _order = 1
    _nstages = 1


"""2nd order ARS time integration scheme"""
class ARS222IntegratorAstd(ARSabcIntegratorAstd):
    intg_kind = "ars-222"

    gamma = 1.-(2.**0.5)/2.
    delta = 1. - 1./(2.*gamma)
    _A = [[0., 0., 0.], [gamma, 0., 0.], [delta, 1-delta, 0.]]
    A = [[0., 0., 0.], [0., gamma, 0.], [0., 1-gamma, gamma]]

    _nscratch = 5
    _nscratch_moms = 5
    _order = 2
    _nstages = 2

    @staticmethod
    def _integrate(sm, explicit, t, dt, nacptsteps, d_ucoeff, 
        scratch, scratch_moms):

        attr = lambda a: getattr(ARS222IntegratorAstd, a)
        base_integrate = getattr(ARSabcIntegratorAstd, '_integrate')
        base_integrate(sm, explicit, 
            t, dt, nacptsteps, d_ucoeff, 
            scratch, scratch_moms,
            attr('_order'), attr('_A'), attr('A'), attr('_nstages'))


"""4th order ARS time integration scheme"""
class ARS443IntegratorAstd(ARSabcIntegratorAstd):
    intg_kind = "ars-443"

    _A = [
        [0., 0., 0., 0., 0.], 
        [1./2., 0., 0., 0., 0.], 
        [11./18., 1./18., 0., 0., 0.],
        [5./6., -5./6., 1./2., 0., 0.],
        [1./4., 7./4., 3./4., -7./4., 0.]
    ]
    A = [
        [0., 0., 0., 0., 0.], 
        [0., 1./2., 0., 0., 0.], 
        [0., 1./6., 1./2., 0., 0.],
        [0., -1./2., 1./2., 1./2., 0.],
        [0., 3./2., -3./2., 1./2., 1./2.]
    ]

    _nscratch = 11
    _nscratch_moms = 9
    _order = 4
    _nstages = 4


"""(a,b,c) order SSP L-stable time integration scheme. The difference is the 
last step of weighting i.e., _A[-1,:] neq _w; and A[-1,:] neq w"""
class SSPLabcIntegratorAstd(DGFSIntegratorAstd):

    @staticmethod
    def _integrate(sm, explicit, t, dt, nacptsteps, d_ucoeff, 
        scratch, scratch_moms, _order, _A, A, _nstages, _w, w):    
        moment, updateMoment = sm.moment, sm.updateMomentARS
        updateDist, consMaxwellian = sm.updateDistARS, sm.constructMaxwellian
        updateDistWeight = sm.updateDistWeightSSPL

        F, M, L = scratch[::3], scratch[1::3], scratch[2::3]
        U, LU = scratch_moms[::2], scratch_moms[1::2]

        # the first and the last registers are the initial data
        F = [d_ucoeff] + F + [d_ucoeff]
        dLU, dL = LU[0], L[0]  # dummy indices (their values are never used)

        # Compute the moment of the initial distribution
        moment(t, F[0], U[0])
        updateMoment(dt, *[0., A[0][0], dLU, *U[:2]]) # _A[0][0] = 0
        consMaxwellian(t, U[1], M[0])
        updateDist(dt, *[0., A[0][0], dL, *U[:2], *M[:1], *F[:2]])

        # loop over the stages
        for i in range(1, _nstages):

            _Aj, Aj = _A[i][:i+1], A[i][:i+1]

            # Compute the explicit part; L[i] = -∇·f(d_ucoeff);
            explicit(t, F[i], L[i-1])

            # Compute the moment of the explicit part
            moment(t, L[i-1], LU[i-1])

            # update the moments
            updateMoment(dt, *[*_Aj, *Aj, *LU[:i], dLU, *U[:i+2]])

            # implictly construct the Maxwellian/Gaussian given moments
            consMaxwellian(t, U[i+1], M[i])

            # update the distribution
            updateDist(dt, *[*_Aj, *Aj, *L[:i+1], *U[:i+2], 
                *M[:i+1], *F[:i+2]])

        # update the distribution
        explicit(t, F[_nstages], L[-1])
        updateDistWeight(dt, *[*_w, *w, *L[:_nstages], *U[:_nstages+1], 
            *M[:_nstages], *F[:_nstages+2]])


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        base_integrate = getattr(SSPLabcIntegratorAstd, '_integrate')
        base_integrate(self._sm, self._explicit, 
            t, dt, nacptsteps, d_ucoeff, 
            self.scratch, self.scratch_moms,
            self._order, self._A, self.A, self._nstages, 
            self._w, self.w)


class SSPL222IntegratorAstd(SSPLabcIntegratorAstd):
    intg_kind = "sspl-222"

    gamma = 1.-(2.**0.5)/2.
    _A = [[0., 0.], [1., 0.]]
    A = [[gamma, 0.], [1-2*gamma, gamma]]
    _w = [0.5, 0.5]
    w = [0.5, 0.5]

    _nscratch = 6
    _nscratch_moms = 5
    _order = 2
    _nstages = 2


class SSPL443IntegratorAstd(SSPLabcIntegratorAstd):
    intg_kind = "sspl-443"

    alpha = 0.24169426078821; beta = 0.06042356519705; eta = 0.12915286960590;
    _A = [
        [0., 0., 0., 0.], 
        [0., 0., 0., 0.], 
        [0., 1., 0., 0.],
        [0., 1./4., 1./4., 0.]
    ]
    A = [
        [alpha, 0., 0., 0.], 
        [-alpha, alpha, 0., 0.], 
        [ 0., 1.-alpha, alpha, 0.],
        [beta, eta, 1./2.-beta-eta-alpha, alpha]
    ]
    _w = [0., 1./6., 1./6., 2./3.]
    w = [0., 1./6., 1./6., 2./3.]

    _nscratch = 12
    _nscratch_moms = 9
    _order = 3
    _nstages = 4


"""1st order BDF time integration scheme"""
class Bdf111IntegratorAstd(DGFSIntegratorAstd):
    intg_kind = "bdf-111"

    A = [-1., 1.]
    G = [1.]
    B = 1.

    _nscratch = 2
    _nscratch_moms = 3


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit = self._sm, self._explicit
        moment, updateMoment = sm.moment, sm.updateMomentBDF
        updateDist, consMaxwellian = sm.updateDistBDF, sm.constructMaxwellian

        L0, M = self.scratch
        U0, LU0, U = self.scratch_moms
        a1, a2, g1, b = [*self.A, *self.G, self.B]  

        # Compute the moment of the initial distribution
        moment(t, d_ucoeff, U0)

        # Compute the explicit part; L0 = -∇·f(d_ucoeff);
        explicit(t, d_ucoeff, L0)

        # Compute the moment of the explicit part
        moment(t, L0, LU0) 

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U, b)
        #pex(U.get().reshape(-1,5))

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M)
        #pex(gpuarray.sum(L0))

        if nacptsteps==1: 
            #pex(LU0.get().reshape(-1,5))
            #pex(gpuarray.sum(d_ucoeff))
            pass

        # update the distribution
        updateDist(dt, a1, d_ucoeff, -g1, L0, b, M, a2, U, d_ucoeff)
        #pex(gpuarray.sum(d_ucoeff))
        


"""2nd order BDF time integration scheme"""
class Bdf222IntegratorAstd(DGFSIntegratorAstd):
    intg_kind = "bdf-222"

    A = [1./3., -4./3., 1.]
    G = [-2./3., 4./3.]
    B = 2./3.
    nstepsInit = 100 # when benchmarking for small number of steps, use 1

    _nscratch = 5
    _nscratch_moms = 5


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit = self._sm, self._explicit
        moment, updateMoment = sm.moment, sm.updateMomentBDF
        updateDist, consMaxwellian = sm.updateDistBDF, sm.constructMaxwellian

        L0, f0, L1, f1, M = self.scratch
        U0, LU0, U1, LU1, U = self.scratch_moms
        a1, a2, a3 = self.A
        g1, g2  = self.G
        b = self.B

        # if this is the first step, use the ars-222 scheme
        if nacptsteps==0: 
            _integrate = getattr(ARS222IntegratorAstd, '_integrate')

            tloc, dtloc = t, dt/self.nstepsInit
            fn = d_ucoeff.copy() # copy the data at f(t=t0)

            # compute the data at t+dt
            for step in range(self.nstepsInit):
                _integrate(sm, explicit, tloc, dtloc, 0, 
                    d_ucoeff, [L0, M, f0, L1, f1], [U0, LU0, U1, LU1, U])
                tloc += dtloc

            # Now, we have f(t0+dt) = d_ucoeff
            # We need to compute the explicit information at t=t0 for the next 
            # update. The explicit information at t0+dt using f(t0+dt) is 
            # computed by the integrator during the next time-step. 
            moment(t, fn, U1); explicit(t, fn, L1); moment(t, L1, LU1);
            copy(f1, fn); 
            del fn; return


        # Compute the moment of the initial distribution
        copy(U0, U1);
        moment(t, d_ucoeff, U1)

        # Compute the explicit part; L1 = -∇·f(d_ucoeff);
        copy(L0, L1);
        explicit(t, d_ucoeff, L1)

        # Compute the moment of the explicit part
        copy(LU0, LU1); 
        moment(t, L1, LU1)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U1, -g2, LU1, a3, U, b)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M)

        # update the distribution
        copy(f0, f1); copy(f1, d_ucoeff);
        updateDist(dt, a1, f0, -g1, L0, a2, f1, -g2, L1, 
            b, M, a3, U, d_ucoeff)



"""3rd order BDF time integration scheme"""
class Bdf333IntegratorAstd(DGFSIntegratorAstd):
    intg_kind = "bdf-333"

    A = [-2./11., 9./11., -18./11., 1.]
    G = [6./11., -18./11., 18./11.]
    B = 6./11.
    nstepsInit = 100 # when benchmarking for small number of steps, use 1

    _nscratch = 7
    _nscratch_moms = 7


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit = self._sm, self._explicit
        moment, updateMoment = sm.moment, sm.updateMomentBDF
        updateDist, consMaxwellian = sm.updateDistBDF, sm.constructMaxwellian

        L0, f0, L1, f1, L2, f2, M = self.scratch
        U0, LU0, U1, LU1, U2, LU2, U = self.scratch_moms
        a1, a2, a3, a4 = self.A
        g1, g2, g3  = self.G
        b = self.B

        if nacptsteps==0: 
            _integrate = getattr(ARS222IntegratorAstd, '_integrate')
            #ars443 = ARS443IntegratorAstd(explicit, sm, self._shape, 
            # self._dtype)

            tloc, dtloc = t, dt/self.nstepsInit

            # Compute f(t+dt)
            fn = d_ucoeff.copy() # copy the data at f(t=t0)
            for step in range(self.nstepsInit):
                _integrate(sm, explicit, tloc, dtloc, 0, 
                    d_ucoeff, [L0, M, f0, L1, f1], [U0, LU0, U1, LU1, U])
                #ars443.integrate(tloc, dtloc, step, d_ucoeff)
                tloc += dtloc

            # Now, we have f(t0+dt) = d_ucoeff
            # We need to compute the explicit information at t=t0 for the next 
            # update. 
            moment(t, fn, U1); explicit(t, fn, L1); moment(t, L1, LU1)
            copy(f1, fn)  # Retain the information at t=t0

            # Compute f(t+2dt)
            copy(fn, d_ucoeff)  # copy the data at f(t=t0+dt)
            for step in range(self.nstepsInit):
                _integrate(sm, explicit, tloc, dtloc, 0, 
                    d_ucoeff, [L0, M, f0, L2, f2], [U0, LU0, U2, LU2, U])
                #ars443.integrate(tloc, dtloc, step, d_ucoeff)
                #tloc = tloc + dtloc

            # Now, we have f(t0+2dt) = d_ucoeff
            # We need to compute the explicit information at t=t0+dt for next 
            # update. The explicit information at t0+2dt using f(t0+2dt) is 
            # computed by the integrator during the next time-step. 
            moment(t, fn, U2); explicit(t, fn, L2); moment(t, L2, LU2)
            copy(f2, fn) # Retain the information at t=t0+dt

            del fn; return

        # if this is the second step, use the ars-222 scheme
        # Ofcouse `dt` must not change between the first and the second step
        if nacptsteps==1: return

        # Compute the moment of the initial distribution
        copy(U0, U1); copy(U1, U2);
        moment(t, d_ucoeff, U2)

        # Compute the explicit part; L2 = -∇·f(d_ucoeff);
        copy(L0, L1); copy(L1, L2);
        explicit(t, d_ucoeff, L2)

        # Compute the moment of the explicit part
        copy(LU0, LU1); copy(LU1, LU2); 
        moment(t, L2, LU2)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U1, -g2, LU1, 
            a3, U2, -g3, LU2, a4, U, b)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M)

        # update the distribution
        copy(f0, f1); copy(f1, f2); copy(f2, d_ucoeff);
        updateDist(dt, a1, f0, -g1, L0, a2, f1, -g2, L1, a3, f2, -g3, L2, 
            b, M, a4, U, d_ucoeff)

"""1st order ARS time integration scheme for penalized Boltzmann schemes"""
class ARS111IntegratorABstd(DGFSIntegratorAstd):
    intg_kind = "boltz-ars-111"

    A = [-1., 1.]
    G = [1.]
    B = 1.

    _nscratch = 5
    _nscratch_moms = 3

    def __init__(self, explicit, sm, shape, dtype, **kwargs):

        super().__init__(explicit, sm, shape, dtype, **kwargs)
        self._explicitQ = kwargs.get('explicitQ')
        self._limit1 = kwargs.get('limit1')

    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit, boltz = self._sm, self._explicit, self._explicitQ
        limit1 = self._limit1
        moment, updateMoment = sm.moment, sm.updateMomentBDF
        updateDist, consMaxwellian = sm.updateDistBDF, sm.constructMaxwellian 
        linear = sm.collide

        L0, M0, M, Q0, lQ0 = self.scratch
        U0, LU0, U = self.scratch_moms
        a1, a2, g1, b = [*self.A, *self.G, self.B]  

        # Compute the moment of the initial distribution
        moment(t, d_ucoeff, U0)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U0, M0)

        # compute the boltzmann operator
        boltz(t, d_ucoeff, Q0)

        # evaluate the penalization operator: LQ0 = nu ( M-f )
        linear(t, U0, M0, d_ucoeff, lQ0)
 
        # Compute the explicit part; L0 = -∇·f(d_ucoeff);
        explicit(t, d_ucoeff, L0)

        # Compute the moment of the explicit part
        moment(t, L0, LU0) 

        # assemble explicit part with contribution from linear and quadratic
        self.add3(1., L0, 1., Q0, -1., lQ0)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U, b)

        # limit the moments
        #limit1(U, U)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M)

        # update the distribution
        updateDist(dt, a1, d_ucoeff, -g1, L0, b, M, a2, U, d_ucoeff)



"""1st order ARS time integration scheme for penalized Boltzmann schemes"""
class ARS111IntegratorABEstd(DGFSIntegratorAstd):
    intg_kind = "boltz-ars-111"

    A = [-1., 1.]
    G = [1.]
    B = 1.

    _nscratch = 5
    _nscratch_moms = 4

    def __init__(self, explicit, sm, shape, dtype, **kwargs):

        super().__init__(explicit, sm, shape, dtype, **kwargs)
        self._explicitQ = kwargs.get('explicitQ')
        self._limit1 = kwargs.get('limit1')

    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit, boltz = self._sm, self._explicit, self._explicitQ
        limit1 = self._limit1
        moment, updateMoment = sm.moment, sm.updateMomentBDF
        updateDistNu, linearNu = sm.updateDistNuBDF, sm.collideNu
        consMaxwellian = sm.constructMaxwellian

        L0, M0, M, Q0, lQ0 = self.scratch
        U0, LU0, U, Unu = self.scratch_moms
        a1, a2, g1, b = [*self.A, *self.G, self.B]  

        # Compute the moment of the initial distribution
        moment(t, d_ucoeff, U0)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U0, M0)

        # compute the boltzmann operator
        boltz(t, d_ucoeff, Q0, Unu)
        #print(Unu.get()); exit(0)

        # evaluate the penalization operator: LQ0 = nu ( M-f )
        linearNu(t, Unu, M0, d_ucoeff, lQ0)
 
        # Compute the explicit part; L0 = -∇·f(d_ucoeff);
        explicit(t, d_ucoeff, L0)

        # Compute the moment of the explicit part
        moment(t, L0, LU0) 

        # assemble explicit part with contribution from linear and quadratic
        self.add3(1., L0, 1., Q0, -1., lQ0)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U, b)

        # limit the moments
        #limit1(U, U)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M)

        # update the distribution
        updateDistNu(dt, a1, d_ucoeff, -g1, L0, b, M, a2, Unu, d_ucoeff)


def roll(a, b):
    x = a[0]
    a[:-1] = a[1:]
    a[-1] = x
    return a

"""s order IMEX linear multi-step time integration scheme"""
class IMEXLMs_IntegratorAstd(DGFSIntegratorAstd):
    intg_kind = None
    nstepsInit = 10

    def updater(self, dist, idx, t_):
        sm, explicit, boltz = self._sm, self._explicit, self._explicitQ
        moment, updateMoment, collide = sm.moment, sm.updateMomentLM, sm.collide
        updateDist, consMaxwellian = sm.updateDistLM, sm.constructMaxwellian

        scratch, scratch_moms = self.scratch, self.scratch_moms
        M, F, L = scratch[::3], scratch[1::3], scratch[2::3]
        U, LU, LM = scratch_moms[::3], scratch_moms[1::3], scratch_moms[2::3]

        moment(t_, dist, U[idx]); 
        consMaxwellian(t_, U[idx], M[idx])
        collide(t_, U[idx], M[idx], dist, M[idx])
        moment(t_, M[idx], LM[idx]) 

        explicit(t_, dist, L[idx]); 
        if boltz is not None:
            boltz(t_, dist, F[idx])
            self.add3(1., L[idx], 1., F[idx], -1., M[idx])
        moment(t_, L[idx], LU[idx]);

        copy(F[idx], dist); 


    def pre_integrate(self, **kwargs):
        sm, explicit = self._sm, self._explicit
        ars443 = ARS443IntegratorAstd(explicit, sm, self._shape, self._dtype, 
            **kwargs)
        fn = [0]*self._nstages

        t, dt, d_ucoeff = kwargs.get('t'), kwargs.get('dt'), kwargs.get('f0')        
        tloc, dtloc = t, dt/self.nstepsInit
        print(">>", gpuarray.sum(d_ucoeff))

        for n in range(self._nstages):
            fn[n] = d_ucoeff.copy() # copy the data at f(t=t0)
            for step in range(self.nstepsInit):
                ars443.integrate(tloc, dtloc, step, d_ucoeff)
                tloc += dtloc

        print(">>", gpuarray.sum(d_ucoeff))

        del ars443
        copy(d_ucoeff, fn[0])
        return fn;


    def __init__(self, *args, **kwargs):
        super().init(*args, **kwargs)

        t, dt = kwargs.get('t'), kwargs.get('dt')

        fn = self.pre_integrate(**kwargs)
        super().allocate(*args, **kwargs)
        for n in range(self._nstages): 
            self.updater(fn[n], n, t+n*dt)

        del fn;


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        sm, explicit, boltz = self._sm, self._explicit, self._explicitQ
        moment, updateMoment, collide = sm.moment, sm.updateMomentLM, sm.collide
        updateDist, consMaxwellian = sm.updateDistLM, sm.constructMaxwellian

        scratch, scratch_moms = self.scratch, self.scratch_moms
        M, F, L = scratch[::3], scratch[1::3], scratch[2::3]
        U, LU, LM = scratch_moms[::3], scratch_moms[1::3], scratch_moms[2::3]

        # Prepare the initial solution
        if nacptsteps<self._nstages-1: 
            copy(d_ucoeff, F[nacptsteps+1])
            return

        # update the moments
        updateMoment(dt, self.A, self.B, self.C, LU, LM, U, self._nstages)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U[-1], M[-1])

        if nacptsteps%1==0: print(gpuarray.sum(U[-1]), gpuarray.sum(LM[-1]), gpuarray.sum(M[-1]), end=" ")

        # Computes f^{l+1} 
        #  and P(f^{l+1}) = nu^{l+1} (M(f^{l+1}) - f^{l+1}); (storage in M[-1])
        updateDist(dt, self.A, self.B, self.C, L, M, F, d_ucoeff, U, 
            self._nstages)

        # compute the moment of P(f^{l+1}) for next time step
        collide(t, U[-1], M[-1], d_ucoeff, M[-1])
        moment(t, M[-1], LM[0])

        if nacptsteps%1==0: print(gpuarray.sum(M[-1]), gpuarray.sum(LM[0]))

        # first we do a circular shift to discard the last stage value
        M, F, L = map(lambda v: roll(v, -1), (M, F, L))
        U, LU, LM = map(lambda v: roll(v, -1), (U, LU, LM))
        scratch[::3], scratch[1::3], scratch[2::3] = M, F, L
        scratch_moms[::3], scratch_moms[1::3], scratch_moms[2::3] = U, LU, LM

        # Compute the explicit part; L0 = -∇·f(d_ucoeff) + (Q - P);
        explicit(t, d_ucoeff, L[-1])
        if boltz is not None:
            boltz(t, d_ucoeff, F[-1])
            self.add3(1., L[-1], 1., F[-1], -1., M[0])
        moment(t, L[-1], LU[-1]) # moment of explicit part

        if nacptsteps==1: exit(1)

        # copy the distribution for next time-step
        copy(F[-1], d_ucoeff)


"""1st order BDF time integration scheme"""
class IMEXLM_BDF1_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-bdf1"

    A = [-1., ]
    B = [1., ]
    C = [1., 0.]

    _nstages = 1
    _norder = 1
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""2nd order Crank Nicholson time integration scheme"""
class IMEXLM_CN2_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-cn2"

    A = [-1., 0.]
    B = [1.5, -0.5]
    C = [0.5, 0.5, 0.]

    _nstages = 2
    _norder = 2
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""2nd order Adams time integration scheme"""
class IMEXLM_AD2_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-ad2"

    A = [-1., 0.]
    B = [1.5, -0.5]
    C = [9./16., 3./8., 1./16.]

    _nstages = 2
    _norder = 2
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""2nd order BDF time integration scheme"""
class IMEXLM_BDF2_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-bdf2"

    A = [-4./3., 1./3.]
    B = [4./3., -2./3.]
    C = [2./3., 0., 0.]

    _nstages = 2
    _norder = 2
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""2nd order SG time integration scheme"""
class IMEXLM_SG2_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-sg2"

    A = [-3./4., 0., -1./4.]
    B = [3./2., 0., 0.]
    C = [1., 0., 0., 1./2.]

    _nstages = 3
    _norder = 2
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""3rd order Backward difference time integration scheme"""
class IMEXLM_BDF3_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-bdf3"

    A = [-18./11., 9./11., -2./11.]
    B = [18./11., -18./11., 6./11.]
    C = [6./11., 0., 0., 0.]

    _nstages = 3
    _norder = 3
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""3rd order Adams Bashforth time integration scheme"""
class IMEXLM_AD3_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-ad3"

    A = [-1., 0., 0.]
    B = [23./12., -4./3., 5./12.]
    C = [4661./10000., 15551./30000., 1949./30000., -1483./30000.]

    _nstages = 3
    _norder = 3
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""3rd order Total Variation bounded time integration scheme"""
class IMEXLM_TVB3_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-tvb3"

    A = [-3909./2048., 1367./1024., -873./2048.]
    B = [18463./12288., -1271./768., 8233./12288.]
    C = [1089./2048., -1139./12288., -367./6144., 1699./12288.]

    _nstages = 3
    _norder = 3
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""4th order Backward differentiation time integration scheme"""
class IMEXLM_BDF4_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-bdf4"

    A = [-48./25., 36./25., -16./25., 3./25.]
    B = [48./25., -72./25., 48./25., -12./25.]
    C = [12./25., 0., 0., 0., 0.]

    _nstages = 4
    _norder = 4
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""4th order Total Variation bounded time integration scheme"""
class IMEXLM_TVB4_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-tvb4"

    A = [-21531./8192., 22753./8192., -12245./8192., 2831./8192.]
    B = [13261./8192., -75029./24576., 54799./24576., -15245./24576.]
    C = [4207./8192., -3567./8192., 697./24576., 4315./24576., -41./384.]

    _nstages = 4
    _norder = 4
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""3rd order Shu time integration scheme"""
class IMEXLM_Shu3_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-shu3"

    A = [-16./27, 0., 0., -11./27.]
    B = [16./9., 0., 0., 4./9.]
    C = [9035./19683., 13541./19683., 1127./2187., 7927./19683., 3094./19683.]

    _nstages = 4
    _norder = 3
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


"""4th order Shu time integration scheme"""
class IMEXLM_Shu4_IntegratorAstd(IMEXLMs_IntegratorAstd):
    intg_kind = "imex-lm-shu4"

    A = [-137./400., 0., 0., -959./5000., -8781./94000., -87487/235000.]
    B = [976903./470000., 0., 0., 136757./117500., 266997./470000., 0.]
    C = [237./500., 7547./10000., 299./400., 4513./5875., 118099./235000., 
            174527./470000., 90349./470000.]

    _nstages = 6
    _norder = 4
    _nscratch = 3*_nstages+1
    _nscratch_moms = 3*_nstages+1 # can be optimized


        

"""s order IMEX linear multi-step time integration scheme"""
class IMEXLMs_IntegratorAstd2(DGFSIntegratorAstd):
    intg_kind = None

    @staticmethod
    def _integrate(sm, explicit, t, dt, nacptsteps, d_ucoeff, 
        scratch, scratch_moms, _order, _A, A, _nstages):    
        #sm, explicit = self._sm, self._explicit
        moment, updateMoment = sm.moment, sm.updateMomentARS
        updateDist, consMaxwellian = sm.updateDistARS, sm.constructMaxwellian

        L, M, F = scratch[::3], scratch[1::3], scratch[2::3]
        U, LU = scratch_moms[::2], scratch_moms[1::2]

        # the first and the last registers are the initial data
        F = [d_ucoeff] + F + [d_ucoeff]

        # Compute the moment of the initial distribution
        moment(t, F[0], U[0])

        # loop over the stages
        for i in range(_nstages):

            _Aj, Aj = _A[i+1][0:i+1], A[i+1][1:i+2]

            # Compute the explicit part; L[i] = -∇·f(d_ucoeff);
            explicit(t, F[i], L[i])

            # Compute the moment of the explicit part
            moment(t, L[i], LU[i])

            # update the moments
            updateMoment(dt, *[*_Aj, *Aj, *LU[:i+1], *U[:i+2]])

            # implictly construct the Maxwellian/Gaussian given moments
            consMaxwellian(t, U[i+1], M[i])

            # update the distribution
            updateDist(dt, *[*_Aj, *Aj, *L[:i+1], *U[:i+2], 
                *M[:i+1], *F[:i+2]])


    def integrate(self, t, dt, nacptsteps, d_ucoeff):
        base_integrate = getattr(ARSabcIntegratorAstd, '_integrate')
        base_integrate(self._sm, self._explicit, 
            t, dt, nacptsteps, d_ucoeff, 
            self.scratch, self.scratch_moms,
            self._order, self._A, self.A, self._nstages)

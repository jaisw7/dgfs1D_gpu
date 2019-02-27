from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma
from dgfs1D.nputil import npeval, ndrange

class DGFSInitConditionStd(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, name, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh
        self._init_vals = None

        # read model parameters
        self.load_parameters(name, **kwargs)

        # perform any necessary computation
        self.perform_precomputation()
        print('Finished initial condition precomputation for', name)

    @abstractmethod
    def load_parameters(self, name, **kwargs):
        pass

    @abstractmethod
    def perform_precomputation(self):
        pass

    def get_init_vals(self):
        return self._init_vals

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol):
        initvals = self._init_vals
        for j in range(self.vm.vsize()):
            scal_upts_full[:, :, j] = initvals[j]


class DGFSMaxwellianInitConditionStd(DGFSInitConditionStd):
    model = 'maxwellian'

    def __init__(self, cfg, velocitymesh, name, **kwargs):
        super().__init__(cfg, velocitymesh, name, **kwargs)

    def load_parameters(self, name, wall=False):
        u0 = self.vm.u0()
        self._rhoini = 1.
        if not wall:
            self._rhoini = self.cfg.lookupfloat(name, 'rho')/self.vm.rho0()
        #self._rhoini = self.cfg.lookupordefault(name, 
        #                'rho', self.vm.rho0())/self.vm.rho0()
        self._Tini = self.cfg.lookupfloat(name, 'T')/self.vm.T0()
        uinix = self.cfg.lookupfloat(name, 'ux')/u0
        uiniy = self.cfg.lookupfloat(name, 'uy')/u0
        uiniz = self.cfg.lookupfloat(name, 'uz')/u0
        # because the cv is of shape (3, Nv)
        self._uini = np.array([uinix, uiniy, uiniz]).reshape((3,1))

    def perform_precomputation(self):
        self._init_vals = ((self._rhoini/(np.pi*self._Tini)**1.5)*
            np.exp(-np.sum((self.vm.cv()-self._uini)**2, axis=0)/self._Tini)
        )
        # test the distribution support
        cv = self.vm.cv()
        vsize = self.vm.vsize()
        cw = self.vm.cw()
        T0 = self.vm.T0()
        rho0 = self.vm.rho0()
        molarMass0 = self.vm.molarMass0()
        u0 = self.vm.u0()
        mr = molarMass0/molarMass0
        mcw = mr*cw

        ele_sol = np.zeros(5)
        soln = self._init_vals

        #non-dimensional mass density
        ele_sol[0] = np.sum(soln)*mcw

        #non-dimensional velocities
        ele_sol[1] = np.tensordot(soln, cv[0,:], axes=(0,0))*mcw
        ele_sol[1] /= ele_sol[0]
        ele_sol[2] = np.tensordot(soln, cv[1,:], axes=(0,0))*mcw
        ele_sol[2] /= ele_sol[0]
        ele_sol[3] = np.tensordot(soln, cv[2,:], axes=(0,0))*mcw
        ele_sol[3] /= ele_sol[0]

        # peculiar velocity
        cx = cv[0,:]-ele_sol[1]
        cy = cv[1,:]-ele_sol[2]
        cz = cv[2,:]-ele_sol[3]
        cSqr = cx*cx + cy*cy + cz*cz

        # non-dimensional temperature
        ele_sol[4] = np.sum(soln*cSqr, axis=0)*(2.0/3.0*mcw*mr)
        ele_sol[4] /= ele_sol[0]

        print("bulk-property: input calculated")
        print("mass-density:", self._rhoini*rho0, ele_sol[0]*rho0)
        print("x-velocity:", self._uini[0,0]*u0, ele_sol[1]*u0)
        print("y-velocity:", self._uini[1,0]*u0, ele_sol[2]*u0)
        print("z-velocity:", self._uini[2,0]*u0, ele_sol[3]*u0)
        print("temperature:", self._Tini*T0, ele_sol[4]*T0)

        if( not(
            np.allclose(self._rhoini*rho0, ele_sol[0]*rho0, atol=1e-5)
            and
            np.allclose(self._uini[0,0]*u0, ele_sol[1]*u0, atol=1e-5)
            and
            np.allclose(self._uini[1,0]*u0, ele_sol[2]*u0, atol=1e-5)
            and
            np.allclose(self._uini[2,0]*u0, ele_sol[3]*u0, atol=1e-5)
            and
            np.allclose(self._Tini*T0, ele_sol[4]*T0, atol=1e-5)
        )):
            #raise Warning("Bulk properties not conserved! Check Nv, dev")
            raise ValueError("Bulk properties not conserved! Check Nv, dev")

    def unondim(self):
        return self._uini



class DGFSBKWInitConditionStd(DGFSInitConditionStd):
    model = 'bkw'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def load_parameters(self, name):
        self._t0 = self.cfg.lookupfloat(name, 't0')
        self._init_vals = np.zeros(self.vm.vsize())

    def perform_precomputation(self):
        K = 1.0-np.exp(-self._t0/6.0)
        dK = np.exp(-self._t0/6.0)/6.0
        cv = self.vm.cv()
        for l in range(self.vm.vsize()):
            cSqr = np.dot(cv[:,l], cv[:,l])
            self._init_vals[l] = (1/(2*pow(2*np.pi*K, 1.5))*np.exp(-cSqr/(2.0*K))
                    *((5*K-3)/K+(1-K)/(K*K)*(cSqr)) )

    def unondim(self):
        raise ValueError("Not defined for bkw")


class DGFSMaxwellianExprInitConditionStd(DGFSInitConditionStd):
    model = 'maxwellian-expr'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def load_parameters(self, name):
        self._vars = {}
        self.name = name
        if any(d in self._vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

    def perform_precomputation(self):
        pass

    def maxwellian(self, rhoini, uxini, uyini, uzini, Tini):
        uini = np.array([uxini, uyini, uzini]).reshape((3,1))
        soln = ((rhoini/(np.pi*Tini)**1.5)*
            np.exp(-np.sum((self.vm.cv()-uini)**2, axis=0)/Tini)
        )

        # test the distribution support
        rho_bulk = np.sum(soln)*self.vm.cw()
        if( not(
            np.allclose(rhoini, rho_bulk, atol=1e-5)
        )):
            raise ValueError("Bulk properties not conserved! Check Nv, dev")

        return soln

    def unondim(self):
        raise ValueError("this model depends on spatial coordinates!")

    def get_init_vals(self):
        raise ValueError("this model depends on spatial coordinates!")

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol):
        vars = self._vars

        # Get the physical location of each solution point
        coords = xsol
        vars.update(dict({'x': coords}))

        # Evaluate the ICs from the config file
        rhoini = npeval(self.cfg.lookupexpr(self.name,'rho'),vars)/self.vm.rho0()
        uxini = npeval(self.cfg.lookupexpr(self.name,'ux'),vars)/self.vm.u0()
        uyini = npeval(self.cfg.lookupexpr(self.name,'uy'),vars)/self.vm.u0()
        uzini = npeval(self.cfg.lookupexpr(self.name,'uz'),vars)/self.vm.u0()
        Tini = npeval(self.cfg.lookupexpr(self.name,'T'),vars)/self.vm.T0()

        def isf(data): return isinstance(data, self.cfg.dtype)

        if all(map(isf, [rhoini, uxini, uyini, uzini, Tini])):
            mxwl = self.maxwellian(rhoini,uxini, uyini, uzini,Tini)
            for j in range(self.vm.vsize()):
                scal_upts_full[:, :, j] = mxwl[j]
        else:
            def shape(ds): 
                if isf(ds): return np.full((Nq, Ne), ds)
                else: return ds

            rhoini, uxini, uyini, uzini,Tini = map(shape, 
                [rhoini, uxini, uyini, uzini,Tini])

            for upt, elem in ndrange(Nq, Ne):
                scal_upts_full[upt, elem, :] = self.maxwellian(
                    rhoini[upt, elem],
                    uxini[upt, elem], uyini[upt, elem], uzini[upt, elem],
                    Tini[upt, elem]
                )
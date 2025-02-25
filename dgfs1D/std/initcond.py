from abc import ABCMeta, abstractmethod
import numpy as np
from dgfs1D.nputil import npeval, ndrange
from loguru import logger

class DGFSInitConditionStd(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, name, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh
        self._init_vals = None

        # read model parameters
        self.load_parameters(name, **kwargs)

        # perform any necessary computation
        self.perform_precomputation()
        logger.info('Finished initial condition precomputation for {}', name)

    @abstractmethod
    def load_parameters(self, name, **kwargs):
        pass

    @abstractmethod
    def perform_precomputation(self):
        pass

    def get_init_vals(self):
        return self._init_vals

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
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

        logger.info("bulk-property: input calculated")
        logger.info("mass-density: {}", self._rhoini*rho0, ele_sol[0]*rho0)
        logger.info("x-velocity: {}", self._uini[0,0]*u0, ele_sol[1]*u0)
        logger.info("y-velocity: {}", self._uini[1,0]*u0, ele_sol[2]*u0)
        logger.info("z-velocity: {}", self._uini[2,0]*u0, ele_sol[3]*u0)
        logger.info("temperature: {}", self._Tini*T0, ele_sol[4]*T0)

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
            pass
            #raise Warning("Bulk properties not conserved! Check Nv, dev")
            #raise ValueError("Bulk properties not conserved! Check Nv, dev")

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
        np.exp(-self._t0/6.0)/6.0
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
        np.sum(soln)*self.vm.cw()
        #if( not(
        #    np.allclose(rhoini, rho_bulk, atol=1e-3)
        #)):
        #    raise ValueError("Bulk properties not conserved! Check Nv, dev: %e", rho_bulk)

        return soln

    def ce1(self, rhoini, uxini, uyini, uzini, Tini,
            drhoini, duxini, duyini, duzini, dTini):
        cv = self.vm.cv()
        uini = np.array([uxini, uyini, uzini]).reshape((3,1))
        c2 = np.sum((cv-uini)**2, axis=0)
        soln = (
            (c2/Tini- 2.5)*(cv[0,:]-uxini)*dTini/Tini
            + 2*((cv[0,:]-uxini)*(cv[0,:]-uxini)/Tini - c2/Tini/3.)*duxini
            + 2*((cv[0,:]-uxini)*(cv[1,:]-uyini)/Tini )*duyini
            + 2*((cv[0,:]-uxini)*(cv[2,:]-uzini)/Tini )*duzini
        )
        return soln

    def unondim(self):
        raise ValueError("this model depends on spatial coordinates!")

    def get_init_vals(self):
        raise ValueError("this model depends on spatial coordinates!")

    def read_params(self, vars):
        # Get the physical location of each solution point
        vars.update(self.cfg.section_values('non-dim', np.float64))
        vars.update(self.cfg.section_values('mesh', np.float64))

        # Evaluate the ICs from the config file
        self._rhoini = npeval(self.cfg.lookupexpr(self.name,'rho'),vars)/self.vm.rho0()
        self._uxini = npeval(self.cfg.lookupexpr(self.name,'ux'),vars)/self.vm.u0()
        self._uyini = npeval(self.cfg.lookupexpr(self.name,'uy'),vars)/self.vm.u0()
        self._uzini = npeval(self.cfg.lookupexpr(self.name,'uz'),vars)/self.vm.u0()
        self._Tini = npeval(self.cfg.lookupexpr(self.name,'T'),vars)/self.vm.T0()

    def apply_init_val(self, scal_upts_full, Nq, Ne, xsol, **kwargs):

        rhoini, uxini, uyini, uzini, Tini = (self._rhoini, self._uxini,
            self._uyini, self._uzini, self._Tini)

        def isf(data): return isinstance(data, (self.cfg.dtype, float))
        def make_shape(ds): return np.full((Nq, Ne), ds) if isf(ds) else ds

        if all(map(isf, [rhoini, uxini, uyini, uzini, Tini])):
            mxwl = self.maxwellian(rhoini,uxini, uyini, uzini, Tini)
            for j in range(self.vm.vsize()):
                scal_upts_full[:, :, j] = mxwl[j]
        else:
            rhoini, uxini, uyini, uzini, Tini = map(make_shape,
                [rhoini, uxini, uyini, uzini, Tini])

            for upt, elem in ndrange(Nq, Ne):
                scal_upts_full[upt, elem, :] = self.maxwellian(
                    rhoini[upt, elem],
                    uxini[upt, elem], uyini[upt, elem], uzini[upt, elem],
                    Tini[upt, elem]
                )


    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
        vars = self._vars

        # Get the physical location of each solution point
        coords = xsol
        vars.update(dict({'x': coords}))
        self.read_params(vars)
        self.apply_init_val(scal_upts_full, Nq, Ne, xsol, **kwargs)



class DGFSMaxwellianExprOrder1InitConditionStd(
        DGFSMaxwellianExprInitConditionStd):
    """Initial condition with first order term from Chapman-Enskog theory
    for linear kinetic equations: BGK, ESBGK, Shakov"""
    model = 'maxwellian-expr-ce1'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
        super().apply_init_vals(scal_upts_full, Nq, Ne, xsol, **kwargs)

        rhoini, uxini, uyini, uzini, Tini = (self._rhoini, self._uxini,
            self._uyini, self._uzini, self._Tini)

        def isf(data): return isinstance(data, (self.cfg.dtype, float))
        def make_shape(ds): return np.full((Nq, Ne), ds) if isf(ds) else ds
        rhoini, uxini, uyini, uzini, Tini = map(make_shape,
                [rhoini, uxini, uyini, uzini, Tini])

        invjac=kwargs.get('mesh').invjac; basis=kwargs.get('basis');
        sm=kwargs.get('sm')
        ep=1/sm._prefactor

        Dr = basis.derivMat
        drhoini, duxini, duyini, duzini, dTini = map(
            lambda v: np.matmul(Dr, v), (rhoini, uxini, uyini, uzini, Tini))

        for upt, elem in ndrange(Nq, Ne):
            iJ = invjac[elem,0];
            fac=ep/sm.nu(rhoini[upt, elem], Tini[upt, elem])
            scal_upts_full[upt, elem, :] *= (1 - fac*self.ce1(
                rhoini[upt, elem],
                uxini[upt, elem], uyini[upt, elem], uzini[upt, elem],
                Tini[upt, elem],
                iJ*drhoini[upt, elem],
                iJ*duxini[upt,elem], iJ*duyini[upt,elem], iJ*duzini[upt,elem],
                iJ*dTini[upt, elem],
            ))



class DGFSMaxwellianExprNonDimInitConditionStd(
        DGFSMaxwellianExprInitConditionStd):
    model = 'maxwellian-expr-nondim'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def read_params(self, vars):
        # Get the physical location of each solution point
        vars.update(self.cfg.section_values('non-dim', np.float64))
        vars.update(self.cfg.section_values('mesh', np.float64))

        # Evaluate the ICs from the config file
        self._rhoini = npeval(self.cfg.lookupexpr(self.name,'rho'),vars)
        self._uxini = npeval(self.cfg.lookupexpr(self.name,'ux'),vars)
        self._uyini = npeval(self.cfg.lookupexpr(self.name,'uy'),vars)
        self._uzini = npeval(self.cfg.lookupexpr(self.name,'uz'),vars)
        self._Tini = npeval(self.cfg.lookupexpr(self.name,'T'),vars)


class DGFSMaxwellianExprNonDimOrder1InitConditionStd(
        DGFSMaxwellianExprNonDimInitConditionStd,
        DGFSMaxwellianExprOrder1InitConditionStd):
    """Initial condition with first order term from Chapman-Enskog theory"""
    model = 'maxwellian-expr-nondim-ce1'


"""
class DGFSMaxwellianNonDimInitConditionStd(
        DGFSMaxwellianExprNonDimInitConditionStd):
    model = 'maxwellian-nondim'

    def get_init_vals(self):
        scal_upts_full = np.zeros((1,1,vsize))
        self.apply_init_val(scal_upts_full,1,1,0)
        return scal_upts_full
"""

class DGFSSodShockTubeInitConditionStd(DGFSInitConditionStd):
    model = 'sod-shock-tube'

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

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
        vars = self._vars

        # Properties on the left
        rhol = self.cfg.lookupfloat(self.name,'rho-left')/self.vm.rho0()
        Tl = self.cfg.lookupfloat(self.name,'T-left')/self.vm.T0()
        ulx = self.cfg.lookupordefault(self.name,'ux-left', 0.)/self.vm.u0()
        Ml = self.maxwellian(rhol, ulx, 0., 0., Tl)

        # Properties on the right
        rhor = self.cfg.lookupfloat(self.name,'rho-right')/self.vm.rho0()
        Tr = self.cfg.lookupfloat(self.name,'T-right')/self.vm.T0()
        urx = self.cfg.lookupordefault(self.name,'ux-right', 0.)/self.vm.u0()
        Mr = self.maxwellian(rhor, urx, 0., 0., Tr)

        # Get the physical location of each solution point
        coords = xsol
        vars.update(dict({'x': coords}))

        if Ne%2==0:
            for upt, elem in ndrange(Nq, Ne):
                if elem < Ne//2:
                    scal_upts_full[upt, elem, :] = Ml
                else:
                    scal_upts_full[upt, elem, :] = Mr
        else:
            raise ValueError("Not implemented")


class DGFSSodShockNonDimTubeInitConditionStd(DGFSInitConditionStd):
    model = 'sod-shock-tube-nondim'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def load_parameters(self, name):
        self._vars = {}
        self.name = name
        if any(d in self._vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

    def perform_precomputation(self):
        pass

    def maxwellian(self, rhoini, uxini, Tini):
        z = np.zeros_like(uxini)
        uini = np.array([uxini, z, z]).reshape((3,-1))
        soln = ((rhoini[...,None]/(np.pi*Tini[...,None])**1.5)*
                np.exp(-np.sum((self.vm.cv()[:, None, :]-uini[..., None])**2, axis=0)/Tini[...,None])
        )

        # test the distribution support
        rho_bulk = np.sum(soln*self.vm.cw()[None,...], axis=1)
        if( not(
            np.allclose(rhoini, rho_bulk, atol=1e-5)
        )):
            raise ValueError("Bulk properties not conserved! Check Nv, dev: %e" % (rho_bulk))

        return soln

    def unondim(self):
        raise ValueError("this model depends on spatial coordinates!")

    def get_init_vals(self):
        raise ValueError("this model depends on spatial coordinates!")

    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
        vars = self._vars

        # Get the physical location of each solution point
        vars.update(self.cfg.section_values('non-dim', np.float64))
        vars.update(self.cfg.section_values('mesh', np.float64))

        # Get the physical location of each solution point
        coords = xsol
        vars.update(dict({'x': coords}))

        # Properties on the left
        rhol, Tl, ulx = map(lambda v: npeval(self.cfg.lookupexpr(self.name,v),vars), \
                ('rho-left', 'T-left', 'ux-left'))

        # Properties on the right
        rhor, Tr, urx = map(lambda v: npeval(self.cfg.lookupexpr(self.name, v),vars), \
                ('rho-right', 'T-right', 'ux-right'))

        def isf(data): return isinstance(data, (self.cfg.dtype, float))
        def make_shape(ds): return np.full((Nq, Ne), ds) if isf(ds) else ds
        rhol, Tl, ulx, rhor, Tr, urx = map(make_shape,
                [rhol, Tl, ulx, rhor, Tr, urx])

        xMid = self.cfg.lookupfloat(self.name,'xMid')

        N = self.vm.vsize()
        left = coords.ravel() <= xMid
        right = coords.ravel() > xMid
        Ml = self.maxwellian(rhol.ravel()[left], ulx.ravel()[left], Tl.ravel()[left])
        scal_upts_full.reshape((-1, N))[left, :] = Ml
        Mr = self.maxwellian(rhor.ravel()[right], urx.ravel()[right], Tr.ravel()[right])
        scal_upts_full.reshape((-1, N))[right, :] = Mr

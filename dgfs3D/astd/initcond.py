from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma
from dgfs1D.nputil import (npeval, subclass_where, get_grid_for_block, 
                            DottedTemplateLookup, ndrange)
from pycuda import compiler, gpuarray
from dgfs1D.util import get_kernel, filter_tol

class DGFSInitConditionStd(object, metaclass=ABCMeta):
    def __init__(self, cfg, velocitymesh, name, **kwargs):
        self.cfg = cfg
        self.vm = velocitymesh
        self._init_vals = None

        self.block = (256, 1, 1)
 
        # read model parameters
        self.load_parameters(name, **kwargs)

        # perform any necessary computation
        self.perform_precomputation()
        print('Finished initial condition precomputation for', name)

    @abstractmethod
    def load_parameters(self, name, **kwargs):
        pass

    def perform_precomputation(self):
        pass

    @abstractmethod
    def apply_init_vals(self, scal_upts_full, Nq, Ne, xsol, **kwargs):
        pass


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

    def read_common(self, vars):
        # Get the physical location of each solution point
        vars.update(self.cfg.section_values('mesh', np.float64))
        vars.update(self.cfg.section_values('basis', np.float64))
        vars.update(self.cfg.section_values('non-dim', np.float64))
        vars.update(self.cfg.section_values('velocity-mesh', np.float64))

    def read_params(self, vars):
        self.read_common(vars)

        # Evaluate the ICs from the config file
        self._rhoini = '((%s)/(%f))' % (self.cfg.lookupexpr(self.name,'rho'), self.vm.rho0())
        self._uxini = '((%s)/(%f))' % (self.cfg.lookupexpr(self.name,'ux'), self.vm.u0())
        self._uyini = '((%s)/(%f))' % (self.cfg.lookupexpr(self.name,'uy'), self.vm.u0())
        self._uzini = '((%s)/(%f))' % (self.cfg.lookupexpr(self.name,'uz'), self.vm.u0())
        self._Tini = '((%s)/(%f))' % (self.cfg.lookupexpr(self.name,'T'), self.vm.T0())

    def apply_init_val(self, d_f, Nq, Ne, xsol, **kwargs):
       #import re
       #self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini = map(
       #        lambda v: npeval(v, self._vars), 
       #    (self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini)
       #)
       #def repl(st):
       #    for a, b in self._vars.items():
       #        st = st.replace(a, )
       #self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini = map(
       #        lambda v: reduce(lambda v1: v.replace(v1[0],v1[1]), self._vars.items()), # .sub(r'(\s*)', lambda x:self._vars.get(x.group(1)),v),
       #    (self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini)
       #)
       from mako.template import Template
       self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini = map(
               lambda v: Template(v).render(**self._vars), 
           (self._rhoini, self._uxini, self._uyini, self._uzini, self._Tini)
       )

       dfltargs = {}
       dfltargs.update({'dtype':self.cfg.dtypename, 'NqT': Nq, 'rho': self._rhoini, 
           'ux': self._uxini, 'uy': self._uyini, 'NeT': Ne, 'vsize': self.vm.vsize(), 
           'uz': self._uzini, 'T': self._Tini, 'dim': self.cfg.dim})
       dfltargs.update(self._vars)
       kernsrc = DottedTemplateLookup('dgfs3D.astd.kernels.initconds', 
                                    dfltargs).get_template(self.model).render()
       kernmod = compiler.SourceModule(kernsrc)

       NqNeNv = Nq*Ne*self.vm.vsize()
       grid_NqNeNv = get_grid_for_block(self.block, NqNeNv)
       kern_Op = lambda *args: get_kernel(kernmod, "applyIC", 'iPPPPP').prepared_call(
              grid_NqNeNv, self.block, NqNeNv, *list(map(lambda c: c.ptr, args)) )
       kern_Op(d_f, self.vm.d_cvx(), self.vm.d_cvy(), self.vm.d_cvz(), xsol)
       
       #x = xsol.get().reshape(Nq,Ne,self.cfg.dim)
       #x, y = x[:,:,0], x[:,:,1]
       #import matplotlib.pyplot as plt
       #Nq, Ne = map(int, (Nq**0.5, Ne**0.5))
       #x, y = map(lambda v: v.reshape(Nq,Nq,Ne,Ne).swapaxes(1,2).reshape(Ne*Nq,Ne*Nq), (x,y,))
       #plt.contourf(x,y,0.0023*(1+np.sin(x)))
       #plt.savefig('plot.pdf')
        


    def apply_init_vals(self, d_f, Nq, Ne, xsol, **kwargs):
        vars = self._vars

        # Get the physical location of each solution point
        #coords = np.meshgrid(*xsol)
        #coords = xsol
        #names = ['x'+str(i) for i in range(self.cfg.dim)]
        #vars.update(dict(zip(names, coords)))
        self.read_params(vars)
        self.apply_init_val(d_f, Nq, Ne, xsol, **kwargs)



class DGFSMaxwellianExprNonDimInitConditionStd(
        DGFSMaxwellianExprInitConditionStd):
    model = 'maxwellian-expr-nondim'

    def __init__(self, cfg, velocitymesh, name):
        super().__init__(cfg, velocitymesh, name)

    def read_params(self, vars):
        super().read_common(vars)

        # Evaluate the ICs from the config file
        #self._rhoini = '({0})'.format(self.cfg.lookupexpr(self.name,'rho'))
        #self._uxini = '({0})'.format(self.cfg.lookupexpr(self.name,'ux'))
        #self._uyini = '({0})'.format(self.cfg.lookupexpr(self.name,'uy'))
        #self._uzini = '({0})'.format(self.cfg.lookupexpr(self.name,'uz'))
        #self._Tini = '({0})'.format(self.cfg.lookupexpr(self.name,'T'))

        self._rhoini = '((%s))' % (self.cfg.lookupexpr(self.name,'rho'), )
        self._uxini = '((%s))' % (self.cfg.lookupexpr(self.name,'ux'), )
        self._uyini = '((%s))' % (self.cfg.lookupexpr(self.name,'uy'), )
        self._uzini = '((%s))' % (self.cfg.lookupexpr(self.name,'uz'), )
        self._Tini = '((%s))' % (self.cfg.lookupexpr(self.name,'T'), )



# The following modules are based on the "PyFR" implementation 
# (See licences/LICENSE_PyFR)

import itertools as it
import re
import numpy as np

def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]))

_npeval_syms = {
    '__builtins__': None,
    'exp': np.exp, 'log': np.log,
    'sin': np.sin, 'asin': np.arcsin,
    'cos': np.cos, 'acos': np.arccos,
    'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
    'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
    'tanh': np.tanh, 'pi': np.pi, 
    'linspace': np.linspace, 'logspace': np.logspace}

def npeval(expr, locals):
    # Disallow direct exponentiation
    if '^' in expr or '**' in expr:
        raise ValueError('Direct exponentiation is not supported; use pow')

    # Ensure the expression does not contain invalid characters
    if not re.match(r'[A-Za-z0-9 \t\n\r.,+\-*/%()<>=]+$', expr):
        raise ValueError('Invalid characters in expression')

    # Disallow access to object attributes
    objs = '|'.join(it.chain(_npeval_syms, locals))
    if re.search(r'(%s|\))\s*\.' % objs, expr):
        raise ValueError('Invalid expression')

    return eval(expr, _npeval_syms, locals)

def ndrange(*args):
    return it.product(*map(range, args))

def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc

def subclass_where(cls, **kwargs):
    k, v = next(iter(kwargs.items()))

    for s in subclasses(cls):
        if hasattr(s, k) and getattr(s, k) == v:
            return s

    raise KeyError("No subclasses of {0} with cls.{1} == '{2}'"
                   .format(cls.__name__, k, v))


import pkgutil
from mako.lookup import TemplateLookup
from mako.template import Template


class DottedTemplateLookup(TemplateLookup):
    def __init__(self, pkg, dfltargs):
        self.dfltpkg = pkg
        self.dfltargs = dfltargs

    def adjust_uri(self, uri, relto):
        return uri

    def get_template(self, name):
        div = name.rfind('.')

        # Break apart name into a package and base file name
        if div >= 0:
            pkg = name[:div]
            basename = name[div + 1:]
        else:
            pkg = self.dfltpkg
            basename = name

        # Attempt to load the template
        src = pkgutil.get_data(pkg, basename + '.mako')
        if not src:
            raise RuntimeError('Template "{}" not found'.format(name))

        # Subclass Template to support implicit arguments
        class DefaultTemplate(Template):
            def render(iself, *args, **kwargs):
                return super().render(*args, **dict(self.dfltargs, **kwargs))

        return DefaultTemplate(src, lookup=self)


import os

# MPI based routines
def get_comm_rank_root():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm, comm.rank, 0


def get_local_rank():
    envs = ['OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK']

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        hostn = MPI.Get_processor_name()
        grank = MPI.COMM_WORLD.rank
        lrank = 0

        for i, n in enumerate(MPI.COMM_WORLD.allgather(hostn)):
            if i >= grank:
                break

            if hostn == n:
                lrank += 1

        return lrank


def get_mpi(attr):
    from mpi4py import MPI

    return getattr(MPI, attr.upper())

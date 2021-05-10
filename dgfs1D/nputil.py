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

def sndrange(*args):
    return (tuple(i) for i in ndrange(*args) if tuple(reversed(i)) >= tuple(i))

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


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    if len(srtdidx) > 1:
        i, ix = 0, srtdidx[0]
        for j, jx in enumerate(srtdidx[1:], start=1):
            if arrd[jx] - arrd[ix] >= tol:
                if j - i > 1:
                    srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
                i, ix = j, jx

        if i != j:
            srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidxs

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



def computeError(df, df2, mesh, vm, basis):
    from dgfs1D.quadratures import nodal_basis_at, zwglj
    f, f2 = map(lambda v: v.get().reshape(basis.Nq, -1, vm.vsize()), (df,df2))
    Nq, Ne2, nvars = f2.shape
    r, w = zwglj(Nq, 0., 0.);
    r2 = r*0.5 + 0.5;
    r2 = np.array([*(-np.flip(r2)), *(r2)]);
    im = nodal_basis_at(Nq, r, r2)

    fr = np.einsum('rm,mjk->rjk', im, f)
    fr = fr.swapaxes(0,2).reshape((-1, Ne2, Nq)).swapaxes(0,2);
    f2r = f2;
    order = 2 # integer norm order
    diff = np.abs(fr - f2r)**order; 
    L1e = np.einsum('q,qjk,j->jk', w, diff, mesh.jac.ravel())

    res = np.sum(L1e, axis=(0, 1))
    res = np.array([res, 0.])
    comm, rank, root = get_comm_rank_root()
    if rank != root:
        comm.Reduce(res, None, op=get_mpi('sum'), root=root)
        return None
    else:
        comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
        return (res[0]**(1./order))/((mesh.xhi-mesh.xlo)*2*vm.L())


def computeError2(df, df2, mesh, vm, basis):
    from dgfs1D.quadratures import nodal_basis_at, zwglj
    f, f2 = map(lambda v: v.get().reshape(basis.Nq, -1, vm.vsize()), (df,df2))
    Nq, Ne2, nvars = f2.shape
    r, w = zwglj(Nq, 0., 0.);
    r2, w2 = zwglj(Nq, 0., 0.);
    #r2 = r*0.5 + 0.5;
    #r2 = np.array([*(-np.flip(r2)), *(r2)]);
    im = nodal_basis_at(Nq, r, r2)

    fr = np.einsum('rm,mjk->rjk', im, f)
    f2r = np.einsum('rm,mjk->rjk', im, f2)

    xmesh, jac = mesh.xmesh, mesh.jac
    xsol = np.array([0.5*(xmesh[j]+xmesh[j+1])+jac[j]*r2 for j in range(Ne2)]).T
    time = 0.2
    u = np.sin(2*np.pi*(xsol-time))
    #fr = np.stack([u]*nvars,2)
    #fr.set(np.stack([u]*Nv,2).ravel())
    #fr = fr.reshape

    #fr = fr.swapaxes(0,2).reshape((-1, Ne2, Nq)).swapaxes(0,2);
    #fr = f
    #f2r = f2;
    order = 2 # integer norm order
    #diff = np.abs(fr - f2r)**order; 
    diff = np.abs(fr - f2r)**order; 
    L1e = np.einsum('q,qjk,j->jk', w2, diff, mesh.jac.ravel())

    res = np.sum(L1e, axis=(0, 1))
    #res = np.max(diff)
    #res = np.max(np.abs(df - df2))

    res = np.array([res, 0.])
    comm, rank, root = get_comm_rank_root()
    if rank != root:
        comm.Reduce(res, None, op=get_mpi('sum'), root=root)
        return None
    else:
        comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
        return (res[0]**(1./order))/((mesh.xhi-mesh.xlo)*2*vm.L())
        #return (res[0]**(1./order))



def computeError2_3D(df, df2, mesh, vm, basis):
    f, f2 = map(lambda v: v.get().reshape(basis.Nq, -1, vm.vsize()), (df,df2))
    Nq, Ne2, nvars = f2.shape
    
    w = np.array(basis.w[0])
    fr = f 
    f2r = f2 

    jac = mesh.jac
    from dgfs1D.util import ndgrid
    jac = np.array(ndgrid(*jac)[0])

    order = 2 # integer norm order
    diff = np.abs(fr - f2r)**order; 
    L1e = np.einsum('q,qjk,j->jk', w.ravel(), diff, jac.ravel())

    res = np.sum(L1e, axis=(0, 1))
    res = np.array([res, 0.])
    comm, rank, root = get_comm_rank_root()
    if rank != root:
        comm.Reduce(res, None, op=get_mpi('sum'), root=root)
        return None
    else:
        comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
        
        dim = mesh.cfg.dim
        fac = 0
        for i in range(dim): 
            fac += (mesh.xhi[i]-mesh.xlo[i])
        fac *= (2*vm.L())**3

        return (res[0]**(1./order))/fac



def computeError_3D(df, df2, mesh, vm, basis, dx, dx2):
    from dgfs1D.quadratures import nodal_basis_at, zwglj
    f, f2 = map(lambda v: v.get().reshape(basis.Nq, -1, vm.vsize()), (df,df2))
    Nq, Ne2, nvars = f2.shape
    _, Ne, _ = f.shape
    x, x2 = map(lambda v: v[:,:,0].reshape(basis.Nq, -1), (dx, dx2))
    y, y2 = map(lambda v: v[:,:,1].reshape(basis.Nq, -1), (dx, dx2))
    
    w = np.array(basis.w[0])

    # interpolation operator
    dim = mesh.cfg.dim
    Nq1 = int(Nq**(1./dim))
    Ne1 = int(Ne2**(1./dim))

    print(f.shape, f2.shape, Ne1)

    r, _ = zwglj(Nq1, 0., 0.);
    r2 = r*0.5 + 0.5;
    r2 = np.array([*(-np.flip(r2)), *(r2)]);
    Br = nodal_basis_at(basis._K1, r, r2).T  # at "recons" points
    from dgfs1D.util import ndkron
    im = ndkron(*[Br]*dim).T
    print(im.shape)

    fr = np.einsum('rm,mjk->rjk', im, f)
    # fr = (q1*q2, e1*e2, v)
    # 
    #fr = fr.swapaxes(0,2).reshape((-1, Ne2, Nq)).swapaxes(0,2);
    f2r = f2;

    xr, yr = map(lambda v: np.einsum('rm,mj->rj', im, v), (x,y))
    #func = lambda v: v.reshape(Ne2, Nq).T
    #func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq, Ne2)
    func = lambda v: v.T.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq, Ne2)
    func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq1*Ne1, Nq1*Ne1).swapaxes(0,1).ravel().reshape(Nq,Ne2)
    
    func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq,Ne2) 
    xr, yr = map(func, (xr, yr))

    np.set_printoptions(suppress=True)
    print(x2.T)
    print(xr.T)
 
    print(x2.shape, xr.shape)
    import matplotlib.pyplot as plt
    plt.scatter(xr, yr, marker='o', s=20, facecolors='none', edgecolors='r')
    plt.scatter(x2, y2, c='k', marker='.', s=.1)
    plt.savefig('plot.pdf')
    print(np.linalg.norm(x2-xr), np.linalg.norm(y2-yr))
    exit(-1)

    jac = mesh.jac
    from dgfs1D.util import ndgrid
    jac = np.array(ndgrid(*jac)[0])

    order = 2 # integer norm order
    diff = np.abs(fr - f2r)**order; 
    L1e = np.einsum('q,qjk,j->jk', w.ravel(), diff, jac.ravel())

    res = np.sum(L1e, axis=(0, 1))
    res = np.array([res, 0.])
    comm, rank, root = get_comm_rank_root()
    if rank != root:
        comm.Reduce(res, None, op=get_mpi('sum'), root=root)
        return None
    else:
        comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
        
        dim = mesh.cfg.dim
        fac = 0
        for i in range(dim): 
            fac += (mesh.xhi[i]-mesh.xlo[i])
        fac *= (2*vm.L())**3

        return (res[0]**(1./order))/fac

"""

def computeError1_3D(df, df2, mesh, vm, basis, dx, dx2):
    from dgfs1D.quadratures import nodal_basis_at, zwglj
    f, f2 = map(lambda v: v.get().reshape(basis.Nq, -1, vm.vsize()), (df,df2))
    Nq, Ne2, nvars = f2.shape
    Nq_, Ne, _ = f.shape
    x, x2 = map(lambda v: v[:,:,0].reshape(basis.Nq, -1), (dx, dx2))
    y, y2 = map(lambda v: v[:,:,1].reshape(basis.Nq, -1), (dx, dx2))
    
    w = np.array(basis.w[0])

    # interpolation operator
    dim = mesh.cfg.dim
    Nq1 = int(Nq**(1./dim))
    Ne1 = int(Ne2**(1./dim))
    Nq_ = int(Nq_**(1./dim))*2
    Ne_ = int(Ne**(1./dim))

    print(f.shape, f2.shape, Ne1)

    r, _ = zwglj(Nq1, 0., 0.);
    r2 = r*0.5 + 0.5;
    r2 = np.array([*(-np.flip(r2)), *(r2)]);
    Br = nodal_basis_at(basis._K1, r, r2).T  # at "recons" points
    from dgfs1D.util import ndkron
    im = ndkron(*[Br]*dim).T
    print(im.shape)

    fr = np.einsum('rm,mjk->rjk', im, f)
    # fr = (q1*q2, e1*e2, v)
    # 
    #fr = fr.swapaxes(0,2).reshape((-1, Ne2, Nq)).swapaxes(0,2);
    f2r = f2;

    xr, yr = map(lambda v: np.einsum('rm,mj->rj', im, v), (x,y))
    #func = lambda v: v.reshape(Ne2, Nq).T
    #func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq, Ne2)
    func = lambda v: v.T.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq, Ne2)
    func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).swapaxes(1,2).reshape(Nq1*Ne1, Nq1*Ne1).swapaxes(0,1).ravel().reshape(Nq,Ne2)
    
    #func = lambda v: v.T.reshape(Ne1,Ne1,Nq1,Nq1).swapaxes(1,2).reshape(Ne2,Nq).swapaxes(0,1).reshape(Ne1,Ne1,Nq1,Nq1)
    func = lambda v: v.T.reshape(Ne1,Ne1,Nq1,Nq1).swapaxes(1,2).reshape(Ne1*Nq1,Ne1*Nq1).T
   
    func = lambda v: v.reshape(Nq1,Nq1,Ne1,Ne1).reshape(Nq,Ne2)
    func = lambda v: v.reshape(Nq_,Nq_,Ne_,Ne_).swapaxes(1,2).swapaxes(2,3).swapaxes(0,1).reshape(Nq,Ne2)
    func = lambda v: v.reshape(2,Nq1,2,Nq1,Ne_,Ne_).swapaxes(1,3).swapaxes(2,3).swapaxes(3,5).swapaxes(4,5).reshape(Nq,Ne2)
    func = lambda v: v.reshape(2,Nq1,2,Nq1,Ne_,Ne_).swapaxes(2,4).swapaxes(4,5).swapaxes(0,3).swapaxes(2,3).swapaxes(0,1).reshape(Nq,Ne2)
    func = lambda v: v.reshape(2,Nq1,2,Nq1,Ne_,Ne_).swapaxes(2,4).swapaxes(4,5).swapaxes(0,3).swapaxes(0,1).reshape(Nq,Ne2)
    xr, yr = map(func, (xr, yr))
"""

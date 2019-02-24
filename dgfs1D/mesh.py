import numpy as np
from dgfs1D.nputil import npeval, get_mpi, get_comm_rank_root
from dgfs1D.quadratures import zwglj

msect = 'mesh'

class Mesh(object):

    def classic(self):
        xlo = self.cfg.lookupfloat(msect, 'xlo')
        xhi = self.cfg.lookupfloat(msect, 'xhi')
        assert xlo<xhi, "xlo must be less than xhi"

        Ne = self.cfg.lookupint(msect, 'Ne')
        assert Ne>0, "Need atleast one element to define a mesh"

        return np.linspace(xlo, xhi, Ne+1, dtype=self.cfg.dtype)

    def expr(self):
        expr = self.cfg.lookupexpr(msect, 'expr')
        xmesh = np.sort(
            np.array(npeval(expr, {}), dtype=self.cfg.dtype).ravel())
        assert len(xmesh)>=2, "Need atleast two points to define the mesh"
        return xmesh

    def glj(self):
        xlo = self.cfg.lookupfloat(msect, 'xlo')
        xhi = self.cfg.lookupfloat(msect, 'xhi')
        assert xlo<xhi, "xlo must be less than xhi"

        alpha = self.cfg.lookupordefault(msect, 'alpha', 0.0)
        beta = self.cfg.lookupordefault(msect, 'beta', 0.0)
        assert alpha > -1., 'Requirement of gauss quadratures'
        assert beta > -1., 'Requirement of gauss quadratures'

        Ne = self.cfg.lookupint(msect, 'Ne')
        assert Ne>0, "Need atleast one element to define a mesh"

        # quadrature in interval [-1, 1]
        z, _ = zwglj(Ne+1, alpha, beta)
        z = z.astype(self.cfg.dtype)

        # scale to interval [xlo, xhi]
        z = 0.5*(xhi+xlo) + 0.5*(xhi-xlo)*z

        return z

    def bump(self):
        xlo = self.cfg.lookupfloat(msect, 'xlo')
        xhi = self.cfg.lookupfloat(msect, 'xhi')
        assert xlo<xhi, "xlo must be less than xhi"

        mu = self.cfg.lookupordefault(msect, 'mu', 0.0)
        assert mu >= 0., 'Requirement of bump'

        Ne = self.cfg.lookupint(msect, 'Ne')
        assert Ne>0, "Need atleast one element to define a mesh"

        # points in interval [-1, 1]
        z = np.tanh(2*mu*np.arange(Ne+1)/Ne-mu)/np.tanh(mu)
        z = z.astype(self.cfg.dtype)

        # scale to interval [xlo, xhi]
        z = 0.5*(xhi+xlo) + 0.5*(xhi-xlo)*z

        return z

    def loadtxt(self):
        meshfile = self.cfg.lookupexpr(msect, 'file', 'mesh.txt')
        xmesh = np.sort(
            np.loadtxt(meshfile, dtype=self.cfg.dtype, comments='#').ravel())
        assert len(xmesh)>=2, "Need atleast two points to define the mesh"
        return xmesh

    _meshTypes = {
        'classic': classic,
        'expr': expr,
        'glj': glj,
        'bump': bump,
        'loadtxt': loadtxt
    }

    """Defines a simple 1D mesh"""
    def __init__(self, cfg):
        self.cfg = cfg

        mt = cfg.lookupordefault(msect, 'type', 'classic')
        assert mt in self._meshTypes, "Valid mesh:"+str(self._meshTypes.keys())

        self._meshType = self._meshTypes[mt]

        # define 1D mesh (We just need a sorted point distribution)
        xmesh = self._meshType(self)

        # non-dimensionalize the domain
        self._H0 = cfg.lookupfloat(msect, 'H0')
        xmesh /= self._H0

        # number of elements
        self._Ne = len(xmesh) - 1

        # if this is an mpi process with more than one rank
        comm, rank, _ = get_comm_rank_root()
        nranks = comm.size
        perProcNe = self._Ne//nranks
        assert perProcNe*nranks == self._Ne, "Non-uniform number of elements!"
        sidx, eidx = rank*perProcNe, (rank+1)*perProcNe+1
        xmesh = xmesh[sidx:eidx]
        self._Ne = perProcNe
        print("Elements per proc", self._Ne)

        # length of the domain
        self._xlo, self._xhi = np.min(xmesh), np.max(xmesh)
        
        # size of each element 
        h = np.diff(xmesh).reshape(-1,1)
        assert np.min(h)>1e-8, "Too small element size"

        # jacobian of the mapping from D^{st}=[-1,1] to D
        self._jac, self._invjac = h/2., 2./h

        # the primitive mesh points
        self._xmesh = xmesh
        
    @property
    def xlo(self): return self._xlo

    @property
    def xhi(self): return self._xhi

    @property
    def Ne(self): return self._Ne

    @property
    def jac(self): return self._jac

    @property
    def invjac(self): return self._invjac

    @property
    def xmesh(self): return self._xmesh

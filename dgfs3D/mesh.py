import numpy as np
from dgfs1D.nputil import npeval, get_mpi, get_comm_rank_root
from dgfs1D.quadratures import zwglj

msect = 'mesh'

class Mesh(object):

    def classic(self):
        dim = self.cfg.dim
        xlo = self.cfg.lookupfloat_list(msect, 'xlo')
        xhi = self.cfg.lookupfloat_list(msect, 'xhi')
        assert all(map(lambda a: a[0]<a[1], zip(xlo,xhi))), "xlo < xhi ?"

        Ne = self.cfg.lookupint_list(msect, 'Ne')
        assert all(Ne>0), "Need atleast one element to define a mesh"

        x = [0]*dim 
        for i in range(dim):
            x[i] = np.linspace(xlo[i], xhi[i], Ne[i]+1, dtype=self.cfg.dtype)

        return x, Ne, xlo, xhi

    _meshTypes = {
        'classic': classic,
    }

    """Defines a simple 1D mesh"""
    def __init__(self, cfg):
        self.cfg = cfg
        dim = self.cfg.dim 
        self._dim = dim

        mt = cfg.lookupordefault(msect, 'type', 'classic')
        assert mt in self._meshTypes, "Valid mesh:"+str(self._meshTypes.keys())

        self._meshType = self._meshTypes[mt]

        # define 1D mesh (We just need a sorted point distribution)
        xmesh, Ne, xlo, xhi = self._meshType(self)

        # non-dimensionalize the domain
        self._H0 = cfg.lookupfloat(msect, 'H0')
        xmesh /= self._H0

        # if this is an mpi process with more than one rank
        comm, rank, _ = get_comm_rank_root()
        nranks = comm.size
        assert nranks==1, "Not implemented "
        """
        perProcNe = self._Ne//nranks
        assert perProcNe*nranks == self._Ne, "Non-uniform number of elements!"
        sidx, eidx = rank*perProcNe, (rank+1)*perProcNe+1
        xmesh = xmesh[sidx:eidx]
        self._Ne = perProcNe
        print("Elements per proc", self._Ne)
        """
        self._Ne = Ne

        # length of the domain
        self._xlo, self._xhi = xlo, xhi
        
        # size of each element 
        h = [np.diff(xmesh[i]).reshape(-1,1) for i in range(dim)]
        assert all(map(lambda v: np.min(v)>1e-8, h)), "Too small element size"

        # jacobian of the mapping from D^{st}=[-1,1] to D
        self._jac = [hI/2. for hI in h]
        self._invjac = [2./hI for hI in h]

        # the primitive mesh points
        self._xmesh = xmesh
        
    @property
    def xlo(self): return self._xlo

    @property
    def dim(self): return self._dim

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

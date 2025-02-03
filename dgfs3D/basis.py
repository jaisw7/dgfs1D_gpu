from abc import ABCMeta, abstractmethod
import numpy as np
from dgfs1D.quadratures import (zwglj, zwgj, zwgrjm, 
                                ortho_basis_at, jac_ortho_basis_at, 
                                nodal_basis_at, jac_nodal_basis_at, 
                                jacobi, jacobi_diff)

#import pycuda.autoinit
from pycuda import compiler, gpuarray
import pycuda.driver as cuda
from dgfs1D.util import (get_kernel, filter_tol, ndkron, ndgrid,
                        get_mm_kernel, get_mm_proxycopy_kernel)

basissect = 'basis'

"""
To unify the nodal and modal schemes, we need to introduce certain 
redundant operations, for instance, forward/backward transform for nodal 
basis -- which is unneeded. Note, however, that we use the gimmik backend, 
and therefore the matrix multiplication operators for forward/backward 
transform for nodal basis "literally" turn to simple array copy operations. 
See bespoke get_mm_kernel with filter operations
"""

class Basis(object, metaclass=ABCMeta):
    basis_kind = None

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

        # Read parameters
        # number of local degrees of freedom in each element
        self._K1 = cfg.lookupint(basissect, 'K')
        self._K = self._K1**cfg.dim

        # Forward transformation 
        # transforms/projects the solution (at points) to coefficient (space)
        # ucoeff_{me} = invM_{ml} B_{lq} w_{q} usol_{qe}
        self._fwdTransKern = lambda *args: None

        # Backward transformation 
        # transforms the (polynomial) coefficients to solution (at points)
        # usol_{qe} = B_{mq} ucoeff_{me} 
        self._bwdTransKern = lambda *args: None

        # Backward transformation for faces
        # reconstructs the solution at face points
        # usolF_{qe} = Bf_{mq} ucoeff_{me} 
        self._bwdTransFaceKern = lambda *args: None

        # derivative operator (strong form) [Transposed]
        # ux_{le} = S_{lm} ucoeff_{me}
        self._derivKern = lambda *args: None

        # interpolation operator
        # transforms the coefficients to solution (at "desired" points)
        # usol_{qe} = B*_{mq} ucoeff_{me} (Note the *)
        self._interpKern = lambda *args: None

        # inverse matrix operation
        self._invMKern = lambda *args: None

    @property
    def fwdTransOp(self): return self._fwdTransKern

    @property
    def bwdTransOp(self): return self._bwdTransKern

    @property
    def bwdTransFaceOp(self): return self._bwdTransFaceKern

    @property
    def derivOp(self): return self._derivKern

    @property
    def interpOp(self): return self._interpKern

    @property
    def invMassOp(self): return self._invMKern

    @property
    def K(self): return self._K

    @property
    def Nq(self): return self._Nq

    @property
    def Nqf(self): return self._Nqf

    @property
    def z1(self): return self._z1

    @property
    def z(self): return self._z

    @property
    def liftOp(self): return self._liftKern

    @property
    def fwdTransMat(self): return self._fwdTransMat

    @property
    def interpMat(self): return self._interpMat

    @property
    def derivMat(self): return self._derivMat



# The classic sem nodal basis based on lagrange polynomials on GLL
class NodalSemGLL(Basis):
    basis_kind = 'nodal-sem-gll'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq1 = self._K1
        self._Nq = self._K
        dim = self.cfg.dim

        # the Gauss-Lobatto quadrature on standard interval
        z, w = zwglj(self._Nq1, 0., 0.)
        #self._z = np.diag(ndkron(*[np.diag(z.ravel())]*dim))
        #self._w = ndkron([[np.diag(w.ravel())]*dim]).ravel()
        self._w = np.diag(ndkron(*[np.diag(w.ravel())]*dim))
        self._z = [x.reshape(-1,1) for x in ndgrid(*([z]*dim))]
        #print(self._z); exit()
        self._z1 = z

        # define the lagrange nodal basis
        B = nodal_basis_at(self._K1, z, z).T
        self._B = ndkron(*([B]*dim))

        # define the mass matrix 
        # since the basis is identity, this just have weights on diagonal
        M = np.einsum("mq,q,lq->ml", B, w, B)
        invM = np.linalg.inv(M)
        self._M = np.einsum("mq,q,lq->ml", self._B, self._w, self._B)
        self._invM = np.linalg.inv(self._M)

        # Forward transform operator 
        self._fwdTransMat = self._B # identity

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        #zr = np.linspace(-1., 1., Nqr) # the reconstruction points
        zr, _ = zwglj(Nqr, 0., 0.) # the reconstruction points
        Br = nodal_basis_at(self._K1, z, zr).T  # at "recons" points
        self._interpMat = ndkron(*[Br]*dim)

        # define the {K-1} order derivative matrix 
        D = jac_nodal_basis_at(self._K1, z, z)[0]
        self._Sx = [0]*dim
        for i in range(dim):
            B_ = [B]*dim
            B_[i] = D
            self._Sx[dim-1-i] = ndkron(*B_)

        W = np.diag(self._w)
        self._derivMat = [S.T for S in self._Sx]

        # define the operator for reconstructing solution at faces
        #self._Nqf = 2**dim
        #zqf, _ = zwglj(2, 0., 0.)
        #Bqf = nodal_basis_at(self._K1, z, zqf).T
        #self._Bqf = ndkron(*([Bqf]*dim))
        #print(self._z); exit()

        # Number of quadrature points per face
        self._Nqf = self._Nq1**(dim-1)
        Nfaces = 2*dim
        idx = [0]*Nfaces
        self._Bqf = [np.zeros((self._Nq, self._Nqf)) for i in range(Nfaces)]
        for i in range(dim):
          idx[2*i] = np.where(np.isclose(self._z[i].ravel(), -1))[0]
          idx[2*i+1] = np.where(np.isclose(self._z[i].ravel(), 1))[0]
          self._Bqf[2*i][idx[2*i], range(len(idx[2*i]))] = 1
          self._Bqf[2*i+1][idx[2*i+1], range(len(idx[2*i+1]))] = 1
       
        self._Bqf = np.hstack(self._Bqf)

        # consistency check
        #lift = np.diag(np.zeros(self._K1))
        #lift[0,0] = -1; lift[-1,-1] = 1;
        #self._lift = [0]*dim
        #for i in range(dim):
        #    B_ = [M]*dim
        #    B_[i] = lift
        #    self._lift[i] = ndkron(*B_)
        #print([np.linalg.norm((c+b)*W-a) for a, b,c in zip(self._lift, self._Sx, self._derivMat)])
        #self._lift = np.hstack(self._lift)
        #self._lift = np.matmul(self._invM, self._lift)

        # lift matrix
        idx = [0]*Nfaces
        self._lift = [np.zeros((self._Nq, self._Nqf)) for i in range(Nfaces)]
        wf = np.diag(ndkron(*[np.diag(w.ravel())]*(dim-1)))
        for i in range(dim):
          idx[2*i] = np.where(np.isclose(self._z[i].ravel(), -1))[0]
          idx[2*i+1] = np.where(np.isclose(self._z[i].ravel(), 1))[0]
          self._lift[2*i][idx[2*i], range(len(idx[2*i]))] = wf
          self._lift[2*i+1][idx[2*i+1], range(len(idx[2*i+1]))] = wf
          #self._lift[2*i][idx[2*i], range(len(idx[2*i]))] = 1./wf
          #self._lift[2*i+1][idx[2*i+1], range(len(idx[2*i+1]))] = 1./wf

        self._lift = np.hstack(self._lift)
        #self.lift = self._lift.T
        self._lift = np.matmul(self._invM, self._lift).T
        #print(filter_tol(self._lift))
        #print(self._lift.shape)
        #print(wf)
        #print(idx)
        #exit(0)

        # prepare kernels
        # since B is identity, (fwdTrans, bwdTrans) just copy data
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(self._B.T)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = [get_mm_kernel(Dr) for Dr in self._derivMat]
        self._invMKern = get_mm_kernel(self._B.T)
        self._liftKern = get_mm_kernel(self._lift.T)

        #print([Dr for Dr in self._derivMat])
        #exit(0)


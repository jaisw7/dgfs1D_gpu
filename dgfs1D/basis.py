from abc import ABCMeta, abstractmethod
import numpy as np
from dgfs1D.quadratures import (zwglj, zwgj, zwgrjm, 
                                ortho_basis_at, jac_ortho_basis_at, 
                                nodal_basis_at, jac_nodal_basis_at, 
                                jacobi, jacobi_diff)

#import pycuda.autoinit
from pycuda import compiler, gpuarray
import pycuda.driver as cuda
from dgfs1D.util import (get_kernel, filter_tol, 
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
        self._K = cfg.lookupint(basissect, 'K')

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

        # kernels for performing multiplication by U and V
        # the svd of H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        self._uTransKerns = [lambda *args: None for k in range(self._K)]
        self._vTransKerns = [lambda *args: None for k in range(self._K)]

        # non-zero entries (non redundant computations) aka significant modes
        self._sigModes = [0]*self._K

        # interpolation operator
        # transforms the coefficients to solution (at "desired" points)
        # usol_{qe} = B*_{mq} ucoeff_{me} (Note the *)
        self._interpKern = lambda *args: None

        # inverse matrix operation
        self._invMKern = lambda *args: None

        # for computing cell average
        self.computeCellAvgKern = lambda *args: None
        self.extractDrLinKern = lambda *args: None

    @property
    def fwdTransOp(self): return self._fwdTransKern

    @property
    def bwdTransOp(self): return self._bwdTransKern

    @property
    def bwdTransFaceOp(self): return self._bwdTransFaceKern

    @property
    def derivOp(self): return self._derivKern

    @property
    def uTransOps(self): return self._uTransKerns

    @property
    def vTransOps(self): return self._vTransKerns

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
    def z(self): return self._z

    @property
    def gLD(self): return self._gLD

    @property
    def gRD(self): return self._gRD

    @property
    def fwdTransMat(self): return self._fwdTransMat

    @property
    def interpMat(self): return self._interpMat

    @property
    def derivMat(self): return self._derivMat

    @property
    def sigModes(self): return self._sigModes


# The classic orthonormal modal basis based on legendre polynomials
class ModalOrthonormalLegendre(Basis):
    basis_kind = 'modal-orthonormal-legendre'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq = self.cfg.lookupordefault(basissect, 'Nq', 
            np.int(np.ceil(1.5*self._K)))

        # define the orthonormal basis
        # the Gauss-Radau quadrature on standard interval
        self._z, self._w = zwgrjm(self._Nq, 0., 0.)

        # define the orthonormal basis
        self._B = ortho_basis_at(self._K, self._z)

        # define the mass matrix (see Karniadakis, page 122)
        # since the basis is orthonormal, this is identity
        # the operators have been simplified using this fact
        self._M = np.einsum("mq,q,lq->ml", self._B, self._w, self._B)
        self._invM = np.linalg.inv(self._M)
        assert np.allclose(self._M, np.eye(self._K, self._K)), (
            "Insufficient quadrature points")

        # Forward transform operator (Uses the fact that M is identity)
        self._fwdTransMat = np.einsum("mq,q->mq", self._B, self._w)

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        zr = np.linspace(-1, 1, Nqr) # the reconstruction points
        Br = ortho_basis_at(self._K, zr)   # basis at the reconstruction points
        #self._interpMat = np.einsum('kr,kq->rq', Br, self._fwdTransMat)
        self._interpMat = Br

        # the H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        H = np.einsum("mq,aq,bq,q->mab", self._B, self._B, self._B, self._w)
        invM_H = np.einsum("ml,mab->lab", self._invM, H)

        # svd decomposition
        # defines the U, V matrices, and identifies the significant modes
        U, V, sigModes = [0]*self._K, [0]*self._K, [0]*self._K
        for m in range(self._K):
            u, s, v = np.linalg.svd(invM_H[m,:,:])
            U[m], V[m] = u, np.dot(np.diag(s),v)
            U[m], V[m] = map(filter_tol, (U[m], V[m]))
            nnzUm = np.where(np.sum(U[m], axis=0)!=0)[0]
            nnzVm = np.where(np.sum(V[m], axis=1)!=0)[0]
            sigModes[m] = np.intersect1d(nnzUm, nnzVm)
        
        self._U, self._V, self._sigModes = U, V, sigModes

        # define the {K-1} order derivative matrix 
        D = jac_ortho_basis_at(self._K, self._z)[0]
        self._Sx = np.einsum("lq,q,mq->ml", self._B, self._w, D)

        # define the operator for reconstructing solution at faces
        self._Nqf = 2  # number of points used in reconstruction at faces
        zqf, _ = zwglj(self._Nqf, 0., 0.)
        self._Bqf = ortho_basis_at(self._K, zqf)

        # define the correction functions at the left and right boundaries
        self._gLD, self._gRD = self._Bqf[:,0], self._Bqf[:,-1]

        # prepare kernels
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(self._B.T)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = get_mm_kernel(self._Sx.T)
        self._invMKern = get_mm_kernel(self._invM.T)

        # U, V operator kernels
        for m in range(self._K):
            self._uTransKerns[m] = get_mm_kernel(self._U[m].T)
            self._vTransKerns[m] = get_mm_kernel(self._V[m])



# The classic sem nodal basis based on lagrange polynomials on GLL
class NodalSemGLL(Basis):
    basis_kind = 'nodal-sem-gll'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq = self._K

        # the Gauss-Lobatto quadrature on standard interval
        self._z, self._w = zwglj(self._Nq, 0., 0.)

        # define the lagrange nodal basis
        self._B = nodal_basis_at(self._K, self._z, self._z).T

        # define the mass matrix 
        # since the basis is identity, this just have weights on diagonal
        self._M = np.einsum("mq,q,lq->ml", self._B, self._w, self._B)
        self._invM = np.linalg.inv(self._M)

        # Forward transform operator 
        self._fwdTransMat = self._B # identity

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        #zr = np.linspace(-1., 1., Nqr) # the reconstruction points
        zr, _ = zwglj(Nqr, 0., 0.) # the reconstruction points
        Br = nodal_basis_at(self._K, self._z, zr).T  # at "recons" points
        #self._interpMat = np.einsum('kr,kq->rq', Br, self._fwdTransMat)
        self._interpMat = Br

        # the H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        H = np.einsum("mq,aq,bq,q->mab", self._B, self._B, self._B, self._w)
        invM_H = np.einsum("ml,mab->lab", self._invM, H)
        #invM_H = H

        # svd decomposition
        # defines the U, V matrices, and identifies the significant modes
        U, V, sigModes = [0]*self._K, [0]*self._K, [0]*self._K
        for m in range(self._K):
            u, s, v = np.linalg.svd(invM_H[m,:,:])
            U[m], V[m] = u, np.dot(np.diag(s),v)
            U[m], V[m] = map(filter_tol, (U[m], V[m]))
            nnzUm = np.where(np.sum(U[m], axis=0)!=0)[0]
            nnzVm = np.where(np.sum(V[m], axis=1)!=0)[0]
            sigModes[m] = np.intersect1d(nnzUm, nnzVm)
        
        self._U, self._V, self._sigModes = U, V, sigModes

        # define the {K-1} order derivative matrix 
        D = jac_nodal_basis_at(self._K, self._z, self._z)[0]
        self._Sx = D #np.matmul(self._M, D.T)
        self._derivMat = self._Sx.T

        # define the operator for reconstructing solution at faces
        self._Nqf = 2  # number of points used in reconstruction at faces
        zqf, _ = zwglj(self._Nqf, 0., 0.)
        self._Bqf = nodal_basis_at(self._K, self._z, zqf).T

        # define the correction functions at the left and right boundaries
        self._gLD = self._Bqf[:,0]/self._w[0] 
        self._gRD = self._Bqf[:,-1]/self._w[-1]
        #gLD, gRD = invM[0,:], invM[-1,:]

        # prepare kernels
        # since B is identity, (fwdTrans, bwdTrans) just copy data
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(self._B.T)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = get_mm_kernel(self._Sx.T)
        self._invMKern = get_mm_kernel(self._B.T)

        #self._fwdTransKern = get_mm_proxycopy_kernel(self._fwdTransMat)
        #self._bwdTransKern = get_mm_proxycopy_kernel(self._B.T)
        #self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        #self._derivKern = get_mm_kernel(self._Sx.T)
        #self._invMKern = get_mm_proxycopy_kernel(self._B.T)

        # U, V operator kernels
        for m in range(self._K):
            self._uTransKerns[m] = get_mm_kernel(self._U[m].T)
            self._vTransKerns[m] = get_mm_kernel(self._V[m])

        # operators for limiting
        V = ortho_basis_at(self._K, self._z).T;
        invV = np.linalg.inv(V); 
        
        computeCellAvgOp = V.copy(); computeCellAvgOp[:,1:] = 0;
        computeCellAvgOp = np.matmul(computeCellAvgOp, invV);
        
        extractLinOp = V.copy(); extractLinOp[:,2:] = 0; 
        extractLinOp = np.matmul(extractLinOp, invV);
        extractDrLinOp = np.matmul(D.T, extractLinOp);

        self.computeCellAvgKern, self.extractDrLinKern = map(get_mm_kernel, 
            (computeCellAvgOp, extractDrLinOp)
        )

        """
        uhr = np.array([[0.8147,    0.9134,    0.2785], 
                        [0.9058,    0.6324,    0.5469],
                        [0.1270,    0.0975,    0.9575]]);
        print(computeCellAvgOp); print(np.matmul(computeCellAvgOp, uhr))
        #print(extractDrLinOp); print(np.matmul(extractDrLinOp, uhr))
        d_uhr = gpuarray.to_gpu(uhr.ravel());
        d_out = gpuarray.to_gpu(uhr.ravel());
        block = (128, 1, 1)
        grid = (1, 1)
        self.computeCellAvgKern.prepared_call(
                grid, block, 3, d_uhr.ptr, 3, d_out.ptr, 3)
        print(d_uhr.get())
        print(d_out.get())
        exit()
        """


# The nodal DG due to Hasthaven.
class NodalGLL(Basis):
    basis_kind = 'nodal-gll'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq = self._K

        # the Gauss-Lobatto quadrature on standard interval
        self._z, self._w = zwglj(self._Nq, 0., 0.)

        # define the lagrange nodal basis
        #self._B = nodal_basis_at(self._K, self._z, self._z).T
        self._B = ortho_basis_at(self._K, self._z).T

        # define the mass matrix 
        # since the basis is identity, this just have weights on diagonal
        self._invM = np.matmul(self._B, self._B.T)
        self._M = np.linalg.inv(self._invM)

        # Forward transform operator 
        I = np.eye(self._K)
        self._fwdTransMat = I # identity

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        zr, _ = zwglj(Nqr, 0., 0.) # the reconstruction points
        Br = nodal_basis_at(self._K, self._z, zr).T  # at "recons" points
        self._interpMat = Br

        # the H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        #H = np.einsum("mq,aq,bq,q->mab", self._B, self._B, self._B, self._w)
        #invM_H = np.einsum("ml,mab->lab", self._invM, H)
        # this is an approximation ... Here we have used mass lumping.
        H = invM_H = np.einsum("mq,aq,bq->mab", I, I, I)

        # svd decomposition
        # defines the U, V matrices, and identifies the significant modes
        U, V, sigModes = [0]*self._K, [0]*self._K, [0]*self._K
        for m in range(self._K):
            u, s, v = np.linalg.svd(invM_H[m,:,:])
            U[m], V[m] = u, np.dot(np.diag(s),v)
            U[m], V[m] = map(filter_tol, (U[m], V[m]))
            nnzUm = np.where(np.sum(U[m], axis=0)!=0)[0]
            nnzVm = np.where(np.sum(V[m], axis=1)!=0)[0]
            sigModes[m] = np.intersect1d(nnzUm, nnzVm)
        
        self._U, self._V, self._sigModes = U, V, sigModes

        # define the {K-1} order derivative matrix 
        D = jac_nodal_basis_at(self._K, self._z, self._z)[0]
        self._Sx = D 
        self._derivMat = self._Sx.T

        # define the operator for reconstructing solution at faces
        self._Nqf = self._K  # number of points used in reconstruction at faces
        self._Bqf = I

        # define the correction functions at the left and right boundaries
        self._gLD, self._gRD = self._invM[0,:], self._invM[-1,:]

        # prepare kernels
        # fwdTrans, bwdTrans just copy data
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(I)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = get_mm_kernel(self._Sx.T)
        self._invMKern = get_mm_kernel(I)

        # U, V operator kernels
        for m in range(self._K):
            self._uTransKerns[m] = get_mm_kernel(self._U[m].T)
            self._vTransKerns[m] = get_mm_kernel(self._V[m])

        # operators for limiting
        V = ortho_basis_at(self._K, self._z).T;
        invV = np.linalg.inv(V); 
        
        computeCellAvgOp = V.copy(); computeCellAvgOp[:,1:] = 0;
        computeCellAvgOp = np.matmul(computeCellAvgOp, invV);
        
        extractLinOp = V.copy(); extractLinOp[:,2:] = 0; 
        extractLinOp = np.matmul(extractLinOp, invV);
        extractDrLinOp = np.matmul(D.T, extractLinOp);

        self.computeCellAvgKern, self.extractDrLinKern = map(get_mm_kernel, 
            (computeCellAvgOp, extractDrLinOp)
        )

 


# The "modified" modal basis based on Karniadakis 1999
class ModalModifiedKarniadakis(Basis):
    basis_kind = 'modal-modified-karniadakis'

    def modified_basis_at(self, order, pts):
        jp = jacobi(order - 1, 1., 1., pts)
        return np.vstack((0.5*(1-pts), 
            [jp[p]*0.25*(1-pts**2) for p in range(order-2)], 
            0.5*(1+pts)))

    def jac_modified_basis_at(self, order, pts):
        jp = jacobi(order - 1, 1., 1., pts)
        djp = jacobi_diff(order - 1, 1., 1., pts)
        return np.vstack((-0.5*np.ones(len(pts)), 
            [djp[p]*0.25*(1.-pts**2)-jp[p]*0.5*pts for p in range(order-2)], 
            0.5*np.ones(len(pts))))

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq = self.cfg.lookupordefault(basissect, 'Nq', 
            np.int(np.ceil(1.5*self._K)))

        # define the orthonormal basis
        # the Gauss quadrature on standard interval
        self._z, self._w = zwgrjm(self._Nq, 0., 0.)

        # define the modified basis (see Karniadakis, page 64)
        self._B = self.modified_basis_at(self._K, self._z)

        # define the mass matrix 
        self._M = np.einsum("mq,q,lq->ml", self._B, self._w, self._B)
        self._invM = np.linalg.inv(self._M)

        # Forward transform operator 
        self._fwdTransMat = np.einsum("ml,lq,q->mq", 
            self._invM, self._B, self._w)

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        zr = np.linspace(-1, 1, Nqr) # the reconstruction points
        Br = self.modified_basis_at(self._K, zr)   # basis at reconst points
        self._interpMat = Br

        # the H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        H = np.einsum("mq,aq,bq,q->mab", self._B, self._B, self._B, self._w)
        invM_H = np.einsum("ml,mab->lab", self._invM, H)

        # svd decomposition
        # defines the U, V matrices, and identifies the significant modes
        U, V, sigModes = [0]*self._K, [0]*self._K, [0]*self._K
        for m in range(self._K):
            u, s, v = np.linalg.svd(H[m,:,:])
            U[m], V[m] = u, np.dot(np.diag(s),v)
            U[m], V[m] = map(filter_tol, (U[m], V[m]))
            nnzUm = np.where(np.sum(U[m], axis=0)!=0)[0]
            nnzVm = np.where(np.sum(V[m], axis=1)!=0)[0]
            sigModes[m] = np.intersect1d(nnzUm, nnzVm)
        
        self._U, self._V, self._sigModes = U, V, sigModes

        # define the {K-1} order derivative matrix 
        D = self.jac_modified_basis_at(self._K, self._z)
        self._Sx = np.einsum("lq,q,mq->ml", self._B, self._w, D)
        #self._Sx = np.einsum("ml,mn->ln", self._invM, self._Sx)
        #self._Sx = np.einsum("lm,mn->ln", self._invM, self._Sx)

        # define the operator for reconstructing solution at faces
        self._Nqf = 2  # number of points used in reconstruction at faces
        zqf, _ = zwglj(self._Nqf, 0., 0.)
        self._Bqf = self.modified_basis_at(self._K, zqf)
        #self._gBqf = np.einsum("ml,mn->ln", self._invM, self._Bqf)

        # define the correction functions at the left and right boundaries
        #self._gLD, self._gRD = self._gBqf[:,0], self._gBqf[:,-1]
        self._gLD, self._gRD = self._Bqf[:,0], self._Bqf[:,-1]

        # prepare kernels
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(self._B.T)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = get_mm_kernel(self._Sx.T)
        self._invMKern = get_mm_kernel(self._invM.T)

        # U, V operator kernels
        for m in range(self._K):
            self._uTransKerns[m] = get_mm_kernel(self._U[m].T)
            self._vTransKerns[m] = get_mm_kernel(self._V[m])





# The classic sem nodal basis based on lagrange polynomials on GLL
class NodalSemGLL2(Basis):
    basis_kind = 'nodal-sem-gll-2'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # number of quadrature points inside each element
        self._Nq = self._K

        # the Gauss-Lobatto quadrature on standard interval
        self._z, self._w = zwglj(self._Nq, 0., 0.)

        # define the lagrange nodal basis
        self._B = nodal_basis_at(self._K, self._z, self._z).T

        # define the mass matrix 
        # since the basis is identity, this just have weights on diagonal
        self._M = np.einsum("mq,q,lq->ml", self._B, self._w, self._B)
        self._invM = np.linalg.inv(self._M)

        # Forward transform operator 
        self._fwdTransMat = self._B # identity

        # interpolation operator
        Nqr = self.cfg.lookupint(basissect, 'Nqr')
        #zr = np.linspace(-1., 1., Nqr) # the reconstruction points
        zr, _ = zwglj(Nqr, 0., 0.) # the reconstruction points
        Br = nodal_basis_at(self._K, self._z, zr).T  # at "recons" points
        #self._interpMat = np.einsum('kr,kq->rq', Br, self._fwdTransMat)
        self._interpMat = Br

        # the H tensor (see J. Comput. Phys. 378, 178-208, 2019)
        H = np.einsum("mq,aq,bq,q->mab", self._B, self._B, self._B, self._w)
        invM_H = np.einsum("ml,mab->lab", self._invM, H)
        invM_H = H

        # svd decomposition
        # defines the U, V matrices, and identifies the significant modes
        U, V, sigModes = [0]*self._K, [0]*self._K, [0]*self._K
        for m in range(self._K):
            u, s, v = np.linalg.svd(invM_H[m,:,:])
            U[m], V[m] = u, np.dot(np.diag(s),v)
            U[m], V[m] = map(filter_tol, (U[m], V[m]))
            nnzUm = np.where(np.sum(U[m], axis=0)!=0)[0]
            nnzVm = np.where(np.sum(V[m], axis=1)!=0)[0]
            sigModes[m] = np.intersect1d(nnzUm, nnzVm)
        
        self._U, self._V, self._sigModes = U, V, sigModes

        # define the {K-1} order derivative matrix 
        D = jac_nodal_basis_at(self._K, self._z, self._z)[0]
        self._Sx = np.matmul(self._M, D.T).T
        self._derivMat = self._Sx.T

        # define the operator for reconstructing solution at faces
        self._Nqf = 2  # number of points used in reconstruction at faces
        zqf, _ = zwglj(self._Nqf, 0., 0.)
        self._Bqf = nodal_basis_at(self._K, self._z, zqf).T

        # define the correction functions at the left and right boundaries
        self._gLD = self._Bqf[:,0] #/self._w[0] 
        self._gRD = self._Bqf[:,-1] #/self._w[-1]
        #gLD, gRD = invM[0,:], invM[-1,:]

        # prepare kernels
        # since B is identity, (fwdTrans, bwdTrans) just copy data
        self._fwdTransKern = get_mm_kernel(self._fwdTransMat)
        self._bwdTransKern = get_mm_kernel(self._B.T)
        self._bwdTransFaceKern = get_mm_kernel(self._Bqf.T)
        self._derivKern = get_mm_kernel(self._Sx.T)
        self._invMKern = get_mm_kernel(self._invM.T)

        # U, V operator kernels
        for m in range(self._K):
            self._uTransKerns[m] = get_mm_kernel(self._U[m].T)
            self._vTransKerns[m] = get_mm_kernel(self._V[m])

        # operators for limiting
        V = ortho_basis_at(self._K, self._z).T;
        invV = np.linalg.inv(V); 
        
        computeCellAvgOp = V.copy(); computeCellAvgOp[:,1:] = 0;
        computeCellAvgOp = np.matmul(computeCellAvgOp, invV);
        
        extractLinOp = V.copy(); extractLinOp[:,2:] = 0; 
        extractLinOp = np.matmul(extractLinOp, invV);
        extractDrLinOp = np.matmul(D.T, extractLinOp);

        self.computeCellAvgKern, self.extractDrLinKern = map(get_mm_kernel, 
            (computeCellAvgOp, extractDrLinOp)
        )
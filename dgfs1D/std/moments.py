# -*- coding: utf-8 -*-

import os
import numpy as np
import warnings
from dgfs1D.quadratures import ortho_basis_at
from dgfs1D.nputil import get_comm_rank_root, get_mpi

"""Write distribution moments for single-species systems"""
class DGFSMomWriterStd():
    def _compute_moments_1D(self, coeff):
        vm = self.vm
        cv = vm.cv()
        Nv = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr = molarMass0/molarMass0
        mcw = mr*cw

        ele_sol = self._bulksoln
        Nqr, Ne, Ns = ele_sol.shape
        K, Nqr = self.im.shape
        soln = coeff.get().reshape((K, Ne, Nv))
        soln = np.einsum('mr,mej->rej', self.im, soln)

        ele_sol.fill(0)

        #non-dimensional mass density
        ele_sol[:,:,0] = np.einsum('...j->...', soln)*mcw

        if(np.sum(ele_sol[:,:,0])) < 1e-10:
            warnings.warn("density below 1e-10", RuntimeWarning)
            return

        #non-dimensional velocities
        ele_sol[:,:,1] = np.einsum('rej,j->re', soln, cv[0,:])*mcw
        ele_sol[:,:,1] /= ele_sol[:,:,0]
        ele_sol[:,:,2] = np.einsum('rej,j->re', soln, cv[1,:])*mcw
        ele_sol[:,:,2] /= ele_sol[:,:,0]

        # peculiar velocity
        cx = cv[0,:].reshape((1,1,Nv))-ele_sol[:,:,1].reshape((Nqr,Ne,1))
        cy = cv[1,:].reshape((1,1,Nv))-ele_sol[:,:,2].reshape((Nqr,Ne,1))
        cz = cv[2,:].reshape((1,1,Nv))-np.zeros((Nqr,Ne,1))
        cSqr = cx*cx + cy*cy + cz*cz

        # non-dimensional temperature
        ele_sol[:,:,3] = np.einsum('...j,...j->...', soln,cSqr)*(2./3.*mcw*mr)
        ele_sol[:,:,3] /= ele_sol[:,:,0]

        # non-dimensional heat-flux
        ele_sol[:,:,4] = mr*np.einsum('...j,...j,...j->...', soln,cSqr,cx)*mcw
        ele_sol[:,:,5] = mr*np.einsum('...j,...j,...j->...', soln,cSqr,cy)*mcw

        # non-dimensional pressure-tensor components
        ele_sol[:,:,6] = 2*mr*np.einsum('...j,...j,...j->...', soln,cx,cx)*mcw
        ele_sol[:,:,7] = 2*mr*np.einsum('...j,...j,...j->...', soln,cy,cy)*mcw
        ele_sol[:,:,8] = 2*mr*np.einsum('...j,...j,...j->...', soln,cx,cy)*mcw

        # dimensional rho, ux, uy, T, qx, qy, Pxx, Pyy, Pxy
        ele_sol[:,:,0:9] *= np.array([rho0, u0, u0, T0, 
            0.5*rho0*(u0**3), 0.5*rho0*(u0**3), 
            0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2)]).reshape(1,1,9)

        # dimensional pressure
        ele_sol[:,:,9] = (
            (mr*vm.R0/molarMass0)*ele_sol[:,:,0]*ele_sol[:,:,3])


    def __init__(self, tcurr, im, xcoeff, coeff, vm, cfg, cfgsect, 
        suffix=None, extn='.txt'):

        self.vm = vm

        # Construct the solution writer
        self.basedir = cfg.lookuppath(cfgsect, 'basedir', '.', abs=True)
        self.basename = cfg.lookup(cfgsect, 'basename')
        if not self.basename.endswith(extn):
            self.basename += extn

        # these variables are computed
        privarmap = ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'Q:y', 'P:xx', 'P:yy', 
                    'P:xy', 'p']
        Ns = len(privarmap)
        self.fields = ", ".join(privarmap)

        # Output time step and next output time
        self.dt_out = cfg.lookupfloat(cfgsect, 'dt-out')
        self.tout_next = tcurr

        # get info
        _, Ne = xcoeff.shape

        # define the interpolation operator
        K, Nqr = im.shape
        self.im = im

        # size of the output
        self.leaddim = Nqr*Ne

        self.xsol = np.einsum('mr,me->re', self.im, xcoeff)
        self.xsol = self.xsol.T.reshape((-1,1))*vm.H0()

        # get the entire information
        comm, rank, root = get_comm_rank_root()
        self.xsol = comm.gather(self.xsol, root=root)
        if rank==root: self.xsol = np.vstack(self.xsol)

        # allocate variables
        self._bulksoln = np.empty((Nqr, Ne, Ns)) 

        # helps us to compute moments from restart file
        self(1e-10, tcurr, coeff) 

    def __call__(self, dt, tcurr, coeff):
        if abs(self.tout_next - tcurr) > 0.5*dt:
            return

        # compute the moments
        self._compute_moments_1D(coeff)

        # Write out the file
        fname = self.basename.format(t=tcurr)
        solnfname = os.path.join(self.basedir, fname)
        sol = self._bulksoln.swapaxes(0,1).reshape(self.leaddim,-1)   

        # get the entire information
        comm, rank, root = get_comm_rank_root()
        sol = comm.gather(sol, root=root)
        if rank==root:
            sol = np.vstack(sol)
            np.savetxt(solnfname, np.hstack((self.xsol, sol)), 
                header="t={0} \n x {1}".format(tcurr, self.fields), 
                comments="#")

        # Compute the next output time
        self.tout_next = tcurr + self.dt_out


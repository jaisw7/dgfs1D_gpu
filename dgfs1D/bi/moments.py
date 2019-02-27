# -*- coding: utf-8 -*-

import os
import numpy as np
import warnings
from dgfs1D.quadratures import ortho_basis_at
from dgfs1D.nputil import get_comm_rank_root, get_mpi

"""Write distribution moments for multi-species systems"""
class DGFSMomWriterBi():
    def _compute_moments_1D(self, coeffs):
        vm = self.vm
        cv = vm.cv()
        N = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr_ = vm.masses()
        nspcs = vm.nspcs()
        mcw_ = [mr_[p]*cw for p in range(nspcs)]

        self._bulksoln.fill(0.)
        ele_sol = self._bulksoln
        Nqr, Ne, Ns = ele_sol.shape
        NsPerSpcs = Ns//nspcs

        K, Nqr = self.im.shape
        fsoln = [coeffs[p].get().reshape((K, Ne, N)) for p in range(nspcs)]
        fsoln = [np.einsum('mr,mej->rej', self.im, fsoln[p]) 
                    for p in range(nspcs)]

        self._bulksolntot.fill(0.)
        propt = self._bulksolntot

        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = NsPerSpcs*p
            soln = fsoln[p]
            
            #non-dimensional mass density
            ele_sol[:,:,ispcs+0] = np.sum(soln, axis=-1)*mcw

            if(np.sum(ele_sol[:,:,ispcs+0])) < 1e-10:
                warnings.warn("density below 1e-10", RuntimeWarning)
                continue

            #non-dimensional velocities
            ele_sol[:,:,ispcs+1] = np.tensordot(soln, cv[0,:], 
                    axes=(-1,0))*mcw
            ele_sol[:,:,ispcs+1] /= ele_sol[:,:,ispcs+0]
            ele_sol[:,:,ispcs+2] = np.tensordot(soln, cv[1,:], 
                   axes=(-1,0))*mcw
            ele_sol[:,:,ispcs+2] /= ele_sol[:,:,ispcs+0]

            # peculiar velocity for species
            cx = cv[0,:].reshape((1,1,N))-ele_sol[:,:,ispcs+1].reshape((Nqr,Ne,1))
            cy = cv[1,:].reshape((1,1,N))-ele_sol[:,:,ispcs+2].reshape((Nqr,Ne,1))
            cz = cv[2,:].reshape((1,1,N))-np.zeros((Nqr,Ne,1))
            cSqr = cx*cx + cy*cy + cz*cz

            # non-dimensional temperature
            ele_sol[:,:,ispcs+3] = np.sum(soln*cSqr, axis=-1)*(2.0/3.0*mcw*mr)
            ele_sol[:,:,ispcs+3] /= ele_sol[:,:,ispcs+0]

            # total mass density
            propt[:,:,0] += ele_sol[:,:,ispcs+0]

            # total velocity
            propt[:,:,1] += ele_sol[:,:,ispcs+0]*ele_sol[:,:,ispcs+1]
            propt[:,:,2] += ele_sol[:,:,ispcs+0]*ele_sol[:,:,ispcs+2]


        # now we compute properties requiring the total velocity

        # normalize the total velocity
        propt[:,:,1] /= propt[:,:,0]
        propt[:,:,2] /= propt[:,:,0]

        # peculiar velocity
        cx = cv[0,:].reshape((1,1,N))-propt[:,:,1].reshape((Nqr,Ne,1))
        cy = cv[1,:].reshape((1,1,N))-propt[:,:,2].reshape((Nqr,Ne,1))
        cz = cv[2,:].reshape((1,1,N))-np.zeros((Nqr,Ne,1))
        cSqr = cx*cx + cy*cy + cz*cz

        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = NsPerSpcs*p
            soln = fsoln[p]

            if(p==0):
                if(np.sum(propt[:,:,0])) < 1e-10:
                    warnings.warn("density below 1e-10", RuntimeWarning)
                    continue

            # non-dimensional heat-flux
            ele_sol[:,:,ispcs+4] = mr*np.sum(soln*cSqr*cx, axis=-1)*mcw

            # dimensional rho, ux, uy, T, qx
            ele_sol[:,:,ispcs+0:ispcs+5] *= np.array([
                rho0, u0, u0, T0, 0.5*rho0*(u0**3)]).reshape(1,1,5)

            # dimensional pressure
            ele_sol[:,:,ispcs+5] = (
                (mr*vm.R0/molarMass0)
                *ele_sol[:,:,ispcs+0]*ele_sol[:,:,ispcs+3])

            # dimensional number density
            ele_sol[:,:,ispcs+6] = (
                (vm.NA/mr/molarMass0)*ele_sol[:,:,ispcs+0])



    def __init__(self, tcurr, im, xcoeff, coeffs, vm, cfg, cfgsect, 
        suffix=None, extn='.txt'):

        self.vm = vm

        # Construct the solution writer
        self.basedir = cfg.lookuppath(cfgsect, 'basedir', '.', abs=True)
        self.basename = cfg.lookup(cfgsect, 'basename')
        if not self.basename.endswith(extn):
            self.basename += extn

        # these variables are computed
        privarmap = ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'p', 'nden']
        lv = len(privarmap)
        newvar = []
        for p in range(vm.nspcs()):
            newvar.extend(privarmap)
            for ivar in range(-1,-lv-1,-1): newvar[ivar] += ':'+str(p)
        privarmap = newvar

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
        self._bulksolntot = np.empty((Nqr, Ne, 4)) 

        # helps us to compute moments from restart file
        self(1e-10, tcurr, coeffs) 


    def __call__(self, dt, tcurr, coeffs):
        if abs(self.tout_next - tcurr) > 0.5*dt:
            return

        # compute the moments
        self._compute_moments_1D(coeffs)

        # Write out the file
        fname = self.basename.format(t=tcurr)
        solnfname = os.path.join(self.basedir, fname)
        sol = self._bulksoln.swapaxes(0,1).reshape(self.leaddim,-1)   

        # get the entire information
        comm, rank, root = get_comm_rank_root()
        sol = comm.gather(sol, root=root)
        if rank==root:
            sol = np.vstack(sol)
            np.savetxt(solnfname, np.hstack((self.xsol, sol)), "%.5e",
                header="t={0} \nx {1}".format(tcurr, self.fields), 
                comments="#")

        # Compute the next output time
        self.tout_next = tcurr + self.dt_out


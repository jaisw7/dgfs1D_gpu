# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5py
import os
from dgfs1D.nputil import get_comm_rank_root
from dgfs1D.util import check

"""Write distribution slice for single-species systems"""
class DGFSDistributionSliceStd():

    def reconstruct(self, f):
        FTf = np.fft.fft3d(f)
        Nv = self.Nv
        Nvr = self.Nvr
        l, lr = self.l, self.lr

        recon = np.empty((Nvr, Nvr, Nvr))
        for jr in range(Nvr*Nvr*Nvr):
            temp = 2*np.pi*(l[0,:]*lr[0,jr]+l[1,:]*lr[1,jr]+l[2,:]*lr[2,jr])
            temp = FTf.real*cos(temp) - FTf.imag*sin(temp)
            recon[jr] = sum(temp)/self.vm.vsize()

    
    def get_cl(self,N):        
        l0 = np.concatenate((np.arange(0,N/2), np.arange(-N/2, 0)))
        l = np.zeros((3, N*N*N))
        for idv in range(N*N*N):
            I = int(idv/(N*N))
            J = int((idv%(N*N))/N)
            K = int((idv%(N*N))%N)
            l[0,idv] = l0[I];
            l[1,idv] = l0[J];
            l[2,idv] = l0[K];
        
        return l


    def __init__(self, tcurr, shape, im, vm, cfg, cfgsect, extn='.txt'):

        self.K, self.Ne, _ = shape
        self.vm = vm
        self.Nv = self.vm.Nv()

        # Construct the solution writer
        self.basedir = cfg.lookuppath(cfgsect, 'basedir', '.', abs=True)
        self.basename = cfg.lookup(cfgsect, 'basename')
        if not self.basename.endswith(extn):
            self.basename += extn

        # Output time step and next output time
        self.dt_out = cfg.lookupfloat(cfgsect, 'dt-out')
        self.tout_next = tcurr + self.dt_out

        # define the interpolation operator
        _, self.Nqr = im.shape
        self.im = im

        # Find total Ne
        comm, rank, _ = get_comm_rank_root()
        nranks = comm.size
        self.Ne_t = self.Ne*nranks

        # l on the current velocity mesh
        self.l = self.get_cl(self.Nv)

        # l on the reconstructed mesh
        self.Nvr = cfg.lookupordefault(cfgsect, 'Nvr', self.Nv)
        self.lr = self.get_cl(self.Nvr)

        # The spatial location where the slice is to be obtained
        self.Ne_out = cfg.lookupint(cfgsect, 'Ne-out')
        check(self.Ne_out < self.Ne_t, "Ne_out should be within [0, Ne)")

        self.Nqr_out = cfg.lookupint(cfgsect, 'Nqr-out')
        check(self.Nqr_out < self.Nqr, "Ne_out should be within [0, Nqr)")

        self.Nvx_out = cfg.lookupordefault(cfgsect, 'Nvx-out', -1)
        self.Nvy_out = cfg.lookupordefault(cfgsect, 'Nvy-out', -1)
        self.Nvz_out = cfg.lookupordefault(cfgsect, 'Nvz-out', -1)

        if self.Nvx_out==-1 and self.Nvy_out==-1 and self.Nvz_out==-1:
            raise ValueError(
                '''2 of the three variables (Nvx-out, Nvy-out, Nvz-out)
                should be provided''')

        if self.Nvx_out!=-1 and self.Nvy_out!=-1 and self.Nvz_out!=-1:
            raise ValueError(
                '''Exactly 2 of the three variables (Nvx-out, Nvy-out, Nvz-out)
                should be provided''')

        # On which processor does this spatial location lie?
        self._thisRank = self.Ne_out//self.Ne

        # allocate variables
        if self.Nvx_out == -1:
            check(self.Nvy_out<self.Nv and self.Nvy_out>=0, 
                "Nvy-out should be between [0, Ne)")
            check(self.Nvz_out<self.Nv and self.Nvz_out>=0, 
                "Nvz-out should be between [0, Ne)")
            self._vslice = self.vm.cv()[0].reshape(
                (self.Nv, self.Nv, self.Nv))[:, self.Nvy_out, self.Nvz_out]

        elif self.Nvy_out == -1:
            check(self.Nvx_out<self.Ne_t and self.Nvx_out>=0, 
                "Nvx-out should be between [0, Ne)")
            check(self.Nvz_out<self.Ne_t and self.Nvz_out>=0, 
                "Nvz-out should be between [0, Ne)")
            self._vslice = self.vm.cv()[1].reshape(
                (self.Nv, self.Nv, self.Nv))[self.Nvx_out, :, self.Nvz_out]

        elif self.Nvz_out == -1:
            check(self.Nvx_out<self.Ne_t and self.Nvx_out>=0, 
                "Nvx-out should be between [0, Ne)")
            check(self.Nvy_out<self.Ne_t and self.Nvy_out>=0, 
                "Nvy-out should be between [0, Ne)")
            self._vslice = self.vm.cv()[2].reshape(
                (self.Nv, self.Nv, self.Nv))[self.Nvx_out, self.Nvy_out, :]


    def __call__(self, dt, tcurr, coeff):
        if abs(self.tout_next - tcurr) > 0.5*dt:
            return

        # Write out the file
        fname = self.basename.format(t=tcurr)
        solnfname = os.path.join(self.basedir, fname)

        # get the entire information
        comm, rank, root = get_comm_rank_root()        
        if rank==self._thisRank:
            soln = coeff.get().reshape((self.K, self.Ne, self.vm.vsize()))
            soln = np.einsum('mr,mej->rej', self.im, soln)

            soln = soln[self.Nqr_out, self.Ne_out, :]
            soln = soln.reshape((self.Nv, self.Nv, self.Nv))

            if self.Nvx_out == -1:
                soln = soln[:, self.Nvy_out, self.Nvz_out]
            elif self.Nvy_out == -1:
                soln = soln[self.Nvx_out, :, self.Nvz_out]
            elif self.Nvz_out == -1:
                soln = soln[self.Nvx_out, self.Nvy_out, :]

            np.savetxt(solnfname, np.vstack((self._vslice, soln)).T, 
                comments="#")
        
        # Compute the next output time
        self.tout_next = tcurr + self.dt_out
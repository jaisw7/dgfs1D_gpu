# -*- coding: utf-8 -*-

import numpy as np
from pycuda import gpuarray
from dgfs1D.nputil import get_comm_rank_root, get_mpi

"""Write distribution norm in single-species systems"""
class DGFSResidualStd():
    
    def __init__(self, cfg, cfgsect):

        self.nsteps = cfg.lookupint(cfgsect, 'nsteps')

    def __call__(self, tcurr, nsteps, solprev, solcurr):
        if nsteps % self.nsteps == 0: 
            comm, rank, root = get_comm_rank_root()
            res = np.array([gpuarray.sum((solcurr-solprev)**2).get(), 
                        gpuarray.sum((solprev)**2).get()])

            if rank != root:
                comm.Reduce(res, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
                print("residual at t = ", tcurr, np.sqrt(res[0]/res[1]))
                
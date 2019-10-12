# -*- coding: utf-8 -*-

import numpy as np
from pycuda import gpuarray
from dgfs1D.nputil import get_comm_rank_root, get_mpi

"""Write distribution norm in single-species systems"""
class DGFSResidualBi():
    
    def __init__(self, cfg, cfgsect):

        self.nsteps = cfg.lookupint(cfgsect, 'nsteps')

    def __call__(self, tcurr, nsteps, solprevs, solcurrs):
        if (nsteps % self.nsteps == 0) or nsteps==1: 
            comm, rank, root = get_comm_rank_root()
            res_num = np.array([gpuarray.sum((solcurr-solprev)**2).get() 
                        for solprev, solcurr in zip(solprevs, solcurrs)])
            res_den = np.array([gpuarray.sum((solprev)**2).get()
                        for solprev in solprevs ])
            
            if rank != root:
                comm.Reduce(res_num, None, op=get_mpi('sum'), root=root)
                comm.Reduce(res_den, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), res_num, op=get_mpi('sum'),
                            root=root)
                comm.Reduce(get_mpi('in_place'), res_den, op=get_mpi('sum'),
                            root=root)
                print("residual at t = ", tcurr, np.sqrt(res_num/res_den))
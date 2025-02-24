# -*- coding: utf-8 -*-

import numpy as np
from pycuda import gpuarray
from dgfs1D.nputil import get_comm_rank_root, get_mpi
from loguru import logger

"""Write distribution norm in single-species systems"""
class DGFSResidualStd():

    def __init__(self, cfg, cfgsect):

        self.nsteps = cfg.lookupordefault(cfgsect, 'nsteps', -1)

    def __call__(self, tcurr, nsteps, solprev, solcurr):
        if ((self.nsteps>0 and ((nsteps % self.nsteps == 0) or nsteps==1))):
        #    or (self.dt_out>0 and abs(tcurr % self.dt_out) < 1e-8)):

            comm, rank, root = get_comm_rank_root()
            diff = solcurr-solprev
            res = np.array([gpuarray.dot(diff, diff).get(),
                        gpuarray.dot(solprev, solprev).get()])

            if rank != root:
                comm.Reduce(res, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), res, op=get_mpi('sum'),
                            root=root)
                logger.info("residual at t = {}", tcurr, np.sqrt(res[0]/res[1]))
                #print("residual at t = ", tcurr, np.sqrt(res[2]))
                #print("residual at t = ", tcurr, np.linalg.norm(solcurr.get()))

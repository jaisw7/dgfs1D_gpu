# -*- coding: utf-8 -*-

""" 
DGFS in one dimension
"""

import os
import numpy as np
import itertools as it
from timeit import default_timer as timer

import mpi4py.rc
mpi4py.rc.initialize = False

from dgfs1D.initialize import initialize
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup, ndrange)
from dgfs1D.nputil import get_comm_rank_root, get_local_rank, get_mpi
from dgfs1D.basis import Basis
from dgfs1D.std.velocitymesh import DGFSVelocityMeshStd
from dgfs1D.std.scattering import DGFSScatteringModelStd
from dgfs1D.std.initcond import DGFSInitConditionStd
from dgfs1D.std.moments import DGFSMomWriterStd
from dgfs1D.std.residual import DGFSResidualStd
from dgfs1D.std.distribution import DGFSDistributionStd
from dgfs1D.std.bc import DGFSBCStd
from dgfs1D.axnpby import get_axnpby_kerns
from dgfs1D.util import get_kernel, filter_tol, check
from dgfs1D.std.integrator import DGFSIntegratorStd

from pycuda import compiler, gpuarray
import pycuda.driver as cuda
from gimmik import generate_mm

def main():
    # who am I in this world? (Bulleh Shah, 18th century sufi poet)
    comm, rank, root = get_comm_rank_root()

    # read the inputs (from people)
    cfg, mesh, args = initialize()

    # define 1D mesh (construct a 1D world view)
    xmesh = mesh.xmesh

    # number of elements (how refined perspectives do we want/have?)
    Ne = mesh.Ne

    # define the basis (what is the basis for those perspectives?)
    bsKind = cfg.lookup('basis', 'kind')
    basiscls = subclass_where(Basis, basis_kind=bsKind)
    basis = basiscls(cfg)

    # number of local degrees of freedom (depth/granualirity of perspectives)
    K = basis.K

    # number of solution points (how far can I interpolate my learning)
    Nq = basis.Nq

    # left/right face maps
    Nqf = basis.Nqf  # number of points used in reconstruction at faces
    mapL, mapR = np.arange(Ne+1)+(Nqf-1)*Ne-1, np.arange(Ne+1)
    mapL[0], mapR[-1] = 0, Ne*Nqf-1
    Nf = len(mapL)

    # the zeros
    z = basis.z
    
    # jacobian of the mapping from D^{st}=[-1,1] to D
    jac, invjac = mesh.jac, mesh.invjac

    # load the velocity mesh
    vm = DGFSVelocityMeshStd(cfg)
    Nv = vm.vsize()

    # load the scattering model
    smn = cfg.lookup('scattering-model', 'type')
    scatteringcls = subclass_where(DGFSScatteringModelStd, 
        scattering_model=smn)
    sm = scatteringcls(cfg, vm, Ne=Ne)

    # initial time, time step, final time
    ti, dt, tf = cfg.lookupfloats('time-integrator', ('tstart', 'dt', 'tend'))
    nsteps = np.ceil((tf - ti)/dt)
    dt = (tf - ti)/nsteps

    # Compute the location of the solution points 
    xsol = np.array([0.5*(xmesh[j]+xmesh[j+1])+jac[j]*z for j in range(Ne)]).T
    xcoeff = np.einsum("kq,qe->ke", basis.fwdTransMat, xsol)

    # Determine the grid/block
    NeNv = Ne*Nv
    KNeNv = K*Ne*Nv
    NqNeNv = Nq*Ne*Nv
    NqfNeNv = Nqf*Ne*Nv
    NfNv = Nf*Nv
    block = (128, 1, 1)
    grid_Nv = get_grid_for_block(block, Nv)
    grid_NeNv = get_grid_for_block(block, Ne*Nv)
    grid_KNeNv = get_grid_for_block(block, K*Ne*Nv)

    # operator generator for matrix operations
    matOpGen = lambda v: lambda arg0, arg1: v.prepared_call(
                grid_NeNv, block, NeNv, arg0.ptr, NeNv, arg1.ptr, NeNv)
    
    # forward trans, backward, backward (at faces), derivative kernels
    fwdTrans_Op, bwdTrans_Op, bwdTransFace_Op, deriv_Op, invMass_Op = map(
        matOpGen, (basis.fwdTransOp, basis.bwdTransOp, 
            basis.bwdTransFaceOp, basis.derivOp, basis.invMassOp)
    )

    # U, V operator kernels
    trans_U_Op = tuple(map(matOpGen, basis.uTransOps))
    trans_V_Op = tuple(map(matOpGen, basis.vTransOps))

    # prepare the kernel for extracting face/interface values
    dfltargs = dict(
        K=K, Ne=Ne, Nq=Nq, vsize=Nv, dtype=cfg.dtypename,
        mapL=mapL, mapR=mapR, offsetL=0, offsetR=len(mapR)-1,
        invjac=invjac, gRD=basis.gRD, gLD=basis.gLD)
    kernsrc = DottedTemplateLookup('dgfs1D.std.kernels', 
                                    dfltargs).get_template('std').render()
    kernmod = compiler.SourceModule(kernsrc)

    # prepare operators for execution (see std.mako for description)
    (extLeft_Op, extRight_Op, transferBC_L_Op, transferBC_R_Op, 
        insertBC_L_Op, insertBC_R_Op) = map(lambda v: 
        lambda *args: get_kernel(kernmod, v, 'PP').prepared_call(
            grid_Nv, block, *list(map(lambda c: c.ptr, args))
        ), ("extract_left", "extract_right", "transfer_bc_left", 
            "transfer_bc_right", "insert_bc_left", "insert_bc_right")
    )

    # The boundary conditions (by default all boundaries are processor bnds)
    bcl_type, bcr_type = 'dgfs-periodic', 'dgfs-periodic'

    # the mesh is decomposed in linear fashion, so rank 0 gets left boundary
    if rank==0: bcl_type = cfg.lookup('soln-bcs-xlo', 'type')

    # and the last rank comm.size-1 gets the right boundary
    if rank==comm.size-1:  bcr_type = cfg.lookup('soln-bcs-xhi', 'type')
    
    # prepare kernels for left boundary    
    bcl_cls = subclass_where(DGFSBCStd, type=bcl_type)
    bcl = bcl_cls(xmesh[0], -1., vm, cfg, 'soln-bcs-xlo')
    updateBC_L_Op = bcl.updateBCKern
    applyBC_L_Op = bcl.applyBCKern
    
    # prepare kernels for right boundary
    bcr_cls = subclass_where(DGFSBCStd, type=bcr_type)
    bcr = bcr_cls(xmesh[-1], 1., vm, cfg, 'soln-bcs-xhi')
    updateBC_R_Op = bcr.updateBCKern
    applyBC_R_Op = bcr.applyBCKern

    # flux kernel
    flux = get_kernel(kernmod, "flux", 'PPPPP')
    flux_Op = lambda d_uL, d_uR, d_jL, d_jR: flux.prepared_call(
            grid_Nv, block, 
            d_uL.ptr, d_uR.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # multiply the derivative by the advection velocity
    mulbyadv = get_kernel(kernmod, "mul_by_adv", 'PP')
    mulbyadv_Op = lambda d_ux: mulbyadv.prepared_call(
                    grid_KNeNv, block, vm.d_cvx().ptr, d_ux.ptr)

    # multiply the coefficient by the inverse jacobian
    mulbyinvjac = get_kernel(kernmod, "mul_by_invjac", 'P')
    mulbyinvjac_Op = lambda d_ux: mulbyinvjac.prepared_call(
                    grid_Nv, block, d_ux.ptr)

    # \alpha AX + \beta Y kernel (for operations on coefficients)
    axnpbyCoeff = get_axnpby_kerns(2, range(K), NeNv, cfg.dtype)
    axnpbyCoeff_Op = lambda a0, x0, a1, x1: axnpbyCoeff.prepared_call(
                    grid_NeNv, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on physical solutions)
    axnpbySol = get_axnpby_kerns(2, range(Nq), NeNv, cfg.dtype)
    axnpbySol_Op = lambda a0, x0, a1, x1: axnpbySol.prepared_call(
                    grid_NeNv, block, x0.ptr, x1.ptr, a0, a1)

    # total flux kernel (sums up surface and volume terms)
    totalFlux = get_kernel(kernmod, "totalFlux", 'PPPP')
    totalFlux_Op = lambda d_ux, d_jL, d_jR: totalFlux.prepared_call(
            grid_Nv, block, d_ux.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # allocations on gpu
    d_usol = gpuarray.empty(NqNeNv, dtype=cfg.dtype)
    d_usolF = gpuarray.empty(NqfNeNv, dtype=cfg.dtype)
    d_uL = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_uR = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_jL = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_jR = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_bcL = gpuarray.empty(Nv, dtype=cfg.dtype) 
    d_bcR = gpuarray.empty(Nv, dtype=cfg.dtype) 
    d_bcT = gpuarray.empty(Nv, dtype=cfg.dtype) 
    d_ux = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_f = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_g = gpuarray.empty(KNeNv, dtype=cfg.dtype)

    d_ucoeff = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_ucoeffPrev = gpuarray.empty_like(d_ucoeff)

    # check if this is a new run
    if hasattr(args, 'process_run'):
        usol = np.empty((Nq, Ne, Nv), dtype=cfg.dtype)  # temporary storage

        # load the initial condition model
        icn = cfg.lookup('soln-ics', 'type')
        initcondcls = subclass_where(DGFSInitConditionStd, model=icn)
        ic = initcondcls(cfg, vm, 'soln-ics')
        ic.apply_init_vals(usol, Nq, Ne, xsol)

        # transfer the information to the gpu
        d_usol.set(usol.ravel())

        # forward transform to coefficient space
        fwdTrans_Op(d_usol, d_ucoeff)

    # check if we are restarting
    if hasattr(args, 'process_restart'):
        import h5py as h5py
        filename = args.dist.name + '_rank=' + str(rank)
        check(os.file.exists(filename), "Distribution missing")
        with h5py.File(args.dist.name, 'r') as h5f:
            dst = h5f['coeff']
            ti = dst.attrs['time']
            d_ucoeff.set(dst[:])
            check(dst.attrs['K']==K, "Inconsistent distribution K")
            check(dst.attrs['Ne']==Ne, "Inconsistent distribution Ne")
            check(dst.attrs['Nv']==Nv, "Inconsistent distribution N")

        # backward transform to solution space
        bwdTrans_Op(d_ucoeff, d_usol)
    
    # prepare the post-processing handlers    
    # For computing moments
    moments = DGFSMomWriterStd(ti, basis.interpMat, xcoeff, d_ucoeff, vm, cfg, 
        'dgfsmomwriter')

    # For computing residual
    residual = DGFSResidualStd(cfg, 'dgfsresidual')

    # For writing distribution function
    distribution = DGFSDistributionStd(ti, (K, Ne, Nv), cfg, 
        'dgfsdistwriter')

    # Actual algorithm

    # initialize
    axnpbyCoeff_Op(0., d_ucoeffPrev, 1., d_ucoeff)
    sigModes = basis.sigModes

    # define the neighbours
    from mpi4py import MPI
    down_nbr, up_nbr = comm.rank - 1, comm.rank + 1;
    if up_nbr >= comm.size: up_nbr = MPI.PROC_NULL
    if down_nbr < 0: down_nbr = MPI.PROC_NULL

    # define the ode rhs    
    def rhs(time, d_ucoeff_in, d_ucoeff_out):

        # reconstruct solution at faces
        bwdTransFace_Op(d_ucoeff_in, d_usolF)

        # Step:1 extract the solution at faces
        extLeft_Op(d_usolF, d_uL)
        extRight_Op(d_usolF, d_uR)

        # transfer left boundary information in send buffer
        transferBC_L_Op(d_uL, d_bcL)       # Transfer the left ghost BC info
        transferBC_R_Op(d_uR, d_bcR)       # Transfer the right ghost BC info

        # this can be adjusted in case of RDMA enabled MPI support
        h_bcL, h_bcR = d_bcL.get(), d_bcR.get()
        #h_bcL, h_bcR = map(lambda v: v.gpudata.as_buffer(v.nbytes), 
        #               (d_bcL, d_bcR))
        
        # send information
        req1 = comm.isend(d_bcR, dest=up_nbr)  # to upstream neighbour
        req2 = comm.isend(d_bcL, dest=down_nbr)  # to downstream neighbour 

        # recieve information
        h_bcL = comm.recv(source=down_nbr)  # from downstream neighbour
        h_bcR = comm.recv(source=up_nbr)    # from upstream neighbour
        MPI.Request.Waitall([req1, req2])
        
        # set information at left boundary
        if h_bcL:
            d_bcL.set(h_bcL)
        else:
            transferBC_L_Op(d_uL, d_bcL)  # Transfer the ghost BC info
        
        # set information at right boundary
        if h_bcR:
            d_bcR.set(h_bcR)
        else:
            transferBC_R_Op(d_uR, d_bcR)  # Transfer the ghost BC info
        
        # At left boundary        
        #transferBC_L_Op(d_uL, d_bcL)       # Transfer the ghost BC info
        updateBC_L_Op(d_bcL, time)         # now update boundary info 
        applyBC_L_Op(d_bcL, d_bcT, time)   # apply boundary condition
        insertBC_L_Op(d_bcT, d_uL)         # insert info to global face-flux

        # At right boundary        
        #transferBC_R_Op(d_uR, d_bcL)       # Transfer the ghost BC info
        updateBC_R_Op(d_bcR, time)         # now update boundary info 
        applyBC_R_Op(d_bcR, d_bcT, time)   # apply boundary condition
        insertBC_R_Op(d_bcT, d_uR)         # insert info to global face-flux

        # Step:2 Compute the flux and jumps (all operations in single call)
        #fL, fR = cvx*uL, cvx*uR
        #fupw = 0.5*(fL + fR) + 0.5*np.abs(cvx)*(uL - uR)
        #jL = fupw - fL  # Compute the jump at left boundary
        #jR = fupw - fR  # Compute the jump at right boundary
        flux_Op(d_uL, d_uR, d_jL, d_jR)

        # Step:3 evaluate the derivative 
        # ux = -cvx*np.einsum("ml,em->el", Sx, ucoeff)
        deriv_Op(d_ucoeff_in, d_ux)
        mulbyadv_Op(d_ux)
        #print(gpuarray.sum(d_ux))

        # Compute the continuous flux for each element in strong form
        totalFlux_Op(d_ux, d_jL, d_jR)
        
        # multiply by the inverse jacobian
        mulbyinvjac_Op(d_ux)

        # Step:4 Add collision kernel contribution
        #ux += Q(\sum U^{m}_{ar} ucoeff_{aej}, \sum V^{m}_{ra} ucoeff_{aej})
        for m in range(K):
            trans_U_Op[m](d_ucoeff_in, d_f)
            trans_V_Op[m](d_ucoeff_in, d_g)
            for r, e in it.product(sigModes[m], range(Ne)):
               sm.fs(d_f, d_g, d_ux, e, r, m)
        
        #for r, e in it.product(range(K), range(Ne)):
        #    sm.fs(d_ucoeff_in, d_ucoeff_in, d_ux, e, r, r)

        # Step:5 Multiply by inverse mass matrix
        invMass_Op(d_ux, d_ucoeff_out)
        
        # Reconstruct the solution (Useful for highly non-linear probs)
        #bwdTrans_Op(d_ucoeff_out, d_usol)
        #fwdTrans_Op(d_usol, d_ucoeff_out)

    # define a time-integrator
    odestype = cfg.lookup('time-integrator', 'scheme')
    odescls = subclass_where(DGFSIntegratorStd, intg_kind=odestype)
    odes = odescls(rhs, (K, Ne, Nv), cfg.dtype)

    # Finally start everything
    time = ti  # initialize time in case of restart
    nacptsteps = 0 # number of elasped steps in the current run

    # start timer
    start = MPI.Wtime()  # timer()

    while(time < tf):

        # March in time 
        odes.integrate(time, dt, d_ucoeff)

        # increment time
        time += dt 
        nacptsteps += 1

        # Final step: post processing routines
        residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
        moments(time, d_ucoeff)
        distribution(time, d_ucoeff)

        # copy the solution for the next time step
        cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # print elasped time
    end = MPI.Wtime()  # timer()
    if rank==root:
        print("Nsteps", nacptsteps, ", elasped time", end - start, "s")


def __main__():

    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork
        enable_prefork()

    # Define MPI communication world
    from mpi4py import MPI
    MPI.Init()

    # define the local rank based cuda device
    #os.environ['CUDA_DEVICE'] = str(get_local_rank())

    # CUDA device number (used by pycuda.autoinit)
    #from pycuda.autoinit import context

    # define the local rank based cuda device
    print("Local rank", get_local_rank())
    os.environ.pop('CUDA_DEVICE', None)
    devid = get_local_rank()
    os.environ['CUDA_DEVICE'] = str(devid)

    # CUDA device number (used by pycuda.autoinit)
    #from pycuda.autoinit import context
    #import pycuda.autoinit
    cuda.init()
    cudadevice = cuda.Device(devid)
    cudacontext = cudadevice.make_context()

    import atexit
    atexit.register(cudacontext.pop)

    # define the main process
    main()

    # finalize everything
    MPI.Finalize()
# -*- coding: utf-8 -*-

""" 
Asymptotic DGFS in one dimension with entropy based stabilization
"""

import os
import numpy as np
import itertools as it
from timeit import default_timer as timer

import mpi4py.rc
mpi4py.rc.initialize = False

from dgfs1D.mesh import Mesh
from dgfs1D.initialize import initialize
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup, ndrange)
from dgfs1D.nputil import get_comm_rank_root, get_local_rank, get_mpi
from dgfs1D.basis import Basis
from dgfs1D.std.velocitymesh import DGFSVelocityMeshStd
from dgfs1D.astd.scattering import DGFSScatteringModelAstd
from dgfs1D.std.scattering import DGFSScatteringModelStd
from dgfs1D.std.initcond import DGFSInitConditionStd
from dgfs1D.std.moments import DGFSMomWriterStd
from dgfs1D.std.residual import DGFSResidualStd
from dgfs1D.std.distribution import DGFSDistributionStd
from dgfs1D.std.bc import DGFSBCStd
from dgfs1D.axnpby import get_axnpby_kerns, copy
from dgfs1D.util import get_kernel, filter_tol, check
from dgfs1D.astd.integrator import DGFSIntegratorAstd
from dgfs1D.cublas import CUDACUBLASKernels

from pycuda import compiler, gpuarray
import pycuda.driver as cuda
from gimmik import generate_mm
np.set_printoptions(threshold=1000, linewidth=1000)

def main():
    # who am I in this world? (Bulleh Shah, 18th century sufi poet)
    comm, rank, root = get_comm_rank_root()

    # read the inputs (from people)
    cfg, args = initialize()
    mesh = Mesh(cfg)

    # define 1D mesh (construct a 1D world view)
    xmesh = mesh.xmesh
    xhi, xlo = min(xmesh), max(xmesh)

    # number of elements (how refined perspectives do we want/have?)
    Ne = mesh.Ne

    # define the basis (what is the basis for those perspectives?)
    bsKind = cfg.lookup('basis', 'kind')
    #assert bsKind == 'nodal-sem-gll', "Only one supported as of now"
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

    # load the scattering model
    smn = cfg.lookup('penalized-scattering-model', 'type')
    pscatteringcls = subclass_where(DGFSScatteringModelAstd, 
        scattering_model=smn)
    psm = pscatteringcls(cfg, vm, Ne=Ne, eps=sm.prefac)

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
    fwdTrans_Op, bwdTrans_Op, bwdTransFace_Op, deriv_Op, invMass_Op, \
        computeCellAvg_Op, extractDrLin_Op = map(
        matOpGen, (basis.fwdTransOp, basis.bwdTransOp, 
            basis.bwdTransFaceOp, basis.derivOp, basis.invMassOp, 
            basis.computeCellAvgKern, basis.extractDrLinKern)
    )

    # U, V operator kernels
    trans_U_Op = tuple(map(matOpGen, basis.uTransOps))
    trans_V_Op = tuple(map(matOpGen, basis.vTransOps))

    # prepare the kernel for extracting face/interface values
    dfltargs = dict(
        K=K, Ne=Ne, Nq=Nq, vsize=Nv, dtype=cfg.dtypename,
        mapL=mapL, mapR=mapR, offsetL=0, offsetR=len(mapR)-1,
        invjac=invjac, gRD=basis.gRD, gLD=basis.gLD, xsol=xsol)
    kernsrc = DottedTemplateLookup('dgfs1D.std.kernels', 
                                    dfltargs).get_template('std').render()
    kernmod = compiler.SourceModule(kernsrc)

    dfltargs.update(nalph=psm.nalph, Dr=basis.derivMat)    
    kernlimssrc = DottedTemplateLookup('dgfs1D.astd.kernels', 
                                    dfltargs).get_template('limiters').render()
    kernlimsmod = compiler.SourceModule(kernlimssrc)

    ce = cfg.lookupordefault('time-integrator', 'ce', 0.1)
    cmax = cfg.lookupordefault('time-integrator', 'cmax', 0.1)
    #print(ce, cmax); exit(0)
    dfltargs.update(L=vm.L(), hk=(z[1]-z[0])*jac[0,0], ce=ce, cmax=cmax, 
        cw=vm.cw(), H0=vm.H0())    
    kernentssrc = DottedTemplateLookup('dgfs1D.astd.kernels', 
                                    dfltargs).get_template('entropy').render()
    kernentsmod = compiler.SourceModule(kernentssrc)

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

    #if bcl_type == 'dgfs-cyclic' or bcr_type == 'dgfs-cyclic':
    #    assert(bcl_type==bcr_type);

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

    # total flux kernel (sums up surface and volume terms)
    totalFlux = get_kernel(kernmod, "totalFlux", 'PPPP')
    totalFlux_Op = lambda d_ux, d_jL, d_jR: totalFlux.prepared_call(
            grid_Nv, block, d_ux.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # linear limiter
    limitLin = get_kernel(kernlimsmod, "limitLin", 'PPPP')
    limitLin_Op = lambda d_u, d_ulx, d_uavg, d_ulim: \
        limitLin.prepared_call(grid_Nv, block, d_u.ptr, d_ulx.ptr, 
            d_uavg.ptr, d_ulim.ptr)


    # multiply the derivative by the advection velocity
    entropyPair = get_kernel(kernentsmod, "entropyPair", 'PPPP')
    entropyPair_Op = lambda d_f, d_E, d_G: entropyPair.prepared_call(
                    grid_KNeNv, block, vm.d_cvx().ptr, d_f.ptr, d_E.ptr, d_G.ptr)


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
        ic.apply_init_vals(usol, Nq, Ne, xsol, mesh=mesh, basis=basis, 
            psm=psm)

        # transfer the information to the gpu
        d_usol.set(usol.ravel())

        # forward transform to coefficient space
        fwdTrans_Op(d_usol, d_ucoeff)

    # check if we are restarting
    if hasattr(args, 'process_restart'):
        import h5py as h5py
        check(len(args.dist[0])==comm.size, "No. of distributions != nranks")
        with h5py.File(args.dist[0][rank].name, 'r') as h5f:
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


    d_oneV = gpuarray.to_gpu(np.ones(Nv, dtype=cfg.dtype))
    blas = CUDACUBLASKernels() # blas kernels for computing moments

    def innerProdV_Op(d_v1, d_v2):
        nvars = 1

        lda = d_v1.shape[0]//vm.vsize()
        assert lda==d_v2.shape[0]//nvars, "Some issue"

        sA_mom = (lda, vm.vsize())
        sB_mom = (nvars, vm.vsize())
        sC_mom = (lda, nvars)
        blas.mul(d_v1, sA_mom, d_oneV, sB_mom, d_v2, sC_mom)


    grid_Ne = get_grid_for_block(block, Ne)

    # operator generator for matrix operations
    matOpGen1 = lambda v: lambda arg0, arg1: v.prepared_call(
                grid_Ne, block, Ne, arg0.ptr, Ne, arg1.ptr, Ne)
    
    # forward trans, backward, backward (at faces), derivative kernels
    fwdTrans1_Op, bwdTrans1_Op, bwdTransFace1_Op, deriv1_Op, invMass1_Op, \
       computeCellAvg1_Op = map(
        matOpGen1, (basis.fwdTransOp, basis.bwdTransOp, basis.bwdTransFaceOp, \
            basis.derivOp, basis.invMassOp, basis.computeCellAvgKern)
    )

    NqfNe = Nqf*Ne
    KNe = K*Ne

    grid_1 = (1, 1)

    # prepare operators for execution (see std.mako for description)
    (extLeft1_Op, extRight1_Op, transferBC1_L_Op, transferBC1_R_Op, 
        insertBC1_L_Op, insertBC1_R_Op) = map(lambda v: 
        lambda *args: get_kernel(kernentsmod, v, 'PP').prepared_call(
            grid_1, block, *list(map(lambda c: c.ptr, args))
        ), ("extract_left", "extract_right", "transfer_bc_left", 
            "transfer_bc_right", "insert_bc_left", "insert_bc_right")
    )

    # flux kernel
    flux1 = get_kernel(kernentsmod, "flux1", 'PPPPPP')
    flux1_Op = lambda d_eL, d_eR, d_gL, d_gR, d_jeL, d_jeR: flux1.prepared_call(
            grid_1, block, 
            d_eL.ptr, d_eR.ptr, d_gL.ptr, d_gR.ptr, d_jeL.ptr, d_jeR.ptr)

    # total flux kernel
    totalFlux1 = get_kernel(kernentsmod, "totalFlux", 'PPP')
    totalFlux1_Op = lambda d_ex, d_jeL, d_jeR: totalFlux1.prepared_call(
            grid_1, block, d_ex.ptr, d_jeL.ptr, d_jeR.ptr)


    # multiply the coefficient by the inverse jacobian
    mulbyinvjacE = get_kernel(kernentsmod, "mul_by_invjac", 'P')
    mulbyinvjacE_Op = lambda v: mulbyinvjacE.prepared_call(
                    grid_1, block, v.ptr)

    # multiply the coefficient by the inverse jacobian
    entropyViscosity = get_kernel(kernentsmod, "entropyViscosity", 'PPPPPP')
    entropyViscosity_Op = lambda d_e, d_eavg, d_eL, d_eR, d_R, d_eps: entropyViscosity.prepared_call(
                    grid_1, block, d_e.ptr, d_eavg.ptr, d_eL.ptr, d_eR.ptr, d_R.ptr, d_eps.ptr)

    # \alpha AX + \beta Y kernel (for operations on coefficients)
    axnpbyCoeff1 = get_axnpby_kerns(2, range(K), Ne, cfg.dtype)
    axnpbyCoeff1_Op = lambda a0, x0, a1, x1: axnpbyCoeff1.prepared_call(
                    grid_Ne, block, x0.ptr, x1.ptr, a0, a1)

    grid_Nf = get_grid_for_block(block, Nf)
    axnpbyCoeffF = get_axnpby_kerns(2, range(1), Nf, cfg.dtype)
    axnpbyCoeffF_Op = lambda a0, x0, a1, x1: axnpbyCoeffF.prepared_call(
                    grid_Nf, block, x0.ptr, x1.ptr, a0, a1)

    axnpbyCoeff51 = get_axnpby_kerns(5, range(K), Ne, cfg.dtype)
    axnpbyCoeff51_Op = lambda a0, x0, a1, x1, a2, x2, a3, x3, a4, x4: axnpbyCoeff51.prepared_call(
                    grid_Ne, block, x0.ptr, x1.ptr, x2.ptr, x3.ptr, x4.ptr, a0, a1, a2, a3, a4)

    axnpbyCoeff41 = get_axnpby_kerns(4, range(K), Ne, cfg.dtype)
    axnpbyCoeff41_Op = lambda a0, x0, a1, x1, a2, x2, a3, x3: axnpbyCoeff41.prepared_call(
                    grid_Ne, block, x0.ptr, x1.ptr, x2.ptr, x3.ptr, a0, a1, a2, a3)
    
    def pex(*args): print(*args); exit(0) 

    d_eF = gpuarray.empty(NqfNe, dtype=cfg.dtype)
    d_eL = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_eR = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_gF = gpuarray.empty(NqfNe, dtype=cfg.dtype)
    d_gL = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_gR = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_jeL = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_jeR = gpuarray.empty(Nf, dtype=cfg.dtype) 
    d_bceL = gpuarray.empty(1, dtype=cfg.dtype) 
    d_bceR = gpuarray.empty(1, dtype=cfg.dtype) 
    d_bceT = gpuarray.empty(1, dtype=cfg.dtype) 
    d_ex = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_e0 = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_e = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_g = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_rhse = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_rhse0 = gpuarray.zeros(KNe, dtype=cfg.dtype)
    d_eavg = gpuarray.empty(Ne, dtype=cfg.dtype)
    d_eps = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_epsNp = gpuarray.empty(KNe, dtype=cfg.dtype)
    d_E = gpuarray.empty_like(d_ucoeff)
    d_G = gpuarray.empty_like(d_ucoeff)
    d_R = gpuarray.empty(KNe, dtype=cfg.dtype)
    #d_fL = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    #d_fR = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_fx = gpuarray.empty_like(d_ucoeff)
    d_qL = gpuarray.empty(NfNv, dtype=cfg.dtype) 
    d_qR = gpuarray.empty(NfNv, dtype=cfg.dtype) 
 

    # define the explicit part    
    def entropy(time, d_ucoeff_in, d_rhse, d_e):

        #pex(d_ucoeff_in.shape, d_E.shape, d_G.shape, KNeNv)

        # compute the entropy pair
        entropyPair_Op(d_ucoeff_in, d_E, d_G);

        #pex(0)
        # Inner product with the velocity space to form d_e
        innerProdV_Op(d_E, d_e)
        p = lambda v: v.get().reshape(K,-1).T
        #pex(p(d_e));

        #print(p(d_e), p(d_e0))
        #if time>8e-3: exit(0)

        # Inner product with the velocity space to form d_e
        innerProdV_Op(d_G, d_g)
        #pex(p(d_g));

        # reconstruct solution at faces
        bwdTransFace1_Op(d_e, d_eF)
        bwdTransFace1_Op(d_g, d_gF)
        #pex(d_eF, d_gF)

        # Define the flux 
        def fluxE(d_varF, d_varL, d_varR):

            # Step:1 extract the solution at faces
            extLeft1_Op(d_varF, d_varL)
            extRight1_Op(d_varF, d_varR)

            # transfer left boundary information in send buffer
            transferBC1_L_Op(d_varL, d_bceL)       # Transfer the left ghost BC info
            transferBC1_R_Op(d_varR, d_bceR)       # Transfer the right ghost BC info

            # send information
            req1 = comm.isend(d_bceR, dest=up_nbr)  # to upstream neighbour
            req2 = comm.isend(d_bceL, dest=down_nbr)  # to downstream neighbour 

            # recieve information
            h_bceL = comm.recv(source=down_nbr)  # from downstream neighbour
            h_bceR = comm.recv(source=up_nbr)    # from upstream neighbour
            MPI.Request.Waitall([req1, req2])

            # set information at left, right boundary
            if h_bceL: d_bceL.set(h_bceL) 
            else: transferBC1_L_Op(d_varL, d_bceL)
                
            if h_bceR: d_bceR.set(h_bceR) 
            else: transferBC1_R_Op(d_varR, d_bceR)

            # The physical-periodic boundary condition
            if comm.size==1: #and bcr_type=='dgfs-cyclic':
                copy(d_bceT, d_bceL); copy(d_bceL, d_bceR); copy(d_bceR, d_bceT);
            else:
                # At left, receive from right-most communicator; and vice-versa
                req1 = req2 = MPI.REQUEST_NULL
                if bcl_type=='dgfs-cyclic': req1=comm.isend(d_bceL, dest=comm.size-1)
                if bcr_type=='dgfs-cyclic': req2=comm.isend(d_bceR, dest=0)
                if bcr_type=='dgfs-cyclic': h_bceR=comm.recv(source=0)
                if bcl_type=='dgfs-cyclic': h_bceL=comm.recv(source=comm.size-1)
                MPI.Request.Waitall([req1, req2])            
                if bcl_type == 'dgfs-cyclic': d_bceL.set(h_bceL)
                elif bcr_type == 'dgfs-cyclic': d_bceR.set(h_bceR)
                
            # At left boundary        
            #transferBC_L_Op(d_varL, d_bceL)       # Transfer the ghost BC info
            #updateBC_L_Op(d_bceL, time)         # now update boundary info 
            #applyBC_L_Op(d_bceL, d_bceT, time)   # apply boundary condition
            d_bceT.set(d_bceL)
            #insertBC1_L_Op(d_bceT, d_varL)         # insert info to global face-flux

            # At right boundary        
            #transferBC_R_Op(d_varR, d_bceL)       # Transfer the ghost BC info
            #updateBC_R_Op(d_bceR, time)         # now update boundary info 
            #applyBC_R_Op(d_bceR, d_bceT, time)   # apply boundary condition
            d_bceT.set(d_bceR)
            #insertBC1_R_Op(d_bceT, d_varR)         # insert info to global face-flux


        #pex(d_eL, d_eR, d_gL, d_gR)
        fluxE(d_eF, d_eL, d_eR)
        fluxE(d_gF, d_gL, d_gR)
        #pex(d_eL, d_eR, d_gL, d_gR)

        #d_gL.set(np.ones(d_gL.shape))
        #d_gR.set(np.ones(d_gR.shape))

        # Step:2 Compute the flux and jumps (all operations in single call)
        flux1_Op(d_eL, d_eR, d_gL, d_gR, d_jeL, d_jeR)
        #pex(d_jeL, d_jeR)
        #pex(mapL, mapR, d_eL, d_eR, d_gL, d_gR)

        #print(p(d_e), p(d_e0))
        #if time>8e-3: exit(0)

        # Step:3 evaluate the derivative 
        # ux = -cvx*np.einsum("ml,em->el", Sx, ucoeff)
        deriv1_Op(d_g, d_ex)
        #if time>8e-3: pex(p(d_ex))

        #print(p(d_e), p(d_e0))
        #if time>8e-3: exit(0)

        # Compute the continuous flux for each element in strong form
        #totalFlux1_Op(d_ex, d_jeL, d_jeR)

        #print(p(d_e), p(d_e0))
        #if time>8e-3: exit(0)

        # multiply by the inverse jacobian
        # Now we have f* = d_ux
        mulbyinvjacE_Op(d_ex)
        #print(time)
        #print(p(d_ex))
        #print(p(d_e), p(d_e0))
        #if time>8e-3: exit(0)

        # project back to coefficient space
        #copy(d_rhse, d_ex)
        invMass1_Op(d_ex, d_rhse) 


    # total flux kernel
    ptr = lambda vs: [v.ptr if (hasattr(v, 'ptr')) else v for v in vs]
    #insertBC2_L = get_kernel(kernentsmod, "insertBC2_L", 'PPP')
    #insertBC2_L_Op = lambda *args: insertBC2_L.prepared_call(grid_Nv, block, ptr(*args))
    #insertBC2_R = get_kernel(kernentsmod, "insertBC2_R", 'PPP')
    #insertBC2_R_Op = lambda *args: insertBC2_R.prepared_call(grid_Nv, block, ptr(*args))
    constructGrad = get_kernel(kernentsmod, "constructGrad", 'PP')
    constructGrad_Op = lambda v1, v2: constructGrad.prepared_call(grid_Nv, block, v1.ptr, v2.ptr)
    fluxU = get_kernel(kernentsmod, "fluxU", 'PPPPPPP')
    fluxU_Op = lambda v1, v2, v3, v4, v5, v6: fluxU.prepared_call(grid_Nv, block, *ptr([vm.d_cvx(),v1,v2,v3,v4,v5,v6]))
    liftViscosity = get_kernel(kernentsmod, "liftViscosity", 'PP')
    liftViscosity_Op = lambda v1, v2: liftViscosity.prepared_call(grid_Ne, block, *ptr([v1,v2]))
    fluxQ = get_kernel(kernentsmod, "fluxQ", 'PPPPPPPPP')
    fluxQ_Op = lambda v1,v2,v3,v4,v5,v6,v7,v8: fluxQ.prepared_call(grid_Nv, block, vm.d_cvx().ptr, \
                                                                   *ptr([v1,v2,v3,v4,v5,v6,v7,v8]))

    # define the explicit part    
    def explicit(time, d_ucoeff_in, d_ucoeff_out):

        #axnpbyCoeff1_Op(0, d_eps, 0, d_eps)
        #print(d_eps); 
        #if time>8e-3: exit(0);
      
        def pex(*args): print(*args); exit(0)
        def p(*args): 
            if time>8e-3: print(*args); exit(0)
        p2 = lambda v: v.get().reshape(K,-1).T
 
        # reconstruct solution at faces
        bwdTransFace_Op(d_ucoeff_in, d_usolF)

        # Step:1 extract the solution at faces
        extLeft_Op(d_usolF, d_uL)
        extRight_Op(d_usolF, d_uR)

        # transfer left boundary information in send buffer
        transferBC_L_Op(d_uL, d_bcL)       # Transfer the left ghost BC info
        transferBC_R_Op(d_uR, d_bcR)       # Transfer the right ghost BC info

        # send information
        req1 = comm.isend(d_bcR, dest=up_nbr)  # to upstream neighbour
        req2 = comm.isend(d_bcL, dest=down_nbr)  # to downstream neighbour 

        # recieve information
        h_bcL = comm.recv(source=down_nbr)  # from downstream neighbour
        h_bcR = comm.recv(source=up_nbr)    # from upstream neighbour
        MPI.Request.Waitall([req1, req2])

        # set information at left, right boundary
        if h_bcL: d_bcL.set(h_bcL) 
        else: transferBC_L_Op(d_uL, d_bcL)
            
        if h_bcR: d_bcR.set(h_bcR) 
        else: transferBC_R_Op(d_uR, d_bcR)

        # The physical-periodic boundary condition
        if comm.size==1 and bcr_type=='dgfs-cyclic':
            copy(d_bcT, d_bcL); copy(d_bcL, d_bcR); copy(d_bcR, d_bcT);
        else:
            # At left, receive from right-most communicator; and vice-versa
            req1 = req2 = MPI.REQUEST_NULL
            if bcl_type=='dgfs-cyclic': req1=comm.isend(d_bcL, dest=comm.size-1)
            if bcr_type=='dgfs-cyclic': req2=comm.isend(d_bcR, dest=0)
            if bcr_type=='dgfs-cyclic': h_bcR=comm.recv(source=0)
            if bcl_type=='dgfs-cyclic': h_bcL=comm.recv(source=comm.size-1)
            MPI.Request.Waitall([req1, req2])            
            if bcl_type == 'dgfs-cyclic': d_bcL.set(h_bcL)
            elif bcr_type == 'dgfs-cyclic': d_bcR.set(h_bcR)
            
        # copy the flux data
        #copy(d_fL, d_uL); #copy(d_fR, d_uR)

        # At left boundary        
        #transferBC_L_Op(d_uL, d_bcL)      # Transfer the ghost BC info
        updateBC_L_Op(d_bcL, time)         # now update boundary info 
        applyBC_L_Op(d_bcL, d_bcT, time)   # apply boundary condition
        #insertBC2_L_Op(d_bcT, d_uL, d_fL) # insert info to global face-flux
        insertBC_L_Op(d_bcT, d_uL)         # insert info to global face-flux
        #insertBC_L_Op(d_bcT, d_fL)         # insert info to global face-flux
 
        # At right boundary        
        #transferBC_R_Op(d_uR, d_bcL)      # Transfer the ghost BC info
        updateBC_R_Op(d_bcR, time)         # now update boundary info 
        applyBC_R_Op(d_bcR, d_bcT, time)   # apply boundary condition
        #insertBC2_R_Op(d_bcT, d_uR, d_fR) # insert info to global face-flux
        insertBC_R_Op(d_bcT, d_uR)         # insert info to global face-flux
        #insertBC_R_Op(d_bcT, d_fR)         # insert info to global face-flux
 
        # prepare q
        deriv_Op(d_ucoeff_in, d_fx)
        constructGrad_Op(d_fx, d_eps);
        #liftViscosity_Op(d_eps, d_epsNp)
        #p(d_eps.get())

        # reconstruct solution at faces
        bwdTransFace1_Op(d_eps, d_eF)
        extLeft1_Op(d_eF, d_eL)
        extRight1_Op(d_eF, d_eR)
        axnpbyCoeffF_Op(0.5, d_eL, 0.5, d_eR)
        copy(d_eR, d_eL)
 
        fluxU_Op(d_uL, d_uR, d_eL, d_eR, d_jL, d_jR)
        totalFlux_Op(d_fx, d_jL, d_jR)
        #totalFluxQ_Op(d_fx, d_eps, d_fL, d_fR)
        mulbyinvjac_Op(d_fx)
        invMass_Op(d_fx, d_ux)
        copy(d_fx, d_ux)
 
        # reconstruct solution at faces
        bwdTransFace_Op(d_fx, d_usolF)

        # Step:1 extract the solution at faces
        extLeft_Op(d_usolF, d_qL)
        extRight_Op(d_usolF, d_qR)

        # transfer left boundary information in send buffer
        transferBC_L_Op(d_qL, d_bcL)       # Transfer the left ghost BC info
        transferBC_R_Op(d_qR, d_bcR)       # Transfer the right ghost BC info

        # send information
        req1 = comm.isend(d_bcR, dest=up_nbr)  # to upstream neighbour
        req2 = comm.isend(d_bcL, dest=down_nbr)  # to downstream neighbour 

        # recieve information
        h_bcL = comm.recv(source=down_nbr)  # from downstream neighbour
        h_bcR = comm.recv(source=up_nbr)    # from upstream neighbour
        MPI.Request.Waitall([req1, req2])

        # set information at left, right boundary
        if h_bcL: d_bcL.set(h_bcL) 
        else: transferBC_L_Op(d_uL, d_bcL)
            
        if h_bcR: d_bcR.set(h_bcR) 
        else: transferBC_R_Op(d_uR, d_bcR)

        # The physical-periodic boundary condition
        if comm.size==1: #and bcr_type=='dgfs-cyclic':
            copy(d_bcT, d_bcL); copy(d_bcL, d_bcR); copy(d_bcR, d_bcT);
        else:
            # At left, receive from right-most communicator; and vice-versa
            req1 = req2 = MPI.REQUEST_NULL
            if bcl_type=='dgfs-cyclic': req1=comm.isend(d_bcL, dest=comm.size-1)
            if bcr_type=='dgfs-cyclic': req2=comm.isend(d_bcR, dest=0)
            if bcr_type=='dgfs-cyclic': h_bcR=comm.recv(source=0)
            if bcl_type=='dgfs-cyclic': h_bcL=comm.recv(source=comm.size-1)
            MPI.Request.Waitall([req1, req2])            
            if bcl_type == 'dgfs-cyclic': d_bcL.set(h_bcL)
            elif bcr_type == 'dgfs-cyclic': d_bcR.set(h_bcR)
            
        # At left boundary        
        insertBC_L_Op(d_bcL, d_qL)         # insert info to global face-flux

        # At right boundary        
        insertBC_R_Op(d_bcR, d_qR)         # insert info to global face-flux

        # construct gradient for u
        constructGrad_Op(d_fx, d_eps)
        deriv_Op(d_ucoeff_in, d_ux)
        mulbyadv_Op(d_ux)
        axnpbyCoeff_Op(1, d_fx, 1., d_ux) 

        fluxQ_Op(d_uL, d_uR, d_eL, d_eR, d_qL, d_qR, d_jL, d_jR)
        totalFlux_Op(d_fx, d_jL, d_jR)
        mulbyinvjac_Op(d_fx)
        invMass_Op(d_fx, d_ucoeff_out)
        #copy(d_ucoeff_out, d_fx)


    NeA = Ne*psm.nalph
    KNeA = K*Ne*psm.nalph
    grid_KNeA = get_grid_for_block(block, KNeA)
    grid_NeA = get_grid_for_block(block, NeA)
    grid_A = get_grid_for_block(block, psm.nalph)

    # \alpha AX + \beta Y kernel (for operations on coefficients)
    axnpbyMomsCoeff = get_axnpby_kerns(2, range(K), NeA, cfg.dtype)
    axnpbyMomsCoeff_Op = lambda a0, x0, a1, x1: axnpbyMomsCoeff.prepared_call(
                    grid_NeA, block, x0.ptr, x1.ptr, a0, a1)


    # define the explicit collision operator  
    def explicitQ(time, d_ucoeff_in, d_ucoeff_out, nu=None):

        axnpbyCoeff_Op(0., d_ucoeff_out, 0., d_ucoeff_out)
        if nu: axnpbyMomsCoeff_Op(0., nu, 0., nu)

        #for m in range(K):
        #    trans_U_Op[m](d_ucoeff_in, d_f)
        #    trans_V_Op[m](d_ucoeff_in, d_g)
        #    for r, e in it.product(sigModes[m], range(Ne)):
        #       sm.fs(d_f, d_g, d_ucoeff_out, e, r, m, d_nu=nu) 

        for r, e in it.product(range(K), range(Ne)):
            sm.fs(d_ucoeff_in, d_ucoeff_in, d_ucoeff_out, e, r, r, d_nu=nu)

        if nu: nu_max = gpuarray.max(nu); nu.set(np.ones(nu.shape)*nu_max.get());

        # compute the entropy pair
        #entropyPair_Op(d_ucoeff_out, d_E, d_G);
        #innerProdV_Op(d_E, d_e)


    d_uavg, d_ulx = map(gpuarray.empty_like, [d_ucoeff]*2)
    def limit(d_ucoeff_in, d_ucoeff_out):
        assert comm.size == 1, "Not implemented"
        #assert basis.basis_kind == 'nodal-sem-gll', "Not implemented"

        # Extract the cell average
        computeCellAvg_Op(d_ucoeff_in, d_uavg)

        # extract gradient of the linear polynomial
        extractDrLin_Op(d_ucoeff_in, d_ulx)
        mulbyinvjac_Op(d_ulx)

        # limit functions in all cells
        limitLin_Op(d_ucoeff_in, d_ulx, d_uavg, d_ucoeff_out)


    # operator generator for matrix operations
    matOpGen1 = lambda v: lambda arg0, arg1: v.prepared_call(
                grid_NeA, block, NeA, arg0.ptr, NeA, arg1.ptr, NeA)
    
    # forward trans, backward, backward (at faces), derivative kernels
    computeCellAvgA_Op, extractDrLin1_Op = map(
        matOpGen1, (basis.computeCellAvgKern, basis.extractDrLinKern)
    )

    # multiply the coefficient by the inverse jacobian
    mulbyinvjacA = get_kernel(kernlimsmod, "mul_by_invjac1", 'P')
    mulbyinvjacA_Op = lambda d_Ux: mulbyinvjacA.prepared_call(
                    grid_A, block, d_Ux.ptr)

    # linear limiter
    limitLinA = get_kernel(kernlimsmod, "limitLin1", 'PPPP')
    limitLinA_Op = lambda d_U, d_Ulx, d_Uavg, d_Ulim: \
        limitLinA.prepared_call(grid_A, block, d_U.ptr, d_Ulx.ptr, 
            d_Uavg.ptr, d_Ulim.ptr)


    d_Ucoeff = gpuarray.empty(KNeA, dtype=cfg.dtype)
    d_Uavg, d_Ulx = map(gpuarray.empty_like, [d_Ucoeff]*2)
    def limit1(d_Ucoeff_in, d_Ucoeff_out):
        assert comm.size == 1, "Not implemented"
        #assert basis.basis_kind == 'nodal-sem-gll', "Not implemented"

        # Extract the cell average
        computeCellAvgA_Op(d_Ucoeff_in, d_Uavg)

        # extract gradient of the linear polynomial
        extractDrLinA_Op(d_Ucoeff_in, d_Ulx)
        mulbyinvjacA_Op(d_Ulx)

        # limit fUnctions in all cells
        limitLinA_Op(d_Ucoeff_in, d_Ulx, d_Uavg, d_Ucoeff_out)


    # define a time-integrator (we use Euler scheme: good enough for steady)
    odestype = cfg.lookup('time-integrator', 'scheme')
    odescls = subclass_where(DGFSIntegratorAstd, intg_kind=odestype)
    odes = odescls(explicit, psm, (K, Ne, Nv), cfg.dtype, explicitQ=explicitQ, limit1=limit1)
    limitOn = cfg.lookupordefault('time-integrator', 'limiter', 0)

    # Finally start everything
    time = ti  # initialize time in case of restart
    nacptsteps = 0 # number of elasped steps in the current run

    # start timer
    start = timer()
    while(time < tf):

        if limitOn: 
            #print(d_e.get(), d_e0.get()) #, d_rhse.get(), d_rhse0.get(), d_R.get())
            
            # evalaute entropy, rhse, jeR
            entropy(time, d_ucoeff, d_rhse, d_e)

            #print(d_R.get())
            #if time>8e-3: exit(0)

            #print("-----", d_e.get(), d_e0.get(), "-----") #, d_rhse.get(), d_rhse0.get(), d_R.get())
            #if time>8e-3: exit(0); 

            # form residual: d_R = (e-e0)/dt - 0.5*rhse - 0.5*rhse0;
            axnpbyCoeff51_Op(0, d_R, 1./dt, d_e, -1./dt, d_e0, 0.5, d_rhse, 0.5, d_rhse0)
            #axnpbyCoeff41_Op(0, d_R, 1./dt, d_e, -1./dt, d_e0, 1, d_rhse)
            #copy(d_R, d_e)
            
            if nacptsteps==0: #or nacptsteps==1:
                axnpbyCoeff1_Op(0, d_R, 0, d_R)


            #if time> 1e-3: exit(0);

            #print(d_e.get(), ) #, d_rhse.get(), d_rhse0.get(), d_R.get())
            #if time>8e-3: exit(0); 

            # Extract the cell average
            computeCellAvg1_Op(d_e, d_eavg)
            #pex(d_e, d_eavg)
            #print(d_e.get(), d_e0.get()) #, d_rhse.get(), d_rhse0.get(), d_R.get())
            #if time>2e-3: exit(0); 

            # compute entropy viscosity
            entropyViscosity_Op(d_e, d_eavg, d_eL, d_eR, d_R, d_eps)
            #if nacptsteps%20==0 : print(d_eps.get())
            #if time>2e-3: exit(0)

            copy(d_e0, d_e); copy(d_rhse0, d_rhse)
            #print(d_e.get(), d_e0.get())
            #pex(d_eps)


        # March in time 
        odes.integrate(time, dt, nacptsteps, d_ucoeff)
        #if limitOn: limit(d_ucoeff, d_ucoeff)

        # increment time
        time += dt 
        nacptsteps += 1

        # Final step: post processing routines
        residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
        moments(dt, time, d_ucoeff)
        distribution(dt, time, d_ucoeff)

        # copy the solution for the next time step
        cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # print elasped time
    end = timer()
    elapsed = np.array([end - start])
    if rank != root: comm.Reduce(elapsed, None, op=get_mpi('sum'), root=root)
    else:
        comm.Reduce(get_mpi('in_place'), elapsed, op=get_mpi('sum'), root=root)
        avgtime = elapsed[0]/comm.size
        print("Nsteps", nacptsteps, ", elapsed time", avgtime, "s")


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

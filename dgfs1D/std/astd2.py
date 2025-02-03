# -*- coding: utf-8 -*-

""" 
Asymptotic DGFS in one dimension
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
from dgfs1D.cublas import CUDACUBLASKernels

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
    grid_Ne = get_grid_for_block(block, Ne)
    grid_Nv = get_grid_for_block(block, Nv)
    grid_NqNe = get_grid_for_block(block, Nq*Ne)
    grid_KNe = get_grid_for_block(block, K*Ne)
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

    # forward transform for collision frequency
    fwdTransOne_Op = lambda arg0, arg1: basis.fwdTransOp.prepared_call(
                grid_Ne, block, Ne, arg0.ptr, Ne, arg1.ptr, Ne)

    # backward transform for collision frequency
    bwdTransOne_Op = lambda arg0, arg1: basis.bwdTransOp.prepared_call(
                grid_Ne, block, Ne, arg0.ptr, Ne, arg1.ptr, Ne)

    # U, V operator kernels
    trans_U_Op = tuple(map(matOpGen, basis.uTransOps))
    trans_V_Op = tuple(map(matOpGen, basis.vTransOps))

    # 5 moments: \int f (1 v |v|^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
    nalph = 5 
    # 11 moments for ESBGK \int f (1 v |v|^2 vxv)
    nalphES = 11
    # \int f (1 v |v|^2 vxv v|v|^2)
    nalphSK = 14

    # Stages of ARS
    nars = [2, 3, 4, 5]

    # prepare the kernel for extracting face/interface values
    dfltargs = dict(
        K=K, Ne=Ne, Nq=Nq, vsize=Nv, dtype=cfg.dtypename,
        mapL=mapL, mapR=mapR, offsetL=0, offsetR=len(mapR)-1,
        invjac=invjac, gRD=basis.gRD, gLD=basis.gLD, nalph=nalph,
        nalphES=nalphES, Pr=sm.prefacESBGK/sm.prefacBGK,
        omega=sm.omega, prefac=sm.prefac, prefacBGK=sm.prefacBGK, 
        prefacESBGK=sm.prefacESBGK, prefacSK=sm.prefacSK, 
        nalphSK=nalphSK,
        nars=nars)
    kernsrc = DottedTemplateLookup('dgfs1D.std.kernels', 
                                    dfltargs).get_template('astd').render()
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

    if bcl_type == 'dgfs-cyclic' or bcr_type == 'dgfs-cyclic':
        assert(bcl_type==bcr_type);

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

    # construct maxwellian given the moments
    cmaxwellian = get_kernel(kernmod, "cmaxwellian", 'PPPPPP')
    cmaxwellian_Op = lambda d_M, d_moms, d_collFreq: cmaxwellian.prepared_call(
                    grid_Nv, block, 
                    vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr, 
                    d_M.ptr, d_moms.ptr, d_collFreq.ptr)

    # construct maxwellian given the moments
    cmaxwellianES = get_kernel(kernmod, "cmaxwellianES", 'PPPPPP')
    cmaxwellianES_Op = lambda d_M, d_momsES, d_collFreq: \
                        cmaxwellianES.prepared_call(
                            grid_Nv, block, 
                            vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr, 
                            d_M.ptr, d_momsES.ptr, d_collFreq.ptr)

    # construct maxwellian given the moments
    cmaxwellianSK = get_kernel(kernmod, "cmaxwellianSK", 'PPPPPP')
    cmaxwellianSK_Op = lambda d_M, d_momsSK, d_collFreq: \
                        cmaxwellianSK.prepared_call(
                            grid_Nv, block, 
                            vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr, 
                            d_M.ptr, d_momsSK.ptr, d_collFreq.ptr)

    # A utility for generating stage update kernels
    def updateDistKernsGen(kerns): 
        return map(
            lambda idxkern: \
                lambda *args: \
                idxkern[1].prepared_call(
                    grid_Nv, block, *tuple(
                        args[:(2*idxkern[0]-1)] +
                        tuple(map(lambda v: v.ptr, args[(2*idxkern[0]-1):]))
                    )
                ), 
            enumerate(kerns, start=nars[0])
        )

    # compute the moment of the bgk kernel
    updateBGKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_BGKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1+(v-1)*2)), nars
    )

    (updateDistribution2_BGKARS_Op, updateDistribution3_BGKARS_Op,
        updateDistribution4_BGKARS_Op, updateDistribution5_BGKARS_Op) \
        = updateDistKernsGen(updateBGKARSKerns)

    # compute the moment of the esbgk kernel
    updateESBGKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_ESBGKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1+(v-1)*2)), nars
    )
    (updateDistribution2_ESBGKARS_Op, updateDistribution3_ESBGKARS_Op,
        updateDistribution4_ESBGKARS_Op, updateDistribution5_ESBGKARS_Op) \
        = updateDistKernsGen(updateESBGKARSKerns)

    # compute the moment of the shakov kernel
    updateSKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_SKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1+(v-1)*2)), nars
    )
    (updateDistribution2_SKARS_Op, updateDistribution3_SKARS_Op,
        updateDistribution4_SKARS_Op, updateDistribution5_SKARS_Op) \
        = updateDistKernsGen(updateSKARSKerns)

    # compute the moment of the fsbgk kernel
    updateFSBGKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_FSBGKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(3+5*(v-1))), nars
    )
    (updateDistribution2_FSBGKARS_Op, updateDistribution3_FSBGKARS_Op,
        updateDistribution4_FSBGKARS_Op, updateDistribution5_FSBGKARS_Op) \
        = updateDistKernsGen(updateFSBGKARSKerns)

    # compute the moment of the fsesbgk kernel
    updateFSESBGKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_FSESBGKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(3+5*(v-1))), nars
    )
    (updateDistribution2_FSESBGKARS_Op, updateDistribution3_FSESBGKARS_Op,
        updateDistribution4_FSESBGKARS_Op, updateDistribution5_FSESBGKARS_Op) \
        = updateDistKernsGen(updateFSESBGKARSKerns)

    # compute the moment of the fsesbgk kernel
    updateFSSKARSKerns = map(
        lambda v: get_kernel(kernmod, "updateDistribution{0}_FSSKARS".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(3+5*(v-1))), nars
    )
    (updateDistribution2_FSSKARS_Op, updateDistribution3_FSSKARS_Op,
        updateDistribution4_FSSKARS_Op, updateDistribution5_FSSKARS_Op) \
        = updateDistKernsGen(updateFSSKARSKerns)


    # \alpha AX + \beta Y kernel (for operations on coefficients)
    axnpbyCoeff = get_axnpby_kerns(2, range(K), NeNv, cfg.dtype)
    axnpbyCoeff_Op = lambda a0, x0, a1, x1: axnpbyCoeff.prepared_call(
                    grid_NeNv, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on physical solutions)
    axnpbySol = get_axnpby_kerns(2, range(Nq), NeNv, cfg.dtype)
    axnpbySol_Op = lambda a0, x0, a1, x1: axnpbySol.prepared_call(
                    grid_NeNv, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on physical solutions)
    axnpbySolOne = get_axnpby_kerns(2, range(Nq), Ne, cfg.dtype)
    axnpbySolOne_Op = lambda a0, x0, a1, x1: axnpbySolOne.prepared_call(
                    grid_Ne, block, x0.ptr, x1.ptr, a0, a1)

    # total flux kernel (sums up surface and volume terms)
    totalFlux = get_kernel(kernmod, "totalFlux", 'PPPP')
    totalFlux_Op = lambda d_ux, d_jL, d_jR: totalFlux.prepared_call(
            grid_Nv, block, d_ux.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # maxOp
    #totalFlux = get_kernel(kernmod, "totalFlux", 'PPPP')
    #totalFlux_Op = lambda d_ux, d_jL, d_jR: totalFlux.prepared_call(
    #        grid_Nv, block, d_ux.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # Kernel "specifically" for computing moments   
    # compute mBGK
    mBGK = np.vstack(
        (np.ones(Nv)*vm.cw(), 
            vm.cv()*vm.cw(), 
            np.einsum('ij,ij->j', vm.cv(), vm.cv())*vm.cw()
        )
    ) # 5 x Nv
    d_mBGK = gpuarray.to_gpu((mBGK).ravel()) # Nv x 5 flatenned

    # \alpha AX + \beta Y kernel (for operations on moments)
    axnpbySolMom = get_axnpby_kerns(2, range(nalph), Nq*Ne, cfg.dtype)
    axnpbySolMom_Op = lambda a0, x0, a1, x1: axnpbySolMom.prepared_call(
                    grid_NqNe, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on moments)
    axnpbyCoeffMom = get_axnpby_kerns(2, range(nalph), K*Ne, cfg.dtype)
    axnpbyCoeffMom_Op = lambda a0, x0, a1, x1: axnpbyCoeffMom.prepared_call(
                    grid_KNe, block, x0.ptr, x1.ptr, a0, a1)

    nalphNe = nalph*Ne
    grid_nalphNe = get_grid_for_block(block, nalphNe)

    # forward transform for moments
    fwdTransMom_Op = lambda arg0, arg1: basis.fwdTransOp.prepared_call(
                grid_nalphNe, block, nalphNe, arg0.ptr, 
                nalphNe, arg1.ptr, nalphNe)

    # backward transform for moments
    bwdTransMom_Op = lambda arg0, arg1: basis.bwdTransOp.prepared_call(
                grid_nalphNe, block, nalphNe, arg0.ptr, 
                nalphNe, arg1.ptr, nalphNe)

    # prepare the momentum computation kernel
    blas = CUDACUBLASKernels()
    sA_mom = (Nq*Ne, Nv)
    sB_mom = (nalph, Nv)
    sC_mom = (Nq*Ne, nalph)        
    moments_Op = lambda A, B, C: blas.mul(A, sA_mom, 
        B, sB_mom, C, sC_mom)
    d_moms = gpuarray.empty(Nq*Ne*nalph, dtype=cfg.dtype)

    # prepare the momentum computation kernel
    sA_momCoeff = (K*Ne, Nv)
    sB_momCoeff = (nalph, Nv)
    sC_momCoeff = (K*Ne, nalph)        
    momentsCoeff_Op = lambda A, B, C: blas.mul(A, sA_momCoeff, 
        B, sB_momCoeff, C, sC_momCoeff)
    d_momsCoeff = gpuarray.empty(K*Ne*nalph, dtype=cfg.dtype) 

    # For ESBGK systems
    # Kernel "specifically" for computing ESBGK moments   
    cv = vm.cv()
    mBGKES = np.vstack(
            (np.ones(Nv), # mass
            cv, # momentum
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:], # off-diag 
            np.einsum('ij,ij->j', vm.cv(), vm.cv())
        ))*vm.cw() # 11 x Nv
    d_mBGKES = gpuarray.to_gpu((mBGKES).ravel()) # Nv x 11 flat

    # \alpha AX + \beta Y kernel (for operations on ESBGK moments)
    axnpbySolMomES = get_axnpby_kerns(2, range(nalphES), Nq*Ne, cfg.dtype)
    axnpbySolMomES_Op = lambda a0, x0, a1, x1: axnpbySolMomES.prepared_call(
                    grid_NqNe, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on moments)
    axnpbyCoeffMomES = get_axnpby_kerns(2, range(nalphES), K*Ne, cfg.dtype)
    axnpbyCoeffMomES_Op = lambda a0, x0, a1, x1: axnpbyCoeffMomES.prepared_call(
                    grid_KNe, block, x0.ptr, x1.ptr, a0, a1)

    nalphESNe = nalphES*Ne
    grid_nalphESNe = get_grid_for_block(block, nalphESNe)

    # forward transform for moments
    fwdTransMomES_Op = lambda arg0, arg1: basis.fwdTransOp.prepared_call(
                grid_nalphESNe, block, nalphESNe, arg0.ptr, 
                nalphESNe, arg1.ptr, nalphESNe)

    # backward transform for moments
    bwdTransMomES_Op = lambda arg0, arg1: basis.bwdTransOp.prepared_call(
                grid_nalphESNe, block, nalphESNe, arg0.ptr, 
                nalphESNe, arg1.ptr, nalphESNe)

    # prepare the momentum computation kernel
    sA_momES = (Nq*Ne, Nv)
    sB_momES = (nalphES, Nv)
    sC_momES = (Nq*Ne, nalphES)        
    momentsES_Op = lambda A, B, C: blas.mul(A, sA_momES, 
        B, sB_momES, C, sC_momES)
    d_momsES = gpuarray.empty(Nq*Ne*nalphES, dtype=cfg.dtype)

    # prepare the momentum computation kernel
    sA_momESCoeff = (K*Ne, Nv)
    sB_momESCoeff = (nalphES, Nv)
    sC_momESCoeff = (K*Ne, nalphES)        
    momentsESCoeff_Op = lambda A, B, C: blas.mul(A, sA_momESCoeff, 
        B, sB_momESCoeff, C, sC_momESCoeff)
    d_momsESCoeff = gpuarray.empty(K*Ne*nalphES, dtype=cfg.dtype) 

    # moment normalization
    momentsESNorm = get_kernel(kernmod, "momentsESNorm", 'P')
    momentsESNorm_Op = lambda d_momsES: momentsESNorm.prepared_call(
            grid_NqNe, block, d_momsES.ptr)


    # For Shakov systems
    # Kernel "specifically" for computing Shakov moments   
    cv = vm.cv()
    mSK = np.vstack(
            (np.ones(Nv), # mass
            cv, # momentum
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:], # off-diag 
            np.einsum('ij,ij->j', vm.cv(), vm.cv()), # temperature
            np.einsum('ij,ij->j', cv, cv)*cv[0,:], # x-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[1,:], # y-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[2,:] # z-heat-flux
        ))*vm.cw() # 14 x Nv
    d_mSK = gpuarray.to_gpu((mSK).ravel()) # Nv x 14 flat

    # \alpha AX + \beta Y kernel (for operations on ESBGK moments)
    axnpbySolMomSK = get_axnpby_kerns(2, range(nalphSK), Nq*Ne, cfg.dtype)
    axnpbySolMomSK_Op = lambda a0, x0, a1, x1: axnpbySolMomSK.prepared_call(
                    grid_NqNe, block, x0.ptr, x1.ptr, a0, a1)

    # \alpha AX + \beta Y kernel (for operations on moments)
    axnpbyCoeffMomSK = get_axnpby_kerns(2, range(nalphSK), K*Ne, cfg.dtype)
    axnpbyCoeffMomSK_Op = lambda a0, x0, a1, x1: axnpbyCoeffMomSK.prepared_call(
                    grid_KNe, block, x0.ptr, x1.ptr, a0, a1)

    nalphSKNe = nalphSK*Ne
    grid_nalphSKNe = get_grid_for_block(block, nalphSKNe)

    # forward transform for moments
    fwdTransMomSK_Op = lambda arg0, arg1: basis.fwdTransOp.prepared_call(
                grid_nalphSKNe, block, nalphSKNe, arg0.ptr, 
                nalphSKNe, arg1.ptr, nalphSKNe)

    # backward transform for moments
    bwdTransMomSK_Op = lambda arg0, arg1: basis.bwdTransOp.prepared_call(
                grid_nalphSKNe, block, nalphSKNe, arg0.ptr, 
                nalphSKNe, arg1.ptr, nalphSKNe)

    # prepare the momentum computation kernel
    sA_momSK = (Nq*Ne, Nv)
    sB_momSK = (nalphSK, Nv)
    sC_momSK = (Nq*Ne, nalphSK)        
    momentsSK_Op = lambda A, B, C: blas.mul(A, sA_momSK, 
        B, sB_momSK, C, sC_momSK)
    d_momsSK = gpuarray.empty(Nq*Ne*nalphSK, dtype=cfg.dtype)

    # prepare the momentum computation kernel
    sA_momSKCoeff = (K*Ne, Nv)
    sB_momSKCoeff = (nalphSK, Nv)
    sC_momSKCoeff = (K*Ne, nalphSK)        
    momentsSKCoeff_Op = lambda A, B, C: blas.mul(A, sA_momSKCoeff, 
        B, sB_momSKCoeff, C, sC_momSKCoeff)
    d_momsSKCoeff = gpuarray.empty(K*Ne*nalphSK, dtype=cfg.dtype) 

    # moment normalization
    momentsSKNorm = get_kernel(kernmod, "momentsSKNorm", 'P')
    momentsSKNorm_Op = lambda d_momsSK: momentsSKNorm.prepared_call(
            grid_NqNe, block, d_momsSK.ptr)

    
    # A utility for generating moment update kernels
    def updateMomKernsGen(kerns):
        return map(
            lambda idxkern: \
                lambda *args: \
                idxkern[1].prepared_call(
                    grid_NqNe, block, *tuple(
                        args[:(2*idxkern[0]-1)] +
                        tuple(map(lambda v: v.ptr, args[(2*idxkern[0]-1):]))
                    )
                ), 
            enumerate(kerns, start=nars[0])
        )

    # compute the moment of the bgk kernel
    updateMomKerns = map(
        lambda v: get_kernel(kernmod, "update_ARS_Mom_stage{0}".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1)), nars
    )
    (update1Mom_Op, update2Mom_Op, update3Mom_Op, update4Mom_Op) \
        = updateMomKernsGen(updateMomKerns) 

    # compute the moment of the esbgk kernel
    updateMomESKerns = map(
        lambda v: get_kernel(kernmod, "update_ARS_MomES_stage{0}".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1)), nars
    )
    (update1MomES_Op, update2MomES_Op, update3MomES_Op, update4MomES_Op) \
        = updateMomKernsGen(updateMomESKerns)   

    # compute the moment of the shakov kernel
    updateMomSKKerns = map(
        lambda v: get_kernel(kernmod, "update_ARS_MomSK_stage{0}".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(2*v-1)), nars
    )
    (update1MomSK_Op, update2MomSK_Op, update3MomSK_Op, update4MomSK_Op) \
        = updateMomKernsGen(updateMomSKKerns)     

    # compute the moment of the esbgk kernel
    updateMomFSESKerns = map(
        lambda v: get_kernel(kernmod, "update_ARS_MomFSES_stage{0}".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(3*v-2)), nars
    )
    (update1MomFSES_Op, update2MomFSES_Op, update3MomFSES_Op, update4MomFSES_Op) \
        = updateMomKernsGen(updateMomFSESKerns)     

    # compute the moment of the esbgk kernel
    updateMomFSSKKerns = map(
        lambda v: get_kernel(kernmod, "update_ARS_MomFSSK_stage{0}".format(v), 
            cfg.dtypename[0]*(2*v-1)+'P'*(3*v-2)), nars
    )
    (update1MomFSSK_Op, update2MomFSSK_Op, update3MomFSSK_Op, update4MomFSSK_Op) \
        = updateMomKernsGen(updateMomFSSKKerns)     



    d_collFreq = gpuarray.empty(Nq*Ne*1, dtype=cfg.dtype) 
    d_collFreqCoeff = gpuarray.empty(K*Ne*1, dtype=cfg.dtype) 
    #d_collFreq = gpuarray.empty(NqNeNv, dtype=cfg.dtype) 
    #d_collFreqCoeff = gpuarray.empty(KNeNv, dtype=cfg.dtype) 
    
    # summation operation
    mOne = np.vstack(
        (np.ones(Nv)
        )
    ) # 1 x Nv
    d_mOne = gpuarray.to_gpu((mOne).ravel()) # Nv x 1 flatenned
    sA_one = (Nq*Ne, Nv)
    sB_one = (1, Nv)
    sC_one = (Nq*Ne, 1)        
    sumNv_Op = lambda A, B, C: blas.mul(A, sA_one, 
        B, sB_one, C, sC_one)
    d_one = gpuarray.empty(Nq*Ne*1, dtype=cfg.dtype)
    d_oneCoeff = gpuarray.empty(K*Ne*1, dtype=cfg.dtype)

    extractAt = get_kernel(kernmod, "extractAt", 'IIPP')
    extractAt_Op = lambda elem, mode, d_in, d_out: extractAt.prepared_call(
            grid_Nv, block, elem, mode, d_in.ptr, d_out.ptr)

    insertAtOne = get_kernel(kernmod, "insertAtOne", 'IIPP')
    insertAtOne_Op = lambda elem, mode, d_in, d_out: insertAtOne.prepared_call(
            (1, 1), (1, 1, 1), elem, mode, d_in.ptr, d_out.ptr)

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

    # for the first step define the maxwellian
    d_M0coeff = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_M0sol = gpuarray.empty(NqNeNv, dtype=cfg.dtype)
    d_M1coeff = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_M1sol = gpuarray.empty(NqNeNv, dtype=cfg.dtype)

    # extra storage
    d_Q = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_fS = gpuarray.empty(KNeNv, dtype=cfg.dtype)

    # construct initial maxwellian
    # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
    moments_Op(d_usol, d_mBGK, d_moms) # compute moments
    cmaxwellian_Op(d_M0sol, d_moms, d_collFreq) # construct maxwellian
    fwdTrans_Op(d_M0sol, d_M0coeff) # project to coefficient space

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
    def rhs_old(time, d_ucoeff_in):

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
        # Now we have f* = d_ux
        mulbyinvjac_Op(d_ux)

        invMass_Op(d_ux, d_fS) 


        # Next, we compute the moments of the distribution function        
        # evaluate f* = f^n - dt v \cdot \nabla_x f^n 
        #axnpbyCoeff_Op(0., d_fS, 1., d_ucoeff_in, -dt, d_ux)
        axnpbyCoeff_Op(dt, d_fS, 1., d_ucoeff_in)

        # Multiply by inverse mass matrix
        #invMass_Op(d_ucoeff_out, d_M1coeff) 
        axnpbyCoeff_Op(0., d_M1coeff, 1., d_fS)

        # reconstruct the solution
        bwdTrans_Op(d_M1coeff, d_M1sol)

        # Compute moments
        # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
        moments_Op(d_M1sol, d_mBGK, d_moms)

        # construct maxwellian
        cmaxwellian_Op(d_M1sol, d_moms, d_collFreq)

        # project collfreq to coeff space
        #fwdTransOne_Op(d_collFreq, d_collFreqCoeff)       
        fwdTrans_Op(d_collFreq, d_collFreqCoeff)

        # project back to coefficient space
        fwdTrans_Op(d_M1sol, d_M1coeff)



        # Step:4 Compute collision kernel contribution
        #Q += Q(\sum U^{m}_{ar} ucoeff_{aej}, \sum V^{m}_{ra} ucoeff_{aej})
        axnpbyCoeff_Op(0., d_Q, 0., d_Q)
        d_collFreqCoeff.fill(0.)
        for m in range(K):
            trans_U_Op[m](d_ucoeff_in, d_f)
            trans_V_Op[m](d_ucoeff_in, d_g)
            for r, e in it.product(sigModes[m], range(Ne)):
               sm.fs(d_f, d_g, d_Q, e, r, m, d_collFreqCoeff)
        
        # Step:5 Multiply by inverse mass matrix
        #invMass_Op(d_ux, d_ucoeff_out)
        
    
    # define the ode rhs    
    def L(time, d_ucoeff_in, d_ucoeff_out):

        # reconstruct solution at faces
        bwdTransFace_Op(d_ucoeff_in, d_usolF)

        # Step:1 extract the solution at faces
        extLeft_Op(d_usolF, d_uL)
        extRight_Op(d_usolF, d_uR)

        # transfer left boundary information in send buffer
        transferBC_L_Op(d_uL, d_bcL)       # Transfer the left ghost BC info
        transferBC_R_Op(d_uR, d_bcR)       # Transfer the right ghost BC info

        # this can be adjusted in case of RDMA enabled MPI support
        #h_bcL, h_bcR = d_bcL.get(), d_bcR.get()
        #h_bcL, h_bcR = map(lambda v: v.gpudata.as_buffer(v.nbytes), 
        #               (d_bcL, d_bcR))
        
        # send information
        req1 = comm.isend(d_bcR, dest=up_nbr)  # to upstream neighbour
        req2 = comm.isend(d_bcL, dest=down_nbr)  # to downstream neighbour 

        # recieve information
        h_bcL = comm.recv(source=down_nbr)  # from downstream neighbour
        h_bcR = comm.recv(source=up_nbr)    # from upstream neighbour
        MPI.Request.Waitall([req1, req2])

        # set information at left, right boundary
        if h_bcL: 
            d_bcL.set(h_bcL)
        else:
            transferBC_L_Op(d_uL, d_bcL)

        if h_bcR: 
            d_bcR.set(h_bcR)
        else:
            transferBC_R_Op(d_uR, d_bcR)

        # The physical-periodic boundary condition
        if bcl_type == 'dgfs-cyclic' and bcr_type == 'dgfs-cyclic':
            # At the left boundary, receive from the right-most communicator
            # At the right boundary, receive from the left-most communicator
            if comm.rank>1:
                raise ValueError("Not implemented")
                #req1 = comm.isend(d_bcR, dest=0)
            else:
                #pass
                cuda.memcpy_dtod(d_bcT.ptr, d_bcL.ptr, d_bcL.nbytes)
                cuda.memcpy_dtod(d_bcL.ptr, d_bcR.ptr, d_bcR.nbytes)
                cuda.memcpy_dtod(d_bcR.ptr, d_bcT.ptr, d_bcT.nbytes)

        
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
        # Now we have f* = d_ux
        mulbyinvjac_Op(d_ux)

        # project back to coefficient space
        invMass_Op(d_ux, d_ucoeff_out) 


    def moment(d_ucoeff_in, d_moms_out):

        # reconstruct the solution
        bwdTrans_Op(d_ucoeff_in, d_M1sol)

        # Compute moments
        # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
        moments_Op(d_M1sol, d_mBGK, d_moms_out)


    def momentCoeff(d_ucoeff_in, d_moms_out):

        # Compute moments
        # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
        momentsCoeff_Op(d_ucoeff_in, d_mBGK, d_momsCoeff)

        # reconstruct the solution
        bwdTransMom_Op(d_momsCoeff, d_moms_out)


    def constructMaxwellian(d_moms_in, d_Mcoeff_out, d_nuCoeff_out):

        # construct maxwellian
        cmaxwellian_Op(d_M1sol, d_moms_in, d_collFreq)

        # project collfreq to coeff space
        fwdTransOne_Op(d_collFreq, d_nuCoeff_out)

        # project back to coefficient space
        fwdTrans_Op(d_M1sol, d_Mcoeff_out)

    def momentESCoeff(d_ucoeff_in, d_momsES_out):

        # Compute moments
        # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
        momentsESCoeff_Op(d_ucoeff_in, d_mBGKES, d_momsESCoeff)

        # reconstruct the solution
        bwdTransMomES_Op(d_momsESCoeff, d_momsES_out)

        # normalize the moments
        #momentsESNorm_Op(d_momsES_out)

    def constructMaxwellianES(d_momsES_in, d_Mcoeff_out, d_nuCoeff_out):

        # construct maxwellian
        cmaxwellianES_Op(d_M1sol, d_momsES_in, d_collFreq)

        # project collfreq to coeff space
        fwdTransOne_Op(d_collFreq, d_nuCoeff_out)

        # project back to coefficient space
        fwdTrans_Op(d_M1sol, d_Mcoeff_out)


    def momentSKCoeff(d_ucoeff_in, d_momsSK_out):

        # Compute moments
        # \int f (1 v v^2) = ( rho, U, rho(U \cdot U) + 1.5 \rho T )
        momentsSKCoeff_Op(d_ucoeff_in, d_mSK, d_momsSKCoeff)

        # reconstruct the solution
        bwdTransMomSK_Op(d_momsSKCoeff, d_momsSK_out)

        # normalize the moments
        #momentsESNorm_Op(d_momsES_out)

    def constructMaxwellianSK(d_momsSK_in, d_Mcoeff_out, d_nuCoeff_out):

        # construct maxwellian
        cmaxwellianSK_Op(d_M1sol, d_momsSK_in, d_collFreq)

        # project collfreq to coeff space
        fwdTransOne_Op(d_collFreq, d_nuCoeff_out)

        # project back to coefficient space
        fwdTrans_Op(d_M1sol, d_Mcoeff_out)



    def Q(d_ucoeff_in, d_Q_out):

        # Step:4 Compute collision kernel contribution
        #Q += Q(\sum U^{m}_{ar} ucoeff_{aej}, \sum V^{m}_{ra} ucoeff_{aej})
        axnpbyCoeff_Op(0., d_Q, 0., d_Q)
        d_collFreqCoeff.fill(0.)
        for m in range(K):
            trans_U_Op[m](d_ucoeff_in, d_f)
            trans_V_Op[m](d_ucoeff_in, d_g)
            for r, e in it.product(sigModes[m], range(Ne)):
               #sm.fs(d_f, d_g, d_Q_out, e, r, m, d_collFreqCoeff)
               sm.fs(d_f, d_g, d_Q, e, r, m, None)
        
        # Step:5 Multiply by inverse mass matrix
        invMass_Op(d_Q, d_Q_out)


    # define a time-integrator (we use Euler scheme: good enough for steady)
    odestype = cfg.lookup('time-integrator', 'scheme')
    #odescls = subclass_where(DGFSIntegratorStd, intg_kind=odestype)
    #odes = odescls(rhs, (K, Ne, Nv), cfg.dtype)

    # Finally start everything
    time = ti  # initialize time in case of restart
    nacptsteps = 0 # number of elasped steps in the current run

    d_U1 = gpuarray.empty_like(d_moms)
    d_U2 = gpuarray.empty_like(d_moms)
    d_U3 = gpuarray.empty_like(d_moms)
    d_U4 = gpuarray.empty_like(d_moms)
    d_U5 = gpuarray.empty_like(d_moms)
    d_LU1 = gpuarray.empty_like(d_moms)
    d_LU2 = gpuarray.empty_like(d_moms)
    d_LU3 = gpuarray.empty_like(d_moms)
    d_LU4 = gpuarray.empty_like(d_moms)

    d_Ues1 = gpuarray.empty_like(d_momsES)
    d_Ues2 = gpuarray.empty_like(d_momsES)
    d_Ues3 = gpuarray.empty_like(d_momsES)
    d_Ues4 = gpuarray.empty_like(d_momsES)
    d_Ues5 = gpuarray.empty_like(d_momsES)
    d_LUes1 = gpuarray.empty_like(d_momsES)
    d_LUes2 = gpuarray.empty_like(d_momsES)
    d_LUes3 = gpuarray.empty_like(d_momsES)
    d_LUes4 = gpuarray.empty_like(d_momsES)
    d_LUes5 = gpuarray.empty_like(d_momsES)
    d_Qes1 = gpuarray.empty_like(d_momsES)
    d_Qes2 = gpuarray.empty_like(d_momsES)
    d_Qes3 = gpuarray.empty_like(d_momsES)
    d_Qes4 = gpuarray.empty_like(d_momsES)
    d_Qes5 = gpuarray.empty_like(d_momsES)
    d_UesTemp = gpuarray.empty_like(d_momsES)

    d_Usk1 = gpuarray.empty_like(d_momsSK)
    d_Usk2 = gpuarray.empty_like(d_momsSK)
    d_Usk3 = gpuarray.empty_like(d_momsSK)
    d_Usk4 = gpuarray.empty_like(d_momsSK)
    d_Usk5 = gpuarray.empty_like(d_momsSK)
    d_LUsk1 = gpuarray.empty_like(d_momsSK)
    d_LUsk2 = gpuarray.empty_like(d_momsSK)
    d_LUsk3 = gpuarray.empty_like(d_momsSK)
    d_LUsk4 = gpuarray.empty_like(d_momsSK)
    d_LUsk5 = gpuarray.empty_like(d_momsSK)
    d_Qsk1 = gpuarray.empty_like(d_momsSK)
    d_Qsk2 = gpuarray.empty_like(d_momsSK)
    d_Qsk3 = gpuarray.empty_like(d_momsSK)
    d_Qsk4 = gpuarray.empty_like(d_momsSK)
    d_Qsk5 = gpuarray.empty_like(d_momsSK)
    d_UskTemp = gpuarray.empty_like(d_momsSK)


    d_nuCoeff1 = gpuarray.empty_like(d_collFreqCoeff)
    d_nuCoeff2 = gpuarray.empty_like(d_collFreqCoeff)
    d_nuCoeff3 = gpuarray.empty_like(d_collFreqCoeff)
    d_nuCoeff4 = gpuarray.empty_like(d_collFreqCoeff)
    d_nuCoeff5 = gpuarray.empty_like(d_collFreqCoeff)

    d_MFcoeff1 = gpuarray.empty_like(d_M1coeff)
    d_MFcoeff2 = gpuarray.empty_like(d_M1coeff)
    d_MFcoeff3 = gpuarray.empty_like(d_M1coeff)
    d_MFcoeff4 = gpuarray.empty_like(d_M1coeff)
    d_MFcoeff5 = gpuarray.empty_like(d_M1coeff)

    d_Fcoeff1 = gpuarray.empty_like(d_ucoeff)
    d_Fcoeff2 = gpuarray.empty_like(d_ucoeff)
    d_Fcoeff3 = gpuarray.empty_like(d_ucoeff)
    d_Fcoeff4 = gpuarray.empty_like(d_ucoeff)

    d_LFcoeff1 = gpuarray.empty_like(d_ucoeff)
    d_LFcoeff2 = gpuarray.empty_like(d_ucoeff)
    d_LFcoeff3 = gpuarray.empty_like(d_ucoeff)
    d_LFcoeff4 = gpuarray.empty_like(d_ucoeff)

    d_QFcoeff1 = gpuarray.empty_like(d_ucoeff)
    d_QFcoeff2 = gpuarray.empty_like(d_ucoeff)
    d_QFcoeff3 = gpuarray.empty_like(d_ucoeff)
    d_QFcoeff4 = gpuarray.empty_like(d_ucoeff)

    # start timer
    start = timer()

    # ------------------ 1st order IMEX schemes

    #_A = ((0.0, 0.0), (1.0, 0.0))
    #_w = (1.0, 0.0)
    #gamma = 1.
    #A = ((gamma, 0.0), (1.0-gamma, gamma))
    #w = (1.0-gamma, gamma)

    _A = (
        (0.0, 0.0), 
        (1.0, 0.0)
    )
    _w = (1.0, 0.0)
    A = (
        (0.0, 0.0), 
        (0.0, 1.0)
    )
    w = (0.0, 1.0)

    # IMEX BGK (1st order)
    if odestype=='bgk-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # compute L[ f^{n} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^{(2)}) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_BGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX ESBGK (1st order)
    if odestype=='esbgk-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # Compute the moments of the ESBGK distribution
            update1MomES_Op(dt, _A[1][0], A[1][1], d_Ues1, d_LUes1, d_Ues2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues2)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff2, d_nuCoeff2)

            # March in time 
            # compute f^{(n+1)} 
            #   = [f^{n} + dt*nu_1/Kn*(M(f^{(n+1)}))]/(1+dt*nu_1/Kn) 
            updateDistribution2_ESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX Shakov (1st order)
    if odestype=='shakov-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # Compute the moments of the ESBGK distribution
            update1MomSK_Op(dt, _A[1][0], A[1][1], d_Usk1, d_LUsk1, d_Usk2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk2)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff2, d_nuCoeff2)

            # March in time 
            # compute f^{(n+1)} 
            #   = [f^{n} + dt*nu_1/Kn*(M(f^{(n+1)}))]/(1+dt*nu_1/Kn) 
            updateDistribution2_SKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX FS/BGK (1st order)
    if odestype=='fsbgk-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            constructMaxwellian(d_U1, d_MFcoeff1, d_nuCoeff1)

            # compute L[ f^{n} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^{(2)}) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX FS/ESBGK (1st order)
    if odestype=='fsesbgk-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues1)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ f^{n} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentESCoeff(d_QFcoeff1, d_Qes1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1MomFSES_Op(dt, _A[1][0], A[1][1], 
                d_Ues1, d_LUes1, d_Qes1, d_Ues2)

            # Construct the maxwellian [ M(F^{(2)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues2)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX FS/ESBGK (1st order)
    if odestype=='fsshakov-direct-gll.asym.euler':
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk1)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ f^{n} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentSKCoeff(d_QFcoeff1, d_Qsk1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1MomFSSK_Op(dt, _A[1][0], A[1][1], 
                d_Usk1, d_LUsk1, d_Qsk1, d_Usk2)

            # Construct the maxwellian [ M(F^{(2)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk2)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSSKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post procsksing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)



    # ------------------ 2nd order IMEX schemes

    order = 0
    if "ars-222" in odestype:
        print("ars-222");
        gamma = 1.-(2.**0.5)/2.
        delta = 1. - 1./(2.*gamma)
        _A = ((0., 0., 0.), (gamma, 0., 0.), (delta, 1-delta, 0.))
        _w = (delta, 1-delta, 0.)
        A = ((0., 0., 0.), (0., gamma, 0.), (0., 1-gamma, gamma))
        w = (0., 1-gamma, gamma)
        order = 2


    # IMEX BGK (2nd order: ARS(2,2,2))
    if odestype.startswith("bgk-direct-gll.asym") and order==2:
        print("Running bgk-direct-gll.asym.ars-222")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^(2)) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_BGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff2, d_LU2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2Mom_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_U1, d_LU1, d_LU2, d_U2, d_U3)

            # Construct the maxwellian [ M(F^{(3)}) ]
            constructMaxwellian(d_U3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(3)} 
            #   = [f^{n} + dt*a'_{31}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{31}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{32}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{33}*nu/Kn]
            updateDistribution3_BGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX ESBGK (2nd order: ARS(2,2,2))
    if odestype.startswith("esbgk-direct-gll.asym") and order==2:
        print("Running esbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # Compute the moments of the ESBGK distribution
            update1MomES_Op(dt, _A[1][0], A[1][1], d_Ues1, d_LUes1, d_Ues2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomES_Op(0., d_Qes2, 1., d_Ues2)
            momentsESNorm_Op(d_Qes2)
            constructMaxwellianES(d_Qes2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_ESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff2, d_LUes2)

            # Second update of the moments of the ESBGK distribution
            update2MomES_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Ues1, d_LUes1, d_LUes2, d_Ues2, d_Ues3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            #momentsESNorm_Op(d_Ues3)
            #constructMaxwellianES(d_Ues3, d_MFcoeff3, d_nuCoeff3)
            axnpbySolMomES_Op(0., d_Qes3, 1., d_Ues3)
            momentsESNorm_Op(d_Qes3)
            constructMaxwellianES(d_Qes3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_ESBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)



    # IMEX ESBGK (2nd order: ARS(2,2,2))
    if odestype.startswith("shakov-direct-gll.asym") and order==2:
        print("Running shakov-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # Compute the moments of the SKBGK distribution
            update1MomSK_Op(dt, _A[1][0], A[1][1], d_Usk1, d_LUsk1, d_Usk2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk2)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_SKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff2, d_LUsk2)

            # Second update of the moments of the SKBGK distribution
            update2MomSK_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Usk1, d_LUsk1, d_LUsk2, d_Usk2, d_Usk3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            #momentsSKNorm_Op(d_Usk3)
            #constructMaxwellianSK(d_Usk3, d_MFcoeff3, d_nuCoeff3)
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk3)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_SKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post procsksing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)



    # IMEX Boltzmann-BGK (2nd order: ARS(2,2,2))
    if odestype.startswith("fsbgk-direct-gll.asym") and order==2:
        print("Running fsbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            constructMaxwellian(d_U1, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # U^{(2)} = U^{(1)} + dt*\moments[L(F^{(1)})]
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^(2)) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff2, d_LU2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2Mom_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_U1, d_LU1, d_LU2, d_U2, d_U3)

            # Construct the maxwellian [ M(F^{(3)}) ]
            constructMaxwellian(d_U3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_FSBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX Boltzmann-ESBGK (2nd order: ARS(2,2,2))
    if odestype.startswith("fsesbgk-direct-gll.asym") and order==2:
        print("Running fsesbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues1)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentESCoeff(d_QFcoeff1, d_Qes1)

            # Compute the moments of the ESBGK distribution
            update1MomFSES_Op(dt, _A[1][0], A[1][1], 
                d_Ues1, d_LUes1, d_Qes1, d_Ues2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues2)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            #---------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff2, d_LUes2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: \moments[Q(F^{(2)})]
            momentESCoeff(d_QFcoeff2, d_Qes2)

            # Second update of the moments of the ESBGK distribution
            update2MomFSES_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Ues1, 
                d_LUes1, d_LUes2, 
                d_Qes1, d_Qes2, 
                d_Ues2, d_Ues3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            #momentsESNorm_Op(d_Ues3)
            #constructMaxwellianES(d_Ues3, d_MFcoeff3, d_nuCoeff3)
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues3)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_FSESBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX Boltzmann-Shakov (2nd order: ARS(2,2,2))
    if odestype.startswith("fsshakov-direct-gll.asym") and order==2:
        print("Running fsshakov-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk1)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentSKCoeff(d_QFcoeff1, d_Qsk1)

            # Compute the moments of the SKBGK distribution
            update1MomFSSK_Op(dt, _A[1][0], A[1][1], 
                d_Usk1, d_LUsk1, d_Qsk1, d_Usk2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk2)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSSKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            #---------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff2, d_LUsk2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: \moments[Q(F^{(2)})]
            momentSKCoeff(d_QFcoeff2, d_Qsk2)

            # Second update of the moments of the SKBGK distribution
            update2MomFSSK_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Usk1, 
                d_LUsk1, d_LUsk2, 
                d_Qsk1, d_Qsk2, 
                d_Usk2, d_Usk3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk3)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_FSSKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_ucoeff)

            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post procsksing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)




    # ------------------ 3rd order IMEX schemes

    if "ars-443" in odestype:
        print("ars-443");
        _A = (
                (0., 0., 0., 0., 0.), 
                (1./2., 0., 0., 0., 0.), 
                (11./18., 1./18., 0., 0., 0.),
                (5./6., -5./6., 1./2., 0., 0.),
                (1./4., 7./4., 3./4., -7./4., 0.)
        )
        _w = (1./4., 7./4., 3./4., -7./4., 0.)
        A = (
                (0., 0., 0., 0., 0.), 
                (0., 1./2., 0., 0., 0.), 
                (0., 1./6., 1./2., 0., 0.),
                (0., -1./2., 1./2., 1./2., 0.),
                (0., 3./2., -3./2., 1./2., 1./2.)
        )
        w = (0., 3./2., -3./2., 1./2., 1./2.)
        order = 3

    elif "imex-II-gsa2-442" in odestype:
        print("imex-II-gsa2-442");
        _A = (
                (0., 0., 0., 0., 0.), 
                (1./4., 0., 0., 0., 0.), 
                (1./6., 1./6., 0., 0., 0.),
                (-2./3., 0., 4./3., 0., 0.),
                (-1./16., 1./2., 0., 9./16., 0.)
        )
        _w = (-1./16., 1./2., 0., 9./16., 0.)
        A = (
                (0., 0., 0., 0., 0.), 
                (0., 1./4., 0., 0., 0.), 
                (0., 1./12., 1./4., 0., 0.),
                (0., -11./12., 4./3., 1./4., 0.),
                (0., 9./31., 12./31., 9./124., 1./4.)
        )
        w = (0., 9./31., 12./31., 9./124., 1./4.)
        order = 3


    # IMEX BGK (3rd order)
    if odestype.startswith("bgk-direct-gll.asym") and order==3:
        print("Running bgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # Compute the moments of the BGK distribution
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^(2)) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{21}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{21}*nu/Kn]
            updateDistribution2_BGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # ------------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: \moments[L(F^{(2)})]
            momentCoeff(d_LFcoeff2, d_LU2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2Mom_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_U1, d_LU1, d_LU2, d_U2, d_U3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            constructMaxwellian(d_U3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(3)} 
            #   = [f^{n} + dt*a'_{31}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{31}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{32}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{32}*nu/Kn]
            updateDistribution3_BGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)

            # ------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentCoeff(d_LFcoeff3, d_LU3)

            # U^{(4)} = U^{(1)} + dt*a'_{41}*\moments[L(F^{(1)})]
            #             + dt*a'_{42}*\moments[L(F^{(2)})]
            #             + dt*a'_{43}*\moments[L(F^{(3)})]
            update3Mom_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_U1, d_LU1, d_LU2, d_LU3, d_U2, d_U3, d_U4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            constructMaxwellian(d_U4, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_BGKARS_Op(dt, _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)

            # ------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentCoeff(d_LFcoeff4, d_LU4)

            # U^{(5)} = U^{(n)} + dt*a'_{51}*\moments[L(F^{(1)})]
            #           + dt*a'_{52}*\moments[L(F^{(2)})] 
            #           + dt*a'_{53}*\moments[L(F^{(3)})]
            #           + dt*a'_{54}*\moments[L(F^{(4)})]
            update4Mom_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_U1, d_LU1, d_LU2, d_LU3, d_LU4, 
                d_U2, d_U3, d_U4, d_U5)
            
            # Construct the maxwellian [ M(F^{(5)}) ]
            constructMaxwellian(d_U5, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_BGKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4, 
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # ------------


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX ESBGK (3rd order)
    if odestype.startswith("esbgk-direct-gll.asym") and order==3:
        print("Running esbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # Compute the moments of the ESBGK distribution
            update1MomES_Op(dt, _A[1][0], A[1][1], d_Ues1, d_LUes1, d_Ues2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomES_Op(0., d_Qes2, 1., d_Ues2)
            momentsESNorm_Op(d_Qes2)
            constructMaxwellianES(d_Qes2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{21}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{21}*nu/Kn]
            updateDistribution2_ESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # ------------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: \moments[L(F^{(2)})]
            momentESCoeff(d_LFcoeff2, d_LUes2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2MomES_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Ues1, d_LUes1, d_LUes2, d_Ues2, d_Ues3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            #momentsESNorm_Op(d_Ues3)
            #constructMaxwellianES(d_Ues3, d_MFcoeff3, d_nuCoeff3)
            axnpbySolMomES_Op(0., d_Qes3, 1., d_Ues3)
            momentsESNorm_Op(d_Qes3)
            constructMaxwellianES(d_Qes3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(3)} 
            #   = [f^{n} + dt*a'_{31}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{31}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{32}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{32}*nu/Kn]
            updateDistribution3_ESBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)

            # ------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentESCoeff(d_LFcoeff3, d_LUes3)

            # U^{(4)} = U^{(1)} + dt*a'_{41}*\moments[L(F^{(1)})]
            #             + dt*a'_{42}*\moments[L(F^{(2)})]
            #             + dt*a'_{43}*\moments[L(F^{(3)})]
            update3MomES_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_Ues1, d_LUes1, d_LUes2, d_LUes3, d_Ues2, d_Ues3, d_Ues4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            #momentsESNorm_Op(d_Ues4)
            #constructMaxwellianES(d_Ues4, d_MFcoeff4, d_nuCoeff4)
            axnpbySolMomES_Op(0., d_Qes4, 1., d_Ues4)
            momentsESNorm_Op(d_Qes4)
            constructMaxwellianES(d_Qes4, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_ESBGKARS_Op(dt, _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)

            # ------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentESCoeff(d_LFcoeff4, d_LUes4)

            # U^{(5)} = U^{(n)} + dt*a'_{51}*\moments[L(F^{(1)})]
            #           + dt*a'_{52}*\moments[L(F^{(2)})] 
            #           + dt*a'_{53}*\moments[L(F^{(3)})]
            #           + dt*a'_{54}*\moments[L(F^{(4)})]
            update4MomES_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_Ues1, d_LUes1, d_LUes2, d_LUes3, d_LUes4, 
                d_Ues2, d_Ues3, d_Ues4, d_Ues5)
            
            # Construct the maxwellian [ M(F^{(5)}) ]
            #momentsESNorm_Op(d_Ues5)
            #constructMaxwellianES(d_Ues5, d_MFcoeff5, d_nuCoeff5)
            axnpbySolMomES_Op(0., d_Qes5, 1., d_Ues5)
            momentsESNorm_Op(d_Qes5)
            constructMaxwellianES(d_Qes5, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_ESBGKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4, 
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # ------------


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX ESBGK (3rd order)
    if odestype.startswith("shakov-direct-gll.asym") and order==3:
        print("Running shakov-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # Compute the moments of the SKBGK distribution
            update1MomSK_Op(dt, _A[1][0], A[1][1], d_Usk1, d_LUsk1, d_Usk2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomSK_Op(0., d_Qsk2, 1., d_Usk2)
            momentsSKNorm_Op(d_Qsk2)
            constructMaxwellianSK(d_Qsk2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{21}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{21}*nu/Kn]
            updateDistribution2_SKARS_Op(dt, _A[1][0], A[1][1], 
                d_ucoeff, d_LFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # ------------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: \moments[L(F^{(2)})]
            momentSKCoeff(d_LFcoeff2, d_LUsk2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2MomSK_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Usk1, d_LUsk1, d_LUsk2, d_Usk2, d_Usk3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            #momentsSKNorm_Op(d_Usk3)
            #constructMaxwellianSK(d_Usk3, d_MFcoeff3, d_nuCoeff3)
            axnpbySolMomSK_Op(0., d_Qsk3, 1., d_Usk3)
            momentsSKNorm_Op(d_Qsk3)
            constructMaxwellianSK(d_Qsk3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(3)} 
            #   = [f^{n} + dt*a'_{31}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{31}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{32}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{32}*nu/Kn]
            updateDistribution3_SKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)

            # ------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentSKCoeff(d_LFcoeff3, d_LUsk3)

            # U^{(4)} = U^{(1)} + dt*a'_{41}*\moments[L(F^{(1)})]
            #             + dt*a'_{42}*\moments[L(F^{(2)})]
            #             + dt*a'_{43}*\moments[L(F^{(3)})]
            update3MomSK_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_Usk1, d_LUsk1, d_LUsk2, d_LUsk3, d_Usk2, d_Usk3, d_Usk4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            #momentsSKNorm_Op(d_Usk4)
            #constructMaxwellianSK(d_Usk4, d_MFcoeff4, d_nuCoeff4)
            axnpbySolMomSK_Op(0., d_Qsk4, 1., d_Usk4)
            momentsSKNorm_Op(d_Qsk4)
            constructMaxwellianSK(d_Qsk4, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_SKARS_Op(dt, _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)

            # ------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentSKCoeff(d_LFcoeff4, d_LUsk4)

            # U^{(5)} = U^{(n)} + dt*a'_{51}*\moments[L(F^{(1)})]
            #           + dt*a'_{52}*\moments[L(F^{(2)})] 
            #           + dt*a'_{53}*\moments[L(F^{(3)})]
            #           + dt*a'_{54}*\moments[L(F^{(4)})]
            update4MomSK_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_Usk1, d_LUsk1, d_LUsk2, d_LUsk3, d_LUsk4, 
                d_Usk2, d_Usk3, d_Usk4, d_Usk5)
            
            # Construct the maxwellian [ M(F^{(5)}) ]
            #momentsSKNorm_Op(d_Usk5)
            #constructMaxwellianSK(d_Usk5, d_MFcoeff5, d_nuCoeff5)
            axnpbySolMomSK_Op(0., d_Qsk5, 1., d_Usk5)
            momentsSKNorm_Op(d_Qsk5)
            constructMaxwellianSK(d_Qsk5, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_SKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4], 
                d_ucoeff, d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4, 
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # ------------


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post procsksing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)



    # IMEX Boltzmann-BGK (3rd order)
    if odestype.startswith("fsbgk-direct-gll.asym") and order==3:
        print("Running fsbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentCoeff(d_ucoeff, d_U1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            constructMaxwellian(d_U1, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentCoeff(d_LFcoeff1, d_LU1)

            # Compute the moments of the BGK distribution
            update1Mom_Op(dt, _A[1][0], A[1][1], d_U1, d_LU1, d_U2)

            # Construct the maxwellian [ M(F^(2)) ]
            constructMaxwellian(d_U2, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            # ------------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: \moments[L(F^{(2)})]
            momentCoeff(d_LFcoeff2, d_LU2)

            # U^{(3)} = U^{(1)} + dt*a'_{31}*\moments[L(F^{(1)})]
            #             + dt*a'_{32}*\moments[L(F^{(2)})]
            update2Mom_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_U1, d_LU1, d_LU2, d_U2, d_U3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            constructMaxwellian(d_U3, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(3)} 
            #   = [f^{n} + dt*a'_{31}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{31}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{32}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{32}*nu/Kn]
            updateDistribution3_FSBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)

            # ------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # collision Q[ F^{(3)} ] 
            Q(d_Fcoeff3, d_QFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentCoeff(d_LFcoeff3, d_LU3)

            # U^{(4)} = U^{(1)} + dt*a'_{41}*\moments[L(F^{(1)})]
            #             + dt*a'_{42}*\moments[L(F^{(2)})]
            #             + dt*a'_{43}*\moments[L(F^{(3)})]
            update3Mom_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_U1, d_LU1, d_LU2, d_LU3, d_U2, d_U3, d_U4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            constructMaxwellian(d_U4, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_FSBGKARS_Op(dt, _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, 
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)

            # ------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # collision Q[ F^{(4)} ] 
            Q(d_Fcoeff4, d_QFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentCoeff(d_LFcoeff4, d_LU4)

            # U^{(5)} = U^{(n)} + dt*a'_{51}*\moments[L(F^{(1)})]
            #           + dt*a'_{52}*\moments[L(F^{(2)})] 
            #           + dt*a'_{53}*\moments[L(F^{(3)})]
            #           + dt*a'_{54}*\moments[L(F^{(4)})]
            update4Mom_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_U1, d_LU1, d_LU2, d_LU3, d_LU4, 
                d_U2, d_U3, d_U4, d_U5)
            
            # Construct the maxwellian [ M(F^{(5)}) ]
            constructMaxwellian(d_U5, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_FSBGKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3, d_QFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4, 
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # ------------


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX Boltzmann-ESBGK (3rd order)
    if odestype.startswith("fsesbgk-direct-gll.asym") and order==3:
        print("Running fsesbgk-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentESCoeff(d_ucoeff, d_Ues1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues1)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff1, d_LUes1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentESCoeff(d_QFcoeff1, d_Qes1)

            # Compute the moments of the ESBGK distribution
            update1MomFSES_Op(dt, _A[1][0], A[1][1], 
                d_Ues1, d_LUes1, d_Qes1, d_Ues2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues2)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSESBGKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            #---------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentESCoeff(d_LFcoeff2, d_LUes2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: \moments[Q(F^{(2)})]
            momentESCoeff(d_QFcoeff2, d_Qes2)

            # Second update of the moments of the ESBGK distribution
            update2MomFSES_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Ues1, 
                d_LUes1, d_LUes2, 
                d_Qes1, d_Qes2, 
                d_Ues2, d_Ues3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues3)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_FSESBGKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)


            #--------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentESCoeff(d_LFcoeff3, d_LUes3)

            # collision Q[ F^{(3)} ] 
            Q(d_Fcoeff3, d_QFcoeff3)

            # Compute the moments: \moments[Q(F^{(3)})]
            momentESCoeff(d_QFcoeff3, d_Qes3)

            # Second update of the moments of the ESBGK distribution
            update3MomFSES_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_Ues1, 
                d_LUes1, d_LUes2, d_LUes3, 
                d_Qes1, d_Qes2, d_Qes3, 
                d_Ues2, d_Ues3, d_Ues4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues4)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_FSESBGKARS_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3,
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)


            #--------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentESCoeff(d_LFcoeff4, d_LUes4)

            # collision Q[ F^{(4)} ] 
            Q(d_Fcoeff4, d_QFcoeff4)

            # Compute the moments: \moments[Q(F^{(4)})]
            momentESCoeff(d_QFcoeff4, d_Qes4)

            # Second update of the moments of the ESBGK distribution
            update4MomFSES_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_Ues1, 
                d_LUes1, d_LUes2, d_LUes3, d_LUes4, 
                d_Qes1, d_Qes2, d_Qes3, d_Qes4, 
                d_Ues2, d_Ues3, d_Ues4, d_Ues5)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            axnpbySolMomES_Op(0., d_UesTemp, 1., d_Ues5)
            momentsESNorm_Op(d_UesTemp)
            constructMaxwellianES(d_UesTemp, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_FSESBGKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3, d_QFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4,
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post processing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)


    # IMEX Boltzmann-Shakov (3rd order)
    if odestype.startswith("fsshakov-direct-gll.asym") and order==3:
        print("Running fsshakov-direct-gll.asym")
        while(time < tf):

            # Compute the moments [ U^{(1)} = U^{(n)} ]
            momentSKCoeff(d_ucoeff, d_Usk1)

            # Construct the maxwellian [ M(F^{(1)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk1)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff1, d_nuCoeff1)

            # compute L[ F^{(1)} ]
            L(dt, d_ucoeff, d_LFcoeff1)

            # Compute the moments: \moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff1, d_LUsk1)

            # collision Q[ F^{(1)} ] 
            Q(d_ucoeff, d_QFcoeff1)

            # Compute the moments: \moments[Q(F^{(1)})]
            momentSKCoeff(d_QFcoeff1, d_Qsk1)

            # Compute the moments of the SKBGK distribution
            update1MomFSSK_Op(dt, _A[1][0], A[1][1], 
                d_Usk1, d_LUsk1, d_Qsk1, d_Usk2)

            # Construct the maxwellian [ M(F^(2)) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk2)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff2, d_nuCoeff2)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a_{11}*nu/Kn*(M(F^{(1)}))]/[1+dt*a_{11}*nu/Kn]
            updateDistribution2_FSSKARS_Op(dt, _A[1][0], A[1][1], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1,
                d_QFcoeff1,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2)

            #---------

            # compute L[ F^{(2)} ]
            L(dt, d_Fcoeff2, d_LFcoeff2)

            # Compute the moments: U^{(2)} = U^{(n)} + a21*dt*\moments[L(F^{(1)})]
            momentSKCoeff(d_LFcoeff2, d_LUsk2)

            # collision Q[ F^{(2)} ] 
            Q(d_Fcoeff2, d_QFcoeff2)

            # Compute the moments: \moments[Q(F^{(2)})]
            momentSKCoeff(d_QFcoeff2, d_Qsk2)

            # Second update of the moments of the SKBGK distribution
            update2MomFSSK_Op(dt, 
                _A[2][0], _A[2][1], A[2][1], A[2][2], 
                d_Usk1, 
                d_LUsk1, d_LUsk2, 
                d_Qsk1, d_Qsk2, 
                d_Usk2, d_Usk3)
            
            # Construct the maxwellian [ M(F^{(3)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk3)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff3, d_nuCoeff3)

            # compute F^{(2)} 
            #   = [f^{n} + dt*a'_{21}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{21}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{22}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution3_FSSKARS_Op(dt, _A[2][0], _A[2][1], 
                A[2][1], A[2][2], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, 
                d_QFcoeff1, d_QFcoeff2, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3)


            #--------------

            # compute L[ F^{(3)} ]
            L(dt, d_Fcoeff3, d_LFcoeff3)

            # Compute the moments: \moments[L(F^{(3)})]
            momentSKCoeff(d_LFcoeff3, d_LUsk3)

            # collision Q[ F^{(3)} ] 
            Q(d_Fcoeff3, d_QFcoeff3)

            # Compute the moments: \moments[Q(F^{(3)})]
            momentSKCoeff(d_QFcoeff3, d_Qsk3)

            # Second update of the moments of the SKBGK distribution
            update3MomFSSK_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], A[3][1], A[3][2], A[3][3], 
                d_Usk1, 
                d_LUsk1, d_LUsk2, d_LUsk3, 
                d_Qsk1, d_Qsk2, d_Qsk3, 
                d_Usk2, d_Usk3, d_Usk4)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk4)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff4, d_nuCoeff4)

            # compute F^{(4)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution4_FSSKARS_Op(dt, 
                _A[3][0], _A[3][1], _A[3][2], 
                A[3][1], A[3][2], A[3][3], 
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3,
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3,
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4)


            #--------------

            # compute L[ F^{(4)} ]
            L(dt, d_Fcoeff4, d_LFcoeff4)

            # Compute the moments: \moments[L(F^{(4)})]
            momentSKCoeff(d_LFcoeff4, d_LUsk4)

            # collision Q[ F^{(4)} ] 
            Q(d_Fcoeff4, d_QFcoeff4)

            # Compute the moments: \moments[Q(F^{(4)})]
            momentSKCoeff(d_QFcoeff4, d_Qsk4)

            # Second update of the moments of the SKBGK distribution
            update4MomFSSK_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3],
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_Usk1, 
                d_LUsk1, d_LUsk2, d_LUsk3, d_LUsk4, 
                d_Qsk1, d_Qsk2, d_Qsk3, d_Qsk4, 
                d_Usk2, d_Usk3, d_Usk4, d_Usk5)
            
            # Construct the maxwellian [ M(F^{(4)}) ]
            axnpbySolMomSK_Op(0., d_UskTemp, 1., d_Usk5)
            momentsSKNorm_Op(d_UskTemp)
            constructMaxwellianSK(d_UskTemp, d_MFcoeff5, d_nuCoeff5)

            # compute F^{(5)} 
            #   = [f^{n} + dt*a'_{41}*(L[F^{(1)}] + 1/Kn*G_p[F^{(1)}]) 
            #       + dt*a_{41}*nu/Kn*(M(F^{(1)})-F^{(1)}) 
            #       + dt*a_{42}*nu/Kn*(M(F^{(2)}))]/[1+dt*a_{22}*nu/Kn]
            updateDistribution5_FSSKARS_Op(dt, 
                _A[4][0], _A[4][1], _A[4][2], _A[4][3], 
                A[4][1], A[4][2], A[4][3], A[4][4],
                d_nuCoeff1, d_MFcoeff1, d_ucoeff, 
                d_LFcoeff1, d_LFcoeff2, d_LFcoeff3, d_LFcoeff4, 
                d_QFcoeff1, d_QFcoeff2, d_QFcoeff3, d_QFcoeff4, 
                d_nuCoeff2, d_MFcoeff2, d_Fcoeff2, 
                d_nuCoeff3, d_MFcoeff3, d_Fcoeff3, 
                d_nuCoeff4, d_MFcoeff4, d_Fcoeff4,
                d_nuCoeff5, d_MFcoeff5, d_ucoeff)


            # increment time
            time += dt 
            nacptsteps += 1

            # Final step: post procsksing routines
            residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
            moments(dt, time, d_ucoeff)
            distribution(dt, time, d_ucoeff)

            # copy the solution for the next time step
            cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)




    # print elasped time
    end =  timer()
    elapsed = np.array([end - start])
    if rank==root:
        comm.Allreduce(get_mpi('in_place'), elapsed, op=get_mpi('sum'))
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

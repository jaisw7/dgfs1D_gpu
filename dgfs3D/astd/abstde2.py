# -*- coding: utf-8 -*-

""" 
Asymptotic DGFS in one dimension
"""

import os
import numpy as np
import itertools as it
from timeit import default_timer as timer
np.set_printoptions(linewidth=1200, precision=2)
from functools import reduce

import mpi4py.rc
mpi4py.rc.initialize = False
import matplotlib.pyplot as plt

from dgfs3D.mesh import Mesh
from dgfs1D.initialize import initialize
from dgfs1D.nputil import (subclass_where, get_grid_for_block, 
                            DottedTemplateLookup, ndrange)
from dgfs1D.nputil import get_comm_rank_root, get_local_rank, get_mpi
from dgfs3D.basis import Basis
from dgfs1D.std.velocitymesh import DGFSVelocityMeshStd

from dgfs1D.astd.scattering import DGFSScatteringModelAstd
from dgfs1D.std.scattering import DGFSScatteringModelStd
from dgfs3D.astd.initcond import DGFSInitConditionStd
from dgfs1D.std.moments import DGFSMomWriterStd
from dgfs1D.std.residual import DGFSResidualStd
from dgfs1D.std.distribution import DGFSDistributionStd
from dgfs3D.astd.bc import DGFSBCStd
from dgfs1D.axnpby import get_axnpby_kerns, copy
from dgfs1D.util import get_kernel, filter_tol, check
from dgfs1D.astd.integrator import DGFSIntegratorAstd

from pycuda import compiler, gpuarray
import pycuda.driver as cuda
from gimmik import generate_mm

def plot(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.savefig('plot.pdf')

def main():
    # who am I in this world? (Bulleh Shah, 18th century sufi poet)
    comm, rank, root = get_comm_rank_root()

    # read the inputs (from people)
    cfg, args = initialize()
    mesh = Mesh(cfg)
    dim = cfg.dim
    #assert dim==2, "Should be two dimensional"

    # define 3D mesh (construct a 3D world view)
    xmesh = mesh.xmesh

    # number of elements (how refined perspectives do we want/have?)
    Ne = mesh.Ne[:dim]
    NeT = reduce(lambda a, b: a*b, Ne[:dim])

    #xmeshes = np.meshgrid(*xmesh)
    #plt.contourf(x, y, np.sin(2*np.pi*x)*np.cos(2*np.pi*y))
    #plt.savefig('plot.pdf')
    #print(Ne)
    #exit(0)
    #print(xmesh.shape)

    # define the basis (what is the basis for those perspectives?)
    bsKind = cfg.lookup('basis', 'kind')
    basiscls = subclass_where(Basis, basis_kind=bsKind)
    basis = basiscls(cfg)

    # number of local degrees of freedom (depth/granualirity of perspectives)
    K = basis.K
    print(K, Ne, NeT)

    # number of solution points (how far can I interpolate my learning)
    Nq = basis.Nq

    # left/right face maps
    Nqf = basis.Nqf  # number of points used in reconstruction at each face
    #mapL, mapR = np.arange(Ne+1)+(Nqf-1)*Ne-1, np.arange(Ne+1)
    #mapL[0], mapR[-1] = 0, Ne*Nqf-1
    #Nf = len(mapL)
    Nfaces = 2*dim
    NqfT = Nqf*Nfaces
    print("NqfT", NqfT)

    prod = lambda a, b: a*b

    # boundary elements
    # (0, .), (Ne[0], .); (., 0), (., Ne[1])
    bndElems = [0]*Nfaces
    calIdx = lambda v: reduce(prod, v)
    reducer = lambda v: sum(v[j]*calIdx(Ne[:j+1]) if j!=dim-1 else v[j] for j in range(dim))
    calcNe = lambda e: sum([v*calIdx(Ne[:d+1]) for d, v in enumerate(reversed(e[:-1]))]) + e[-1]
    for i in range(dim):
      for c, idx in enumerate([0,Ne[i]-1]):
        NeL = list(reversed([range(Ne[j]) if j!=i else [idx] for j in range(dim)]))
        if dim==3: reducer = calcNe
        bndElems[2*i+c] = list(map(reducer, it.product(*NeL)))
    Nb = list(map(len, bndElems))
    #print(bndElems); exit(0)

    # the zeros
    z1 = basis.z1
    z = basis.z

    # jacobian of the mapping from D^{st}=[-1,1] to D
    jac, invjac = mesh.jac, mesh.invjac
    pex = lambda *args: print(*args)+ exit(0)
    #pex(invjac[0][0,0])

    # load the velocity mesh
    vm = DGFSVelocityMeshStd(cfg)
    Nv = vm.vsize()

    # load the scattering model
    smn = cfg.lookup('scattering-model', 'type')
    scatteringcls = subclass_where(DGFSScatteringModelStd, 
        scattering_model=smn)
    sm = scatteringcls(cfg, vm, Ne=NeT)

    # load the scattering model
    smn = cfg.lookup('penalized-scattering-model', 'type')
    pscatteringcls = subclass_where(DGFSScatteringModelAstd, 
        scattering_model=smn)
    psm = pscatteringcls(cfg, vm, Ne=NeT, eps=sm.prefac)

    # initial time, time step, final time
    ti, dt, tf = cfg.lookupfloats('time-integrator', ('tstart', 'dt', 'tend'))
    nsteps = np.ceil((tf - ti)/dt)
    dt = (tf - ti)/nsteps

    # Compute the location of the solution points 
    #xsol = [np.array([0.5*(xmesh[i][j]+xmesh[i][j+1])+jac[i][j]*z1
    #                  for j in range(Ne[i])]).T for i in range(dim)]
    #xcoeff = np.einsum("kq,qe->ke", basis.fwdTransMat, xsol)
 
    xsol = [np.zeros((Nq,NeT)) for i in range(dim)]

    for e in ndrange(*Ne):
        #idx = sum([v*calIdx(Ne[:d+1]) for d, v in enumerate(reversed(e[:-1]))]) + e[-1]
        idx = sum([v*calIdx(Ne[:d+1]) for d, v in enumerate(e[1:])]) + e[0]
        for d in range(dim):
            xsol[d][:,idx] = 0.5*(xmesh[d][e[d]]+xmesh[d][e[d]+1]) + jac[d][e[d]]*z[d].ravel()

    #if dim==2:
    #  for e0, e1 in ndrange(*Ne):
    #      xsol[0][:,e1*Ne[0]+e0] = 0.5*(xmesh[0][e0]+xmesh[0][e0+1]) + jac[0][e0]*z[0].ravel()
    #      xsol[1][:,e1*Ne[0]+e0] = 0.5*(xmesh[1][e1]+xmesh[1][e1+1]) + jac[1][e1]*z[1].ravel()

    #if dim==3:
    #  for e0, e1, e2 in ndrange(*Ne):
    #      idx = e2*Ne[0]*Ne[1]+e1*Ne[0]+e0
    #      #idx = e0*Ne[1]*Ne[2] + e1*Ne[2] + e2
    #      #print(e0,e1,e2,idx)
    #      xsol[0][:,idx] = 0.5*(xmesh[0][e0]+xmesh[0][e0+1]) + jac[0][e0]*z[0].ravel()
    #      xsol[1][:,idx] = 0.5*(xmesh[1][e1]+xmesh[1][e1+1]) + jac[1][e1]*z[1].ravel()
    #      xsol[2][:,idx] = 0.5*(xmesh[2][e2]+xmesh[2][e2+1]) + jac[2][e2]*z[2].ravel()

    Nq1, Ne1 = int(np.ceil(Nq**(1./dim))), Ne[0]
    def form(dat):
        return dat.reshape(Nq1,Nq1,Nq1,Ne1,Ne1,Ne1).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Nq1*Ne1,Nq1*Ne1,Nq1*Ne1)
          
    #pex(form(xsol[0]).ravel())
    #pex(xsol[2])
    #pex(xsol, dim)
    #pex(*map(filter_tol, xsol))

    # define connectivity
    import networkx as nx
    G = nx.grid_graph(dim=Ne.tolist())
    #calcNe = lambda e: sum([v*calIdx(Ne[:d+1]) for d, v in enumerate(e[1:])]) + e[0]
    mapping = {n: calcNe(n) for n in G.nodes()}
    #pex(mapping)
    #G = nx.relabel_nodes(G, mapping)

    mapL, mapR = map(lambda v: np.arange(NqfT*NeT,dtype=int).reshape(Nfaces,Nqf,NeT), range(2))
    one = np.ones((1,NqfT))
    for e0, e1 in G.edges():
        #xM, xP = xFace[:,:,e0], xFace[:,:,e1]
        #vM, vP = mapL[:,e0], mapL[:,e1]
        #for f in range(Nfaces):
        #  nf = e2f[e0,f]
        #  vM, vP = mapL[:,f,e0], mapL[:,nf,e1]
 
        diff = np.array(e1) - np.array(e0)
        dir = np.where(abs(diff) > 0)[0][0]
        f = 2*(dim-1-dir) + (0 if diff[dir]<0 else 1)
        nf = 2*(dim-1-dir) + (0 if diff[dir]>0 else 1)
        #print(e0, e1, f, nf)

        e0_, e1_ = mapping[e0], mapping[e1]
        vM, vP = mapL[f,:,e0_], mapL[nf,:,e1_]
        mapR[f,:,e0_] = vP
        mapR[nf,:,e1_] = vM

        #for i,j in ndrange(NqfT,NqfT):
        #  if np.sum((xM[:,i]-xP[:,j])**2, axis=0) < 1e-6:
        #    mapR[i,e0] = vP[j]
        #    #mapR[j,e1] = vM[i] 
  
    #print(mapR.ravel())
    e0 = (0,0,0)
    e1 = (0,1,0)
    f, nf = 3, 0
    #e0_, e1_ = mapping[e0], mapping[e1]
    #print(e0_, e1_)
    #pex(mapL[f,:,e0_], mapR[f,:,e0_])

    #pex(mapR)
    mapL, mapR = map(lambda v: v.reshape(NqfT,NeT), (mapL, mapR))
    #pex(mapR)
    #pex(*map(lambda v: v.shape, xsol), basis._Bqf.T.shape)
    #pex(xsol)

    xFace = np.array([np.matmul(basis._Bqf.T, x) for x in xsol])
    #print(xFace)
    nFace = np.zeros_like(xFace)
    #print(nFace.shape, Nqf)
    for i in range(dim):
      for dir, b in enumerate([2*i, 2*i+1]):
        norm = -1 if dir==0 else 1
        nFace[i,b*Nqf:(b+1)*Nqf,:] = norm

    xFace = xFace.reshape(dim,-1)
    nFace = nFace.reshape(dim,-1)
    #pex(*[(nFace[0,i],nFace[1,i]) for i in range(NqfT*NeT)])
  
    # define boundary maps
    bmapL = [np.zeros((Nqf,Nb[b],), dtype=int) for b in range(Nfaces)]
    bmapR = [np.zeros((Nqf,Nb[b],), dtype=int) for b in range(Nfaces)]
    for i in range(dim):
        for dir, b in enumerate([2*i, 2*i+1]):
            norm = -1 if dir==0 else 1
            for e0 in range(Nb[b]):
              bmapL[b][:,e0] = mapL[b*Nqf:(b+1)*Nqf,bndElems[b][e0]]
              bmapR[b][:,e0] = mapR[b*Nqf:(b+1)*Nqf,bndElems[b][e0]]

    #pex(bmapL[4])

    mapL, mapR = map(lambda v: v.ravel(), (mapL, mapR))
    for i in range(dim):
      #print(xFace[i,mapL]-xFace[i,mapR])
      assert all(np.isclose(xFace[i,mapL], xFace[i,mapR])), "Issue in connectivity"

    bmapL = [bmap.ravel() for bmap in bmapL]
    bmapR = [bmap.ravel() for bmap in bmapR]
    #pex(bmapL[4])

    bnd_x = [xFace[:,bmap] for bmap in bmapL]
    d_bnd_x = [gpuarray.to_gpu(bnd.swapaxes(0,1).ravel()) for bnd in bnd_x]
    #pex(bnd_x[5])

    xsol = np.stack(xsol,2)
    d_xsol = gpuarray.to_gpu(xsol.ravel())
    d_mapL, d_mapR = map(lambda v: gpuarray.to_gpu(v.astype(cfg.dtype).ravel()), (mapL, mapR))
    d_bmapL = [gpuarray.to_gpu(bmap.astype(cfg.dtype)) for bmap in bmapL]
    #d_bmapR = [gpuarray.to_gpu(bmap.astype(cfg.dtype)) for bmap in bmapR]
    d_nFace = gpuarray.to_gpu(nFace.swapaxes(0,1).ravel())

    # Determine the grid/block
    NeNv = NeT*Nv
    KNeNv = K*NeT*Nv
    NqNeNv = Nq*NeT*Nv
    NqfTNe = NqfT*NeT
    NqfTNeNv = NqfT*NeT*Nv
    #NfNv = Nf*Nv
    block = (128, 1, 1)
    grid_Nv = get_grid_for_block(block, Nv)
    grid_NeNv = get_grid_for_block(block, NeNv)
    grid_KNeNv = get_grid_for_block(block, KNeNv)
    grid_NqfTNeNv = get_grid_for_block(block, NqfTNeNv)
    grid_NbNeNv = get_grid_for_block(block, reduce(max, Nb)*Nqf*Nv)

    # operator generator for matrix operations
    matOpGen = lambda v: lambda arg0, arg1: v.prepared_call(
                grid_NeNv, block, NeNv, arg0.ptr, NeNv, arg1.ptr, NeNv)
    
    # forward trans, backward, backward (at faces), derivative kernels
    fwdTrans_Op, bwdTrans_Op, invMass_Op, bwdTransFace_Op, lift_Op = map(
        matOpGen, (basis.fwdTransOp, basis.bwdTransOp, basis.invMassOp, 
            basis.bwdTransFaceOp, basis.liftOp) 
    )

    # Derivatives, Reconstruction at faces kernels
    deriv_Op = tuple(map(matOpGen, basis.derivOp))
    #bwdTransFace_Op = tuple(map(matOpGen, basis.bwdTransFaceOp))
 
    # prepare the kernel for extracting face/interface values
    dfltargs = dict(
        K=K, Ne=Ne, NeT=int(NeT), Nq=Nq, vsize=int(Nv), dtype=cfg.dtypename,
        dim=cfg.dim, Dr=basis._derivMat, 
        edg = cfg.lookupordefault('config', 'edg', 0)
        #invjac=invjac, gRD=basis.gRD, gLD=basis.gLD, xsol=xsol
    )
    kernsrc = DottedTemplateLookup('dgfs3D.astd.kernels', 
                                    dfltargs).get_template('std').render()
    kernmod = compiler.SourceModule(kernsrc)

    # prepare operators for execution (see std.mako for description)
    extractTrace = get_kernel(kernmod, "extract_trace", 'iPPP')
    extractTrace_Op = lambda *args: extractTrace.prepared_call(
        grid_NqfTNeNv, block, NqfTNeNv, *list(map(lambda c: c.ptr, args))
    )
    extractBndTrace_Op = lambda *args: extractTrace.prepared_call(
        grid_NbNeNv, block, args[-1].shape[0], *list(map(lambda c: c.ptr, args))
    )

    insertTrace = get_kernel(kernmod, "insert_trace", 'iPPP')
    insertBndTrace_Op = lambda *args: insertTrace.prepared_call(
        grid_NbNeNv, block, args[-1].shape[0], *list(map(lambda c: c.ptr, args))
    )

    swap = get_kernel(kernmod, "swap", 'iPP')
    swap_Op = lambda *args: swap.prepared_call(
        grid_NbNeNv, block, args[-1].shape[0], *list(map(lambda c: c.ptr, args))
    )

    computeFlux = get_kernel(kernmod, "computeFlux", 'iPPP')
    computeFlux_Op = tuple(map(lambda v: 
        lambda *args: computeFlux.prepared_call(
            grid_KNeNv, block, KNeNv, v.ptr, *list(map(lambda c: c.ptr, args))
        ), (vm.d_cvx(), vm.d_cvy(), vm.d_cvz())
    ))

    jump = get_kernel(kernmod, "jump", 'iPPPPPPPPP')
    jump_Op = lambda d_uL, d_uR, d_jL: jump.prepared_call(
            grid_NqfTNeNv, block, NqfTNeNv,
            vm.d_cvx().ptr, vm.d_cvy().ptr, vm.d_cvz().ptr,
            d_mapL.ptr, d_mapR.ptr, d_nFace.ptr,
            d_uL.ptr, d_uR.ptr, d_jL.ptr)

    eDeriv = [get_kernel(kernmod, "eDeriv_{0}".format(d), 'iPPP') for d in range(dim)]
    dv = [vm.d_cvx(), vm.d_cvy(), vm.d_cvz()]
    def eDeriv_Op(idx, *args):
        eDeriv[idx].prepared_call(grid_NeNv, block, NeNv, dv[idx].ptr, 
                *list(map(lambda c: c.ptr, args)) )

    #eDeriv_Op = [lambda *args: get_kernel(kernmod, "eDeriv_{0}".format(d), 'iPPP').prepared_call(
    #                  grid_NeNv, block, NeNv, dv[d].ptr, *list(map(lambda c: c.ptr, args))
    #              ) for d in range(dim)]

    """
    # The boundary conditions (by default all boundaries are processor bnds)
    bc_types = ['dgfs-periodic']*
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

    # total flux kernel (sums up surface and volume terms)
    totalFlux = get_kernel(kernmod, "totalFlux", 'PPPP')
    totalFlux_Op = lambda d_ux, d_jL, d_jR: totalFlux.prepared_call(
            grid_Nv, block, d_ux.ptr, vm.d_cvx().ptr, d_jL.ptr, d_jR.ptr)

    # derivative op
    eDeriv = get_kernel(kernlimsmod, "eDeriv", 'PPP')
    eDeriv_Op = lambda d_u, d_ux: eDeriv.prepared_call(
            grid_Nv, block, vm.d_cvx().ptr, d_u.ptr, d_ux.ptr)
    """

    # \alpha AX + \beta Y kernel (for operations on coefficients)
    axnpbyCoeff = get_axnpby_kerns(2, range(K), NeNv, cfg.dtype)
    axnpbyCoeff_Op = lambda a0, x0, a1, x1: axnpbyCoeff.prepared_call(
                    grid_NqfTNeNv, block, x0.ptr, x1.ptr, a0, a1)

    # allocations on gpu
    d_usol = gpuarray.empty(NqNeNv, dtype=cfg.dtype)
    d_ux = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_usolF = gpuarray.empty(NqfTNeNv, dtype=cfg.dtype)
    d_uL, d_uR, d_jL = map(lambda _: gpuarray.empty_like(d_usolF), range(3))
    d_uBndL = [gpuarray.empty(Nb[i]*Nqf*Nv, dtype=cfg.dtype) for i in range(Nfaces)] 
    d_uBndR = [gpuarray.empty(Nb[i]*Nqf*Nv, dtype=cfg.dtype) for i in range(Nfaces)] 
    d_ucoeff = gpuarray.empty(KNeNv, dtype=cfg.dtype)
    d_ucoeffPrev = gpuarray.empty_like(d_ucoeff)
    d_temp = [ gpuarray.empty_like(d_ucoeff) for i in range(3)]

    # check if this is a new run
    if hasattr(args, 'process_run'):
        # load the initial condition model
        icn = cfg.lookup('soln-ics', 'type')
        initcondcls = subclass_where(DGFSInitConditionStd, model=icn)
        ic = initcondcls(cfg, vm, 'soln-ics')
        ic.apply_init_vals(d_usol, Nq, NeT, d_xsol, mesh=mesh, basis=basis, 
            psm=psm)

        # forward transform to coefficient space
        fwdTrans_Op(d_usol, d_ucoeff)

    #pex(gpuarray.sum(d_ucoeff)*vm.cw()/Nq/NeT)

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
    #pex(xsol[:,0,:])
    xcoeff = np.matmul(basis.fwdTransMat, xsol.reshape(Nq,-1)).reshape(K,NeT,dim)
    moments = DGFSMomWriterStd(ti, basis.interpMat, xcoeff, d_ucoeff, vm, cfg, 
        'dgfsmomwriter')
    #exit(0)

    # For computing residual
    residual = DGFSResidualStd(cfg, 'dgfsresidual')

    # For writing distribution function
    distribution = DGFSDistributionStd(ti, (K, Ne, Nv), cfg, 
        'dgfsdistwriter')
  
    # Actual algorithm

    # initialize
    axnpbyCoeff_Op(0., d_ucoeffPrev, 1., d_ucoeff)

    # define the neighbours
    #from mpi4py import MPI
    #down_nbr, up_nbr = comm.rank - 1, comm.rank + 1;
    #if up_nbr >= comm.size: up_nbr = MPI.PROC_NULL
    #if down_nbr < 0: down_nbr = MPI.PROC_NULL

    # prepare kernels for boundaries
    updateBC_Op, applyBC_Op = [0]*Nfaces, [0]*Nfaces
    for i in range(dim):
      for j, dir in enumerate(["lo", "hi"]):
        sec = 'soln-bcs-x%d-%s'%(i,dir)
        bc_type = cfg.lookup(sec, 'type')
        bc_cls = subclass_where(DGFSBCStd, type=bc_type)
        normal = [0,0,0]
        normal[i] = -1 if dir=="lo" else 1
        bc = bc_cls(d_bnd_x[2*i+j], normal, vm, cfg, sec, Ndof=Nb[2*i+j]*Nqf)
        updateBC_Op[2*i+j] = bc.updateBCKern
        applyBC_Op[2*i+j] = bc.applyBCKern
  
    #idx = 2
    #plt.scatter(bnd_x[idx][0,:], bnd_x[idx][1,:]) 
    #plt.savefig('plot.pdf')
    #exit(0)

    # step:1 
    # verify for advection
    #vm.d_cvx().fill(1); vm.d_cvy().fill(1); vm.d_cvz().fill(1)
    #u = np.sin(np.pi*xsol[:,:,0])*np.cos(np.pi*xsol[:,:,1])
    #d_ucoeff.set(np.stack([u]*Nv,2).ravel())
    #print(gpuarray.sum(d_ucoeff))

    def plot(name):
      d_u = d_ucoeff.get().reshape(Nq,NeT,-1)
      x, y = xsol[:,:,0], xsol[:,:,1]
      Nq1 = int(Nq**(1./dim))
      x, y, rho = map(lambda v: v.swapaxes(0,1).reshape(Ne[0],Ne[1],Nq1,Nq1).swapaxes(1,2).reshape(Nq1*Ne[0],Nq1*Ne[1]), (x, y, d_u[:,:,-1])) 
      plt.contourf(x,y, rho);
      plt.savefig('plot.pdf')

    #u = np.arange(Nq*NeT,dtype=cfg.dtype).reshape(Nq,NeT)
    #d_ucoeff.set(np.stack([u]*Nv,2).ravel())
    #pex(d_ucoeff.get().reshape(Nq,NeT))
    #print(xsol[:,:,0])

    #np.set_printoptions(formatter={'float_kind': "{:0.1f}".format})
 
    pp = lambda v: (filter_tol(v.get().reshape(NqfT,NeT)))
    pp1 = lambda v: (filter_tol(v.get().reshape(NqfT,NeT,-1)))
    pp2 = lambda v: pex((v.get().reshape(Nq,NeT)))
    pp3 = lambda v: filter_tol(v.get().reshape(Nq,NeT,-1))
    pp4 = lambda v: gpuarray.sum(v)
    pp21 = lambda v: v.get().reshape(Nq,NeT)
 
    # define the explicit part    
    def explicit(time, d_ucoeff_in, d_ucoeff_out, step=0):

        # reconstruct solution at faces
        bwdTransFace_Op(d_ucoeff_in, d_usolF)

        # Step:1 extract the positive and negative traces at faces
        extractTrace_Op(d_usolF, d_mapL, d_uL)
        extractTrace_Op(d_usolF, d_mapR, d_uR)
      
        #pex(pp(d_uR))

        #bwdTransFace_Op(gpuarray.to_gpu(xsol[:,:,0].ravel()), d_usolF)
        #pp(d_mapL)
        #pp2(d_ucoeff)
        #pex(pp3(d_xsol)[:,:,0])
        #pp(d_uL)
        #pex(d_usolF, d_mapL)
        #pex(pp2(d_ucoeff_in))
        #pex(d_bmapL[0])

        # transfer boundary information
        for i in range(Nfaces):
          extractBndTrace_Op(d_uL, d_bmapL[i], d_uBndL[i]) 
          #extractBndTrace_Op(d_uR, d_bmapR[i], d_uBndR[i]) 
          #print(d_uBndL[i].get().reshape(-1,Nb[i]))

        #exit(0)
        #pex(d_uBndL[0].get().reshape(Nb[0],-1))

        # enforce periodic boundary condition for now
        #for i in range(dim):
        #  cuda.memcpy_dtod(d_uBndR[2*i+1].ptr, d_uBndL[2*i].ptr, d_uBndL[2*i].nbytes)
        #  cuda.memcpy_dtod(d_uBndR[2*i].ptr, d_uBndL[2*i+1].ptr, d_uBndL[2*i+1].nbytes)

        #for i in range(Nfaces):
        #  print(d_uBndR[i].get().reshape(-1,Nb[i]))
        #pex()

        # apply boundary conditions to negative trace
        for i in range(Nfaces):
          updateBC_Op[i](d_uBndL[i], time) 
          applyBC_Op[i](d_uBndL[i], d_uBndR[i], time) 

        #for i in range(Nfaces):
        #  pi = np.pi
        #  ubc = np.sin((bnd_x[i][0,:]-time)*pi)*np.cos((bnd_x[i][1,:]-time)*pi)
        #  d_uBndR[i].set(np.stack([ubc]*Nv,1).ravel())

        # insert boundary information to negative trace
        for i in range(Nfaces):
            insertBndTrace_Op(d_uR, d_bmapL[i], d_uBndR[i])

        #pex(pp(d_uR))

        # Step:2 Compute the jump
        jump_Op(d_uL, d_uR, d_jL)
        #pex(d_nFace.get().reshape(NqfT,NeT,dim)[:,:,0])

        #pex(pp1(d_uR))
        #print(pp(d_uL))

        #print(pp(d_uR))
        #print(pp(d_jL))
        #print(pp(d_uR)-pp(d_uL))
        #pex()

        # Step:3 evaluate the cconvection term: -v \cdot grad(f) 
        axnpbyCoeff_Op(0., d_ux, 0., d_ux)
        for i in range(dim):
          #computeFlux_Op[i](d_ucoeff_in, d_temp[0])
          #deriv_Op[i](d_temp[0], d_temp[1]) # assuming cartesian
          
          axnpbyCoeff_Op(0., d_temp[1], 0., d_temp[1])
          eDeriv_Op(i, d_ucoeff_in, d_temp[1]) # assuming cartesian
          #pex(d_ucoeff_in.get(), d_temp[1].get())
          #print(pp21(d_temp[1]))

          axnpbyCoeff_Op(1., d_ux, invjac[i][0,0], d_temp[1]) # assuming cartesian

        #print(pp21(d_ucoeff_in))
        #print(pp21(d_ux))
        #exit(0)
        #pex(invjac)
        #pex(pp2(d_ucoeff))
        #pex(pp(d_jL))
        #pex(d_jL.get())
        #pex(pp3(d_ux))

        # Compute the continuous flux for each element in strong form
        lift_Op(d_jL, d_temp[0])
        axnpbyCoeff_Op(1., d_ux, -invjac[i][0,0], d_temp[0])
        #totalFlux_Op(d_ux, d_jL, d_jR)

        #pex(pp(d_jL))
        #pex(pp2(d_temp[0]))
        #pex(pp2(d_ux))
        #print(pp3(d_ux))
        #print(-pp3(d_temp[0])*invjac[0][0,0])
        #exit(0)

        #if step==1: pex(pp2(d_ucoeff))

        #for r, e in it.product(range(K), range(NeT)):
        #    sm.fs(d_ucoeff_in, d_ucoeff_in, d_ux, e, r, r)

        #pex(d_uL.get())
        #print(gpuarray.sum(d_ucoeff))
        #print(gpuarray.sum(d_ux))

        # Copy flux
        cuda.memcpy_dtod(d_ucoeff_out.ptr, d_ux.ptr, d_ux.nbytes)
        #exit(0)


    # define the explicit collision operator  
    def explicitQ(time, d_ucoeff_in, d_ucoeff_out, nu=None):

        axnpbyCoeff_Op(0., d_ucoeff_out, 0., d_ucoeff_out)
        if nu: axnpbyMomsCoeff_Op(0., nu, 0., nu)

        #for r, e in it.product(range(K), range(NeT)):
        #    sm.fs(d_ucoeff_in, d_ucoeff_in, d_ucoeff_out, e, r, r, d_nu=nu)

        #if nu: nu_max = gpuarray.max(nu); nu.set(np.ones(nu.shape)*nu_max.get());


    # define a time-integrator (we use Euler scheme: good enough for steady)
    odestype = cfg.lookup('time-integrator', 'scheme')
    odescls = subclass_where(DGFSIntegratorAstd, intg_kind=odestype)
    odes = odescls(explicit, psm, (K, NeT, Nv), cfg.dtype, explicitQ=explicitQ, limit1=None)

    # Finally start everything
    time = ti  # initialize time in case of restart
    nacptsteps = 0 # number of elasped steps in the current run

    # start timer
    start = timer()
    while(time < tf):

        # March in time 
        odes.integrate(time, dt, nacptsteps, d_ucoeff)
        #explicit(time, d_ucoeff, d_temp[2], nacptsteps)
        #axnpbyCoeff_Op(1., d_ucoeff, dt, d_temp[2])
        #print(pp3(d_ucoeff))

        #print(pp21(d_ucoeff))
        #if nacptsteps==2: exit(0)

        # increment time
        time += dt 
        nacptsteps += 1

        # Final step: post processing routines
        #u = (np.sin(np.pi*(xsol[:,:,0]-time))*np.cos(np.pi*(xsol[:,:,1]-time)))
        #if nacptsteps%10==0: 
        #    print("{0:2.2e} {1:2.4f}".format(time, np.linalg.norm(np.stack([u]*Nv,1).ravel()-d_ucoeff.get(), 2)))
        #d_ucoeffPrev.set(np.stack([u]*Nv,2).ravel())
        residual(time, nacptsteps, d_ucoeff, d_ucoeffPrev)
        moments(dt, time, d_ucoeff)
        #distribution(dt, time, d_ucoeff)

        # copy the solution for the next time step
        cuda.memcpy_dtod(d_ucoeffPrev.ptr, d_ucoeff.ptr, d_ucoeff.nbytes)

    #plot("abc")

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

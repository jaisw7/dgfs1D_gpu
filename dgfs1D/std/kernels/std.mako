#define scalar ${dtype}

// for extracting left face values
__global__ void extract_left
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t, mapVal in enumerate(mapL):
            out[${t*vsize} + idx] = in[${mapVal*vsize} + idx];
        % endfor
    }
}

// for extracting right face values
__global__ void extract_right
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t, mapVal in enumerate(mapR):
            out[${t*vsize} + idx] = in[${mapVal*vsize} + idx];
        % endfor
    }
}

// extract the boundary information at the left boundary
__global__ void transfer_bc_left
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        out[idx] = in[${offsetL*vsize}+idx];
    }
}

// extract the boundary information at the right boundary
__global__ void transfer_bc_right
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        out[idx] = in[${offsetR*vsize}+idx];
    }
}

// insert boundary condition at left
__global__ void insert_bc_left
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        out[${offsetL*vsize} + idx] = in[idx];
    }
}

// insert boundary condition at right
__global__ void insert_bc_right
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        out[${offsetR*vsize} + idx] = in[idx];
    }
}

// Multiply by the 
__global__ void mul_by_invjac
(
    //const scalar* invjac,
    scalar* ux
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${vsize}) 
    {
        % for k in range(K):
            % for e in range(Ne):

                id = ${(k*Ne+e)*vsize}+idx;
                ux[id] *= ${invjac[e,0]};

            %endfor 
        %endfor
    }
}


__global__ void flux
(
    const scalar* uL,
    const scalar* uR,
    const scalar* cvx,
    scalar* jL,
    scalar* jR
)
{
    // Compute the flux and jumps
    /*
        fL, fR = advx*uL, advx*uR
        fupw = 0.5*(fL + fR) + 0.5*np.abs(advx)*(uL - uR)
        jL = fupw - fL  # Compute the jump at left boundary
        jR = fupw - fR  # Compute the jump at right boundary

        # We fuse all the four operations into single call
        jL = 0.5*advx*(uR - uL) + 0.5*np.abs(advx)*(uL - uR)
        jR = 0.5*advx*(uL - uR) + 0.5*np.abs(advx)*(uL - uR)
    */

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize} + idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize} + idx] = jL[${t*vsize} + idx];
                
            jL[${t*vsize} + idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize} + idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
        % endfor
    }
}

__global__ void mul_by_adv
(
    const scalar* cvx,
    scalar* ux
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${K*Ne*vsize}) 
    {
        ux[idx] *= -cvx[idx%${vsize}];
    }
}

__global__ void totalFlux
(
    scalar* ux,
    const scalar* cvx,
    const scalar* jL,
    const scalar* jR
)
{
    // Compute the continuous flux for each element

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${vsize}) 
    {
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                ux[id] += -jL[${(e+1)*vsize}+idx]*${gRD[k]} 
                          +jR[${e*vsize}+idx]*${gLD[k]};
            %endfor 
        %endfor
    }
}



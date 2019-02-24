#define scalar ${dtype}

__global__ void applyBC
(
    const scalar* __restrict__ ul,
    scalar* __restrict__ ur,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ bnd_f0,
    const scalar* __restrict__ wall_nden, 
    const scalar time
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        scalar fac = (cvx[idx]-(${ux}))*(${nl});    
        ur[idx] = (fac<0)*(bnd_f0[idx]*wall_nden[0]) + (fac>=0)*(ul[idx]);
    }
}

__global__ void updateBC
(
    const scalar* __restrict__ ul,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ bnd_f0,
    scalar* __restrict__ bc_vals_num,
    scalar* __restrict__ bc_vals_den,
    const scalar time
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        scalar fac = (cvx[idx]-(${ux}))*(${nl});    

        bc_vals_num[idx] = (fac>=0.)*(fac*ul[idx]*${cw});
        bc_vals_den[idx] = (fac<0.)*(fac*bnd_f0[idx]*${cw});
    }    
}
#define scalar ${dtype}

__global__ void applyBC
(
    const scalar* __restrict__ ul,
    scalar* __restrict__ ur,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ bnd_f0,
    const scalar time
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        scalar fac = (cvx[idx]-(${u}))*(${nl});    
        ur[idx] = (fac<0)*(bnd_f0[idx]) + (fac>=0)*(ul[idx]);
    }
}

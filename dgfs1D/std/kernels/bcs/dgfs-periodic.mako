#define scalar ${dtype}

__global__ void applyBC
(
    const scalar* __restrict__ ul,
    scalar* __restrict__ ur,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ bnd_f0,
    const scalar wall_nden
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        ur[idx] = ul[idx];
    }
}
#define scalar ${dtype}
<% import numpy as np %>

__global__ void applyIC
(
    const int size,
    scalar* __restrict__ f,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ _x
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${','.join(['x'+str(i) for i in range(dim)])};
    
    if(idx<size) 
    {
            
            ${';'.join('x{0}=(_x[(idx/{1})*{2}+{0}]*{3})'.format(i,vsize,dim,H0) for i in range(dim))};
 
            f[idx] = ${rho}*exp(-(
                pow(cvx[idx%${vsize}]-${ux}, 2)
                +pow(cvy[idx%${vsize}]-${uy}, 2)
                +pow(cvz[idx%${vsize}]-${uz}, 2)
            )/${T})/pow(${np.pi}*${T}, 1.5);

    }
}


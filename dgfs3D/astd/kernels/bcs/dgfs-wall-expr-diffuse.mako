#define scalar ${dtype}
<% import numpy as np %>

__global__ void applyBC
(
    const int size,
    const scalar* __restrict__ _x,
    const scalar* __restrict__ ul,
    scalar* __restrict__ ur,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ bc_vals_num_sum,
    const scalar* __restrict__ bc_vals_den_sum,
    const scalar t
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${','.join(['x'+str(i) for i in range(dim)])};
    
    if(idx<size) 
    {
        ${';'.join('x{0}=(_x[(idx/{1})*{2}+{0}]*{3})'.format(i,vsize,dim,H0) for i in range(dim))};
 
        scalar fac = (cvx[idx%${vsize}]-(${ux}))*(${nl[0]})
                     + (cvy[idx%${vsize}]-(${uy}))*(${nl[1]}) 
                     + (cvz[idx%${vsize}]-(${uz}))*(${nl[2]});    
        
        scalar mxwl = exp(-(
                pow(cvx[idx%${vsize}]-${ux}, 2)
                +pow(cvy[idx%${vsize}]-${uy}, 2)
                +pow(cvz[idx%${vsize}]-${uz}, 2)
            )/${T})/pow(${np.pi}*${T}, 1.5);

        ur[idx] = (fac<0)*(mxwl*(-bc_vals_num_sum[idx/${vsize}]/bc_vals_den_sum[idx/${vsize}])) 
                  + (fac>=0)*(ul[idx]);
    }
}

__global__ void updateBC
(
    const int size,
    const scalar* __restrict__ _x,
    const scalar* __restrict__ ul,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    scalar* __restrict__ bc_vals_num,
    scalar* __restrict__ bc_vals_den,
    const scalar t
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${','.join(['x'+str(i) for i in range(dim)])};
    
    if(idx<size) 
    {
        ${';'.join('x{0}=(_x[(idx/{1})*{2}+{0}]*{3})'.format(i,vsize,dim,H0) for i in range(dim))};

        scalar fac = (cvx[idx%${vsize}]-(${ux}))*(${nl[0]})
                     + (cvy[idx%${vsize}]-(${uy}))*(${nl[1]}) 
                     + (cvz[idx%${vsize}]-(${uz}))*(${nl[2]});    
        
        scalar mxwl = exp(-(
                pow(cvx[idx%${vsize}]-${ux}, 2)
                +pow(cvy[idx%${vsize}]-${uy}, 2)
                +pow(cvz[idx%${vsize}]-${uz}, 2)
            )/${T})/pow(${np.pi}*${T}, 1.5);

        bc_vals_num[idx] = (fac>=0.)*(fac*ul[idx]*${cw});
        bc_vals_den[idx] = (fac<0.)*(fac*mxwl*${cw});
    }    
}


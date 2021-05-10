#define scalar ${dtype}

<% import numpy as np %>

__device__ int sign(scalar x)
{ 
    int t = x<0 ? -1 : 0;
    return x > 0 ? 1 : t;
}

__device__ scalar minmod(scalar a, scalar b, scalar c)
{
    int m = (sign(a)+sign(b)+sign(c))/3;
    return abs(m)==1 ? m*(min(min(fabs(a), fabs(b)), fabs(c))) : 0;
}

__global__ void limitLin
(
    const scalar* u, 
    const scalar* ulx, const scalar* uavg, scalar* ulim
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    bool limit;
    if(idx<${vsize}) 
    {
        % for e in [0, Ne-1]:
            % for k in range(K):
                ulim[${(k*Ne+e)*vsize}+idx] = u[${(k*Ne+e)*vsize}+idx];
            %endfor
        %endfor

        % for e in range(1,Ne-1):
            <% h = (1./invjac[e,0])*2 %>
            <% x0 = (xsol[0,e] + h/2) %>

            limit =
            (
                fabs(uavg[${(0*Ne+e)*vsize}+idx] 
                    - minmod(
                            (uavg[${(0*Ne+e)*vsize}+idx]
                                -u[${(0*Ne+e)*vsize}+idx]), 
                            (uavg[${(0*Ne+np.min([e+1,Ne-1]))*vsize}+idx] 
                                - uavg[${(0*Ne+e)*vsize}+idx]),
                            (uavg[${(0*Ne+e)*vsize}+idx]
                                - uavg[${(0*Ne+np.max([e-1,0]))*vsize}+idx])
                        )
                    - u[${(0*Ne+e)*vsize}+idx])
                +
                fabs(uavg[${(0*Ne+e)*vsize}+idx] 
                    + minmod(
                            (u[${((K-1)*Ne+e)*vsize}+idx]
                                -uavg[${(0*Ne+e)*vsize}+idx]), 
                            (uavg[${(0*Ne+np.min([e+1,Ne-1]))*vsize}+idx] 
                                - uavg[${(0*Ne+e)*vsize}+idx]),
                            (uavg[${(0*Ne+e)*vsize}+idx]
                                - uavg[${(0*Ne+np.max([e-1,0]))*vsize}+idx])
                        )
                    - u[${((K-1)*Ne+e)*vsize}+idx])
            ) > 1e-6 ? 1 : 0;

            % for k in range(K):
                ulim[${(k*Ne+e)*vsize}+idx] = (limit == 1 ? uavg[${(0*Ne+e)*vsize}+idx] 
                        + (${xsol[k,e]-x0}*
                            minmod(ulx[${(0*Ne+e)*vsize}+idx], 
                                (uavg[${(0*Ne+np.min([e+1,Ne-1]))*vsize}+idx] 
                                    - uavg[${(0*Ne+e)*vsize}+idx])/${h},
                                (uavg[${(0*Ne+e)*vsize}+idx]
                                    - uavg[${(0*Ne+np.max([e-1,0]))*vsize}+idx])/${h}
                            )
                        ) : u[${(k*Ne+e)*vsize}+idx]);
            %endfor 
        %endfor
    }
}


__global__ void limitLin1
(
    const scalar* u, 
    const scalar* ulx, const scalar* uavg, scalar* ulim
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    bool limit;
    if(idx<${nalph}) 
    {
        % for e in [0, Ne-1]:
            % for k in range(K):
                ulim[${(k*Ne+e)*nalph}+idx] = u[${(k*Ne+e)*nalph}+idx];
            %endfor
        %endfor

        % for e in range(1,Ne-1):
            <% h = (1./invjac[e,0])*2 %>
            <% x0 = (xsol[0,e] + h/2) %>

            limit =
            (
                fabs(uavg[${(0*Ne+e)*nalph}+idx] 
                    - minmod(
                            (uavg[${(0*Ne+e)*nalph}+idx]
                                -u[${(0*Ne+e)*nalph}+idx]), 
                            (uavg[${(0*Ne+np.min([e+1,Ne-1]))*nalph}+idx] 
                                - uavg[${(0*Ne+e)*nalph}+idx]),
                            (uavg[${(0*Ne+e)*nalph}+idx]
                                - uavg[${(0*Ne+np.max([e-1,0]))*nalph}+idx])
                        )
                    - u[${(0*Ne+e)*nalph}+idx])
                +
                fabs(uavg[${(0*Ne+e)*nalph}+idx] 
                    + minmod(
                            (u[${((K-1)*Ne+e)*nalph}+idx]
                                -uavg[${(0*Ne+e)*nalph}+idx]), 
                            (uavg[${(0*Ne+np.min([e+1,Ne-1]))*nalph}+idx] 
                                - uavg[${(0*Ne+e)*nalph}+idx]),
                            (uavg[${(0*Ne+e)*nalph}+idx]
                                - uavg[${(0*Ne+np.max([e-1,0]))*nalph}+idx])
                        )
                    - u[${((K-1)*Ne+e)*nalph}+idx])
            ) > 1e-4 ? 1 : 0;

            % for k in range(K):
                ulim[${(k*Ne+e)*nalph}+idx] = (limit == 1 ? uavg[${(0*Ne+e)*nalph}+idx] 
                        + (${xsol[k,e]-x0}*
                            minmod(ulx[${(0*Ne+e)*nalph}+idx], 
                                (uavg[${(0*Ne+np.min([e+1,Ne-1]))*nalph}+idx] 
                                    - uavg[${(0*Ne+e)*nalph}+idx])/${h},
                                (uavg[${(0*Ne+e)*nalph}+idx]
                                    - uavg[${(0*Ne+np.max([e-1,0]))*nalph}+idx])/${h}
                            )
                        ) : u[${(k*Ne+e)*nalph}+idx]);
            %endfor 
        %endfor
    }
}

// Multiply by the 
__global__ void mul_by_invjac1
(
    //const scalar* invjac,
    scalar* ux
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${nalph}) 
    {
        % for k in range(K):
            % for e in range(Ne):

                id = ${(k*Ne+e)*nalph}+idx;
                ux[id] *= ${invjac[e,0]};

            %endfor 
        %endfor
    }
}


// logarithmic mean
__device__ scalar aveL(scalar a, scalar b)
{
  #define VSMALL 1e-20
  scalar xi = fabs(a/(b+VSMALL));
  scalar f = (xi-1)/(xi+1);
  scalar u = f*f;
  return 0.5*(a+b)/(u<1e-2 ? (1 + u/3 + u*u/5 + u*u*u/7): log(xi+VSMALL)/2./(f+VSMALL));
}


__global__ void eDeriv
(
    const scalar* cvx,
    const scalar* u,
    scalar* ux
)
{
    // Compute the entropy derivative in each element
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${vsize}) 
    {
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                % for l in range(K):
                  ux[id] += -2*${Dr[k,l]}*cvx[idx]*aveL(u[id], u[${(l*Ne+e)*vsize}+idx]);
                %endfor
            %endfor 
        %endfor
    }
}


__global__ void limitPos
(
    const scalar* u, 
    const scalar* uavg, scalar* ulim
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    double fmin;
    if(idx<${vsize}) 
    {
        % for e in range(Ne):

            fmin = 1e10;
            % for k in range(K):
                fmin = min(fmin, u[${(k*Ne+e)*vsize}+idx]);
            %endfor

            fmin = min (1., 
                fabsf((uavg[${(0*Ne+e)*vsize}+idx] -1e-13)/
                (uavg[${(0*Ne+e)*vsize}+idx] -fmin))
            );
            
            % for k in range(K):
                ulim[${(k*Ne+e)*vsize}+idx] = 
                uavg[${(0*Ne+e)*vsize}+idx] + fmin*(u[${(k*Ne+e)*vsize}+idx] - uavg[${(0*Ne+e)*vsize}+idx]);
                //u[${(k*Ne+e)*vsize}+idx];
            %endfor 
        %endfor
    }
}

#define scalar ${dtype}

// for extractscalar*g trace values
__global__ void extract_trace
(
    int size,
    const scalar* faceValues,
    const scalar* __restrict__ traceMap, 
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) 
    {
        int t = idx/${vsize};
        out[idx] = faceValues[int(traceMap[t])*${vsize} + idx%${vsize}];
    }
}

// for scalar*sertscalar*g trace values
__global__ void insert_trace
(
    int size,
    scalar* faceValues,
    const scalar* traceMap, 
    const scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) 
    {
        int t = idx/${vsize};
        faceValues[int(traceMap[t])*${vsize} + idx%${vsize}] = out[idx] ;
    }
}

// for swappscalar*g
__global__ void swap
(
    int size,
    scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) 
    {
        scalar temp = in[idx];
        in[idx] = out[idx];
        out[idx] = temp;
    }
}

// Compute flux polynomial
__global__ void computeFlux
(
    const int size,
    const scalar* cv,
    const scalar* u,
    scalar* flux
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) 
    {
        flux[idx] = -cv[idx%${vsize}]*u[idx];
    }
}

/*
// Multiply by the metric jacobian 
__global__ void mulByInvJac
(
    const int size,
    scalar *invjac,
    scalar* ux
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<size) 
    {
       ux[idx] *= invjac[idx/${vsize}];
    }
}
*/

/*
__global__ void totalFlux
(
    const int size,
    scalar* ux,
    const scalar* jL,
    const scalar* jR
)
{
    // Compute the contscalar*uous flux for each element

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) 
    {
    }
}
*/


// logarithmic mean
__device__ scalar aveL(scalar a, scalar b)
{
  #define VSMALL 1e-20
  scalar xi = fabs(a/(b+VSMALL));
  scalar f = (xi-1)/(xi+1);
  scalar u = f*f;
  return 0.5*(a+b)/(u<1e-2 ? (1 + u/3 + u*u/5 + u*u*u/7): log(xi+VSMALL)/2./(f+VSMALL));
}

__global__ void jump
(
    const int size,
    const scalar* cvx,
    const scalar* cvy,
    const scalar* cvz,
    const scalar* mapL, 
    const scalar* mapR, 
    const scalar* norm,
    const scalar* uL,
    const scalar* uR,
    scalar* jL
)
{
    // Compute the flux and jumps

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) 
    {
        scalar fac = cvx[idx%${vsize}]*norm[int(mapL[idx/${vsize}])*${dim}+0];
        % if dim==2: 
          fac += cvy[idx%${vsize}]*norm[int(mapL[idx/${vsize}])*${dim}+1];
        %endif
        % if dim==3: 
          fac += cvz[idx%${vsize}]*norm[int(mapL[idx/${vsize}])*${dim}+2];
        %endif

        %if edg:
          jL[idx] = fac*(aveL(uL[idx],uR[idx])-uL[idx]) - 0.5*fabs(fac)*(uR[idx]-uL[idx]);
        %else:
          jL[idx] = (fac<0) ? (uR[idx] - uL[idx])*fac : 0;
        %endif
    }
}


%for d in range(dim):
<% DrT = Dr[d] %>
__global__ void eDeriv_${d}
(
    int size,
    const scalar* cvx,
    const scalar* u,
    scalar* ux
)
{
    // Compute the entropy derivative in each element
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<size) 
    {
        % for k in range(K):
          id = ${k}*size+idx;
          % for l in range(K):
            %if edg:
              ux[id] += -2*${DrT[k,l]}*cvx[idx%${vsize}]*aveL(u[${l}*size+idx], u[id]);
            %else:
              ux[id] += -(${DrT[k,l]})*cvx[idx%${vsize}]*u[${l}*size+idx];
            %endif
          %endfor 
        %endfor
    }
}
%endfor


#define scalar ${dtype}

<% import numpy as np %>


// Multiply by the 
__global__ void entropyPair
(
    const scalar* cvx,
    const scalar* f, 
    scalar* E, scalar *G
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${K*Ne*vsize}) 
    {
        E[idx] = f[idx]*(f[idx] < 0 ? 0: log(f[idx]))*${cw};
        //E[idx] = f[idx]*(f[idx] < 0 ? 0: 0.5*f[idx])*${cw};
        G[idx] = E[idx]*cvx[idx%${vsize}];
    }
}

// for extracting left face values
__global__ void extract_left
(
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<1) 
    {
        % for t, mapVal in enumerate(mapL):
            out[${t} + idx] = in[${mapVal} + idx];
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
    if(idx<1) 
    {
        % for t, mapVal in enumerate(mapR):
            out[${t} + idx] = in[${mapVal} + idx];
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
    if(idx<1) 
    {
        out[idx] = in[${offsetL}+idx];
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
    if(idx<1) 
    {
        out[idx] = in[${offsetR}+idx];
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
    if(idx<1) 
    {
        out[${offsetL} + idx] = in[idx];
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
    if(idx<1) 
    {
        out[${offsetR} + idx] = in[idx];
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

    if(idx<1) 
    {
        % for k in range(K):
            % for e in range(Ne):

                id = ${(k*Ne+e)}+idx;
                ux[id] *= ${invjac[e,0]};

            %endfor 
        %endfor
    }
}


__global__ void flux1
(
    const scalar* eL, const scalar* eR,
    const scalar* gL, const scalar* gR,
    scalar* jL, scalar* jR
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<1) 
    {
        % for t in range(len(mapL)):
            jL[${t} + idx] = 
                    0.5*${L}*(eL[${t}+idx] - eR[${t}+idx]);

            jR[${t} + idx] = 0.5*${L}*(eL[${t}+idx] - eR[${t}+idx]);
                
            jL[${t} + idx] += 
                0.5*(gR[${t}+idx] - gL[${t}+idx]);
                
            jR[${t} + idx] += 
                0.5*(gL[${t}+idx] - gR[${t}+idx]);
        % endfor
    }
}


__global__ void totalFlux
(
    scalar* ex,
    const scalar* jL, const scalar* jR
)
{
    // Compute the continuous flux for each element

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<1) 
    {
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)}+idx;
                ex[id] = -ex[id] + (-jL[${(e+1)}+idx]*${gRD[k]} 
                          +jR[${e}+idx]*${gLD[k]});
            %endfor 
        %endfor
    }
}

__global__ void entropyViscosity
(
    const scalar* d_e, 
    const scalar* d_eavg, 
    const scalar* d_eL, const scalar* d_eR, 
    scalar* d_R, 
    scalar* d_eps
)
{
    // Compute the continuous flux for each element

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int id;

    scalar scal = 0;

    if(idx<1) 
    {
        % for k in range(K):
            % for e in range(Ne):
                scal = max(scal, fabs(d_e[${k*Ne+e}] - d_eavg[${e}]));        
                //d_R[${e}] = max(fabs(d_R[${e}]), fabs(d_R[${k*Ne+e}]));
                d_R[${k*Ne+e}] = fabs(d_R[${k*Ne+e}]);
           %endfor 
        %endfor

        % for e in range(Ne):
            % for k in range(K):
            //d_eps[${k*Ne+e}] = min(${cmax}*${hk**2}*${L}, ${ce}*${(hk)**2}*d_R[${e}]/(scal+1e-15));
            %endfor
        %endfor 

        % for k in range(K):
          % for e in range(Ne):
            // ${cmax} 
            //d_R[${k*Ne+e}] = min(${cmax}*${hk**2}*${L}, ${ce}*${(hk**2)}*d_R[${k*Ne+e}]/(scal+1e-15));
            d_R[${k*Ne+e}] = min(${cmax}*${hk**2}*${L}, ${ce}*${(hk)**2}*max(d_R[${k*Ne+e}], fabs((d_eL[${e}]-d_eR[${e}])*${L}))/(scal+1e-15));
          %endfor 
        %endfor

        // now we apply smoothing
        % for e in range(Ne):
          % for k in range(K):
            % if k+e*K<Ne*K-1 and k+e*K>=1:
              d_eps[${k+e*K}] =  (d_R[${(k-1)+e*K}] +  2*d_R[${k+e*K}] + d_R[${(k+1)+e*K}])/4;
            %endif
            //d_eps[${k*Ne+e}] = d_R[${k*Ne+e}];
          %endfor 
        %endfor

        d_eps[${0*Ne+0}] =  (d_R[${(0)*Ne+0}] +  2*d_R[${(0)*Ne+0}] + d_R[${(1)*Ne+0}])/4;
        d_eps[${(K-1)*Ne+Ne-1}] =  (d_R[${(K-2)*Ne+Ne-1}] +  2*d_R[${(K-1)*Ne+Ne-1}] + d_R[${(K-1)*Ne+Ne-1}])/4;

        % for e in range(Ne):
            % for k in range(K):
                d_eps[${k*Ne+e}] *= ${H0};
            %endfor
        %endfor 
    }
}


__global__ void constructGrad
(
    scalar* d_fx,
    const scalar* d_eps
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
                d_fx[id] *= sqrt(d_eps[${k*Ne+e}]); 
            %endfor 
        %endfor
    }
}


__global__ void liftViscosity
(
    const scalar* d_eps,
    scalar* d_epsNp
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    if(idx<${Ne}) 
    {
        % for k in range(K):
           id = ${(k*Ne)}+idx;
           d_epsNp[id] = d_eps[idx]; 
        %endfor
    }
}


__global__ void fluxU_V0
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    scalar* jL, scalar* jR 
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

            jL[${t*vsize}+idx] = 
                -0.5*(sqrt(eR[${t}])*uR[${t*vsize}+idx] - sqrt(eL[${t}])*uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] = -0.5*(sqrt(eL[${t}])*uL[${t*vsize}+idx] - sqrt(eR[${t}])*uR[${t*vsize}+idx]); 
 
        % endfor

        // for the boundaries
        <% t = 0 %>
        jR[${0*vsize} + idx] =
                -0.5*(sqrt(eR[${t}])*uR[${t*vsize}+idx] - sqrt(eL[${t}])*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx]));

        // for the boundaries
        <% t = len(mapL)-1 %>
        jL[${t*vsize} + idx] =
                -0.5*(sqrt(eL[${t}])*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx]) - sqrt(eR[${t}])*uR[${t*vsize}+idx]);

    }
}

__global__ void fluxQ_V0
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    const scalar* qL, const scalar* qR,
    scalar* jL, scalar* jR 
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize}+idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize}+idx] = jL[${t*vsize}+idx];
                
            jL[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
 
            jR[${t*vsize}+idx] += 
                0.5*(sqrt(eR[${t}])*qR[${t*vsize}+idx] - sqrt(eL[${t}])*qL[${t*vsize}+idx]); 
            
            jL[${t*vsize}+idx] += 
                -0.5*(sqrt(eR[${t}])*qR[${t*vsize}+idx] - sqrt(eL[${t}])*qL[${t*vsize}+idx]); 
            
        % endfor

    }
}


__global__ void fluxU_v1
(
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    scalar* jL, scalar* jR 
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

            jL[${t*vsize}+idx] = 
                -0.5*sqrt(eL[${t}])*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] = 
                -0.5*sqrt(eR[${t}])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]); 
 
        % endfor

        // for the boundaries
        <% t=0 %> 
        jR[${t*vsize}+idx] = 
             -sqrt(eR[${t}])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]); 
 
        // for the boundaries
        <% t = len(mapL)-1 %>
        jL[${t*vsize}+idx] = 
                -sqrt(eL[${t}])*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
 
    }
}

__global__ void fluxQ_v1
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    const scalar* qL, const scalar* qR,
    scalar* jL, scalar* jR 
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize}+idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize}+idx] = jL[${t*vsize}+idx];
                
            jL[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
 
            % if t!=0:
              jR[${t*vsize}+idx] += 
                -0.5*sqrt(eR[${t}])*(qL[${t*vsize}+idx] - qR[${t*vsize}+idx]); 
            %endif 

            % if t!=len(mapL)-1:
              jL[${t*vsize}+idx] += 
                -0.5*sqrt(eL[${t}])*(qR[${t*vsize}+idx] - qL[${t*vsize}+idx]); 
            %endif 

        % endfor

    }
}


__global__ void fluxU_v2
(
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    scalar* jL, scalar* jR 
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

            jL[${t*vsize}+idx] = 0;
                
            //jR[${t*vsize}+idx] = 
            //    -sqrt(eR[${t}])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]); 
 
        % endfor

        // for the boundaries
        <% t=0 %> 
        jR[${t*vsize}+idx] = 
             -2*sqrt(eR[${t}])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]); 
 
        // for the boundaries
        <% t = len(mapL)-1 %>
        //jL[${t*vsize}+idx] = 
        //        -sqrt(eL[${t}])*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
 
    }
}

__global__ void fluxQ_v2
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    const scalar* qL, const scalar* qR,
    scalar* jL, scalar* jR 
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize}+idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize}+idx] = jL[${t*vsize}+idx];
                
            jL[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
 
            jL[${t*vsize}+idx] += 
                -sqrt(eL[${t}])*(qR[${t*vsize}+idx] - qL[${t*vsize}+idx]); 

        % endfor

    }
}




__global__ void fluxU_v8
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    scalar* jL, scalar* jR 
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

            jL[${t*vsize}+idx] = 
                    -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*uR[${t*vsize}+idx]
                        - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
                    
            jR[${t*vsize}+idx] = 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*uR[${t*vsize}+idx]
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );
        % endfor

        // for the boundaries
        <% t=0 %> 
        jR[${t*vsize}+idx] = 
                - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uL[${t*vsize}+idx]+2*uR[${t*vsize}+idx])
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] ); 
 
        // for the boundaries
        <% t = len(mapL)-1 %>
        jL[${t*vsize}+idx] = 
                - ( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uL[${t*vsize}+idx] + 2*uR[${t*vsize}+idx])
                    - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
 
    }
}

__global__ void fluxQ_v8
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    const scalar* qL, const scalar* qR,
    scalar* jL, scalar* jR 
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize}+idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize}+idx] = jL[${t*vsize}+idx];
                
            jL[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
 
            
            jR[${t*vsize}+idx] += 
                    -(sqrt(eL[${t}])*qL[${t*vsize}+idx] - sqrt(eR[${t}])*qR[${t*vsize}+idx]); 

            jL[${t*vsize}+idx] += 
                    -(sqrt(eR[${t}])*qR[${t*vsize}+idx] - sqrt(eL[${t}])*qL[${t*vsize}+idx]); 

        % endfor

    }
}



__global__ void fluxU
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    scalar* jL, scalar* jR 
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

            // For positive velocities, we use alternating left-right
            // uh = \sqrt{e^-} u^-; qh = 0.5*(\sqrt{e^-}+\sqrt{e^+}) q^+; 
            // For negative velocities, we use alternating right-left
            // qh = \sqrt{e^-} q^-; uh = 0.5*(\sqrt{e^-}+\sqrt{e^+}) u^+; 
            if(cvx[idx]==0) 
            {
                jL[${t*vsize}+idx] = 
                    -( sqrt(eL[${t}])*uL[${t*vsize}+idx]
                        - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
                    
                jR[${t*vsize}+idx] = 
                    - ( sqrt(eL[${t}])*uL[${t*vsize}+idx]
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );                 
            } 
            else
            {
                jL[${t*vsize}+idx] = 
                    -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*uR[${t*vsize}+idx]
                        - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
                    
                jR[${t*vsize}+idx] = 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*uR[${t*vsize}+idx]
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );  
            }
 
        % endfor

        // for the boundaries
        if(cvx[idx]==0) 
        {
            <% t = len(mapL)-1 %>
            jL[${t*vsize}+idx] = 
                    -( sqrt(eL[${t}])*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                        - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
                    
            <% t=0 %> 
            jR[${t*vsize}+idx] = 
                    - ( sqrt(eL[${t}])*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );                 
        } 
        else
        {
            <% t = len(mapL)-1 %>
            jL[${t*vsize}+idx] = 
                    -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                        - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
            //jL[${t*vsize}+idx] = sqrt(eL[${t}])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);            
            //jL[${t*vsize}+idx] = 
            //        -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(uR[${t*vsize}+idx])
            //            - sqrt(eL[${t}])*uL[${t*vsize}+idx] );

            #if 0
            if(cvx[idx]<0)
                jL[${t*vsize}+idx] = 
                        -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                            - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
            else
                jL[${t*vsize}+idx] = 
                        -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                            - sqrt(eL[${t}])*uL[${t*vsize}+idx] );
            #endif                                
                    
            <% t=0 %> 
            jR[${t*vsize}+idx] = 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uR[${t*vsize}+idx]+2*uL[${t*vsize}+idx])
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );  
            //jR[${t*vsize}+idx] = sqrt(eR[${t}])*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
            //jR[${t*vsize}+idx] = 
            //        - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(uR[${t*vsize}+idx])
            //             - sqrt(eR[${t}])*uR[${t*vsize}+idx] );  

            #if 0
            if(cvx[idx]<0)
                jR[${t*vsize}+idx] = 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uL[${t*vsize}+idx]+2*uR[${t*vsize}+idx])
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );
            else
                jR[${t*vsize}+idx] = 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*(-uL[${t*vsize}+idx]+2*uR[${t*vsize}+idx])
                         - sqrt(eR[${t}])*uR[${t*vsize}+idx] );
            #endif
        } 
    }
}

__global__ void fluxQ
(
    const scalar* cvx,
    const scalar* uL, const scalar* uR,
    const scalar* eL, const scalar* eR,
    const scalar* qL, const scalar* qR,
    scalar* jL, scalar* jR 
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        % for t in range(len(mapL)):
            jL[${t*vsize}+idx] = 
                    0.5*fabs(cvx[idx])*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);

            jR[${t*vsize}+idx] = jL[${t*vsize}+idx];
                
            jL[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);
                
            jR[${t*vsize}+idx] += 
                0.5*cvx[idx]*(uL[${t*vsize}+idx] - uR[${t*vsize}+idx]);
 
            
            if(cvx[idx]==0) 
            {
                jL[${t*vsize}+idx] += 
                    -( 0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*qR[${t*vsize}+idx]
                        - sqrt(eL[${t}])*qL[${t*vsize}+idx] );

                jR[${t*vsize}+idx] += 
                    - (0.5*(sqrt(eR[${t}])+sqrt(eL[${t}]))*qR[${t*vsize}+idx]
                         - sqrt(eR[${t}])*qR[${t*vsize}+idx] );
            }
            else
            {
                jL[${t*vsize}+idx] += 
                    -(sqrt(eL[${t}])*qL[${t*vsize}+idx] - sqrt(eL[${t}])*qL[${t*vsize}+idx]);    

                jR[${t*vsize}+idx] += 
                    -(sqrt(eR[${t}])*qL[${t*vsize}+idx] - sqrt(eR[${t}])*qR[${t*vsize}+idx]); 
            }

        % endfor

    }
}


#define scalar ${dtype}

<% import math %>

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


// Construct maxwellian
__global__ void cmaxwellian
(
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* M, const scalar* moms, scalar* collFreq
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar T;

    if(idx<${vsize}) 
    {
        % for k in range(Nq):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                T = (
                        moms[${(k*Ne+e)*nalph}+4]
                        + (
                        - moms[${(k*Ne+e)*nalph}+1]*moms[${(k*Ne+e)*nalph}+1]
                        - moms[${(k*Ne+e)*nalph}+2]*moms[${(k*Ne+e)*nalph}+2]
                        - moms[${(k*Ne+e)*nalph}+3]*moms[${(k*Ne+e)*nalph}+3]
                        )/moms[${(k*Ne+e)*nalph}+0]
                    )/(1.5*moms[${(k*Ne+e)*nalph}+0]);

                M[id] = moms[${(k*Ne+e)*nalph}+0]/pow(${math.pi}*T, 1.5)
                    *exp(
                        -(
                            (cx[idx]-moms[${(k*Ne+e)*nalph}+1]/moms[${(k*Ne+e)*nalph}+0])
                            *(cx[idx]-moms[${(k*Ne+e)*nalph}+1]/moms[${(k*Ne+e)*nalph}+0])
                          + (cy[idx]-moms[${(k*Ne+e)*nalph}+2]/moms[${(k*Ne+e)*nalph}+0])
                            *(cy[idx]-moms[${(k*Ne+e)*nalph}+2]/moms[${(k*Ne+e)*nalph}+0])
                          + (cz[idx]-moms[${(k*Ne+e)*nalph}+3]/moms[${(k*Ne+e)*nalph}+0])
                            *(cz[idx]-moms[${(k*Ne+e)*nalph}+3]/moms[${(k*Ne+e)*nalph}+0])
                        )/T
                    ); 

                if(idx==0)
                {
                    // Compute the collision frequency
                    //collFreq[${k*Ne+e}] = 
                    //    1e10*moms[${(k*Ne+e)*nalph}+0]*pow(T, ${1-omega});
                    //collFreq[${k*Ne+e}] = moms[0];
                    //collFreq[${k*Ne+e}] = 1e5;
                    //collFreq[${k*Ne+e}] = ${4.0*math.pi}*moms[${(k*Ne+e)*nalph}+0]*10;
                    //collFreq[${k*Ne+e}] = ${4.0*math.pi*10.};
                    //collFreq[${k*Ne+e}] = 30; //97.5;
                    collFreq[${k*Ne+e}] = 
                        moms[${(k*Ne+e)*nalph}+0]*pow(T, ${1-omega});
                }
            %endfor 
        %endfor
    }
}

// Construct maxwellian
__global__ void momentsESNorm
(
    scalar* moms_f
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar T;

    if(idx<${Nq*Ne}) 
    {   
        moms_f[idx*${nalphES}+0] = moms_f[idx*${nalphES}+0]; // density
        moms_f[idx*${nalphES}+1] /= moms_f[idx*${nalphES}+0]; // x-velocity
        moms_f[idx*${nalphES}+2] /= moms_f[idx*${nalphES}+0]; // y-velocity
        moms_f[idx*${nalphES}+3] /= moms_f[idx*${nalphES}+0]; // z-velocity

        // T = 1/1.5/rho*(mom4 + mom5 + mom6) - (u*u+v*v+w*w)/1.5
        T = (
                moms_f[idx*${nalphES}+4] 
                + moms_f[idx*${nalphES}+5] 
                + moms_f[idx*${nalphES}+6]
            )/(1.5*moms_f[idx*${nalphES}+0])
          - 
            (
                moms_f[idx*${nalphES}+1]*moms_f[idx*${nalphES}+1]
                + moms_f[idx*${nalphES}+2]*moms_f[idx*${nalphES}+2]
                + moms_f[idx*${nalphES}+3]*moms_f[idx*${nalphES}+3]
            )/1.5;
        
        // Non-dimensionalization factors
        <% nF = 2.0 %> 
        <% nF2 = 1.0 %>

        // P_yx
        moms_f[idx*${nalphES}+7] = ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+7]/moms_f[idx*${nalphES}+0]
                        - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+1]*moms_f[idx*${nalphES}+2];

        // P_yz
        moms_f[idx*${nalphES}+8] = ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+8]/moms_f[idx*${nalphES}+0]
                        - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+2]*moms_f[idx*${nalphES}+3];

        // P_zx
        moms_f[idx*${nalphES}+9] = ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+9]/moms_f[idx*${nalphES}+0]
                        - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+3]*moms_f[idx*${nalphES}+1];

        // P_xx
        moms_f[idx*${nalphES}+4] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+4]/moms_f[idx*${nalphES}+0]
                - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+1]*moms_f[idx*${nalphES}+1];

        // P_yy
        moms_f[idx*${nalphES}+5] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+5]/moms_f[idx*${nalphES}+0]
                - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+2]*moms_f[idx*${nalphES}+2];

        // P_zz
        moms_f[idx*${nalphES}+6] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+6]/moms_f[idx*${nalphES}+0]
                - ${nF}*${1.-1./Pr}*moms_f[idx*${nalphES}+3]*moms_f[idx*${nalphES}+3];

        // Temperature
        moms_f[idx*${nalphES}+10] = T;
    }
}

// Construct maxwellian
__global__ void cmaxwellianES
(
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* Mes, const scalar* moms_f, scalar* collFreq
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar detT;

    if(idx<${vsize}) 
    {
        % for k in range(Nq):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                // Symmetric matrix
                detT = (-moms_f[${(k*Ne+e)*nalphES}+6]*moms_f[${(k*Ne+e)*nalphES}+7]*moms_f[${(k*Ne+e)*nalphES}+7]
                    + 2*moms_f[${(k*Ne+e)*nalphES}+7]*moms_f[${(k*Ne+e)*nalphES}+8]*moms_f[${(k*Ne+e)*nalphES}+9] 
                    - moms_f[${(k*Ne+e)*nalphES}+4]*moms_f[${(k*Ne+e)*nalphES}+8]*moms_f[${(k*Ne+e)*nalphES}+8] 
                    - moms_f[${(k*Ne+e)*nalphES}+5]*moms_f[${(k*Ne+e)*nalphES}+9]*moms_f[${(k*Ne+e)*nalphES}+9] 
                    + moms_f[${(k*Ne+e)*nalphES}+4]*moms_f[${(k*Ne+e)*nalphES}+5]*moms_f[${(k*Ne+e)*nalphES}+6]);

                Mes[id] = 
                moms_f[${(k*Ne+e)*nalphES}+0]/(${(math.pi)**1.5}*sqrt(detT))*exp(
                    -(
                        + (moms_f[${(k*Ne+e)*nalphES}+5]*moms_f[${(k*Ne+e)*nalphES}+6] - moms_f[${(k*Ne+e)*nalphES}+8]*moms_f[${(k*Ne+e)*nalphES}+8])*((cx[idx]-moms_f[${(k*Ne+e)*nalphES}+1])*(cx[idx]-moms_f[${(k*Ne+e)*nalphES}+1]))
                        + (moms_f[${(k*Ne+e)*nalphES}+4]*moms_f[${(k*Ne+e)*nalphES}+6] - moms_f[${(k*Ne+e)*nalphES}+9]*moms_f[${(k*Ne+e)*nalphES}+9])*((cy[idx]-moms_f[${(k*Ne+e)*nalphES}+2])*(cy[idx]-moms_f[${(k*Ne+e)*nalphES}+2]))
                        + (moms_f[${(k*Ne+e)*nalphES}+4]*moms_f[${(k*Ne+e)*nalphES}+5] - moms_f[${(k*Ne+e)*nalphES}+7]*moms_f[${(k*Ne+e)*nalphES}+7])*((cz[idx]-moms_f[${(k*Ne+e)*nalphES}+3])*(cz[idx]-moms_f[${(k*Ne+e)*nalphES}+3]))
                        + 2*(moms_f[${(k*Ne+e)*nalphES}+8]*moms_f[${(k*Ne+e)*nalphES}+9] - moms_f[${(k*Ne+e)*nalphES}+6]*moms_f[${(k*Ne+e)*nalphES}+7])*((cx[idx]-moms_f[${(k*Ne+e)*nalphES}+1])*(cy[idx]-moms_f[${(k*Ne+e)*nalphES}+2]))
                        + 2*(moms_f[${(k*Ne+e)*nalphES}+7]*moms_f[${(k*Ne+e)*nalphES}+9] - moms_f[${(k*Ne+e)*nalphES}+4]*moms_f[${(k*Ne+e)*nalphES}+8])*((cy[idx]-moms_f[${(k*Ne+e)*nalphES}+2])*(cz[idx]-moms_f[${(k*Ne+e)*nalphES}+3]))
                        + 2*(moms_f[${(k*Ne+e)*nalphES}+7]*moms_f[${(k*Ne+e)*nalphES}+8] - moms_f[${(k*Ne+e)*nalphES}+9]*moms_f[${(k*Ne+e)*nalphES}+5])*((cz[idx]-moms_f[${(k*Ne+e)*nalphES}+3])*(cx[idx]-moms_f[${(k*Ne+e)*nalphES}+1]))
                    )/(detT)
                );

                if(idx==0)
                {
                    // Compute the collision frequency
                    //collFreq[${k*Ne+e}] = 
                    //  moms_f[${(k*Ne+e)*nalphES}+0]/(${(2*math.pi)**1.5}*sqrt(detT));
                    collFreq[${k*Ne+e}] = 
                        moms_f[${(k*Ne+e)*nalphES}+0]*pow
                        (
                            moms_f[${(k*Ne+e)*nalphES}+10], ${1-omega}
                        );
                }
            %endfor 
        %endfor
    }
}

// Construct maxwellian
__global__ void momentsSKNorm
(
    scalar* moms_f
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${Nq*Ne}) 
    {   
        moms_f[idx*${nalphSK}+0] = moms_f[idx*${nalphSK}+0]; // density
        moms_f[idx*${nalphSK}+1] /= moms_f[idx*${nalphSK}+0]; // x-velocity
        moms_f[idx*${nalphSK}+2] /= moms_f[idx*${nalphSK}+0]; // y-velocity
        moms_f[idx*${nalphSK}+3] /= moms_f[idx*${nalphSK}+0]; // z-velocity

        // T = 1/1.5/rho*mom4 - (u*u+v*v+w*w)/1.5
        moms_f[idx*${nalphSK}+10] = (
                moms_f[idx*${nalphSK}+10]/moms_f[idx*${nalphSK}+0]
                - (
                    moms_f[idx*${nalphSK}+1]*moms_f[idx*${nalphSK}+1]
                    + moms_f[idx*${nalphSK}+2]*moms_f[idx*${nalphSK}+2]
                    + moms_f[idx*${nalphSK}+3]*moms_f[idx*${nalphSK}+3]
                )
            )/1.5;
        
        // Non-dimensionalization factors
        % for t, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            moms_f[idx*${nalphSK}+${t}+11] += 
                - 2*(
                        moms_f[idx*${nalphSK}+4+${q1}]*moms_f[idx*${nalphSK}+1]
                      + moms_f[idx*${nalphSK}+4+${q2}]*moms_f[idx*${nalphSK}+2]
                      + moms_f[idx*${nalphSK}+4+${q3}]*moms_f[idx*${nalphSK}+3]
                       
                    )
                + moms_f[idx*${nalphSK}+0]*moms_f[idx*${nalphSK}+1+${t}]*(
                    moms_f[idx*${nalphSK}+1]*moms_f[idx*${nalphSK}+1]
                    + moms_f[idx*${nalphSK}+2]*moms_f[idx*${nalphSK}+2]
                    + moms_f[idx*${nalphSK}+3]*moms_f[idx*${nalphSK}+3]
                )
                - 1.5*moms_f[idx*${nalphSK}+1+${t}]*moms_f[idx*${nalphSK}+0]*moms_f[idx*${nalphSK}+10];
        %endfor
    }
}

// Construct maxwellian
__global__ void cmaxwellianSK
(
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* M, const scalar* moms, scalar* collFreq
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${vsize}) 
    {
        % for k in range(Nq):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;

                M[id] = moms[${(k*Ne+e)*nalphSK}+0]/pow(${math.pi}*moms[${(k*Ne+e)*nalphSK}+10], 1.5)
                    *exp(
                        -(
                            (cx[idx]-moms[${(k*Ne+e)*nalphSK}+1])
                            *(cx[idx]-moms[${(k*Ne+e)*nalphSK}+1])
                          + (cy[idx]-moms[${(k*Ne+e)*nalphSK}+2])
                            *(cy[idx]-moms[${(k*Ne+e)*nalphSK}+2])
                          + (cz[idx]-moms[${(k*Ne+e)*nalphSK}+3])
                            *(cz[idx]-moms[${(k*Ne+e)*nalphSK}+3])
                        )/moms[${(k*Ne+e)*nalphSK}+10]
                    ); 

                M[id] *= (
                    1. 
                    + ${2*(1-Pr)/5.}*(
                            moms[${(k*Ne+e)*nalphSK}+11]*(cx[idx]-moms[${(k*Ne+e)*nalphSK}+1])
                          + moms[${(k*Ne+e)*nalphSK}+12]*(cy[idx]-moms[${(k*Ne+e)*nalphSK}+2])
                          + moms[${(k*Ne+e)*nalphSK}+13]*(cz[idx]-moms[${(k*Ne+e)*nalphSK}+3])
                        )/(moms[${(k*Ne+e)*nalphSK}+0]*moms[${(k*Ne+e)*nalphSK}+10]*moms[${(k*Ne+e)*nalphSK}+10])
                        *(
                            2*(
                                (cx[idx]-moms[${(k*Ne+e)*nalphSK}+1])
                                *(cx[idx]-moms[${(k*Ne+e)*nalphSK}+1])
                              + (cy[idx]-moms[${(k*Ne+e)*nalphSK}+2])
                                *(cy[idx]-moms[${(k*Ne+e)*nalphSK}+2])
                              + (cz[idx]-moms[${(k*Ne+e)*nalphSK}+3])
                                *(cz[idx]-moms[${(k*Ne+e)*nalphSK}+3])
                            )/moms[${(k*Ne+e)*nalphSK}+10]
                            - 5
                        )
                );

                if(idx==0)
                {
                    collFreq[${k*Ne+e}] = 
                        moms[${(k*Ne+e)*nalphSK}+0]*pow(moms[idx*${nalphSK}+10], ${1-omega});
                }
            %endfor 
        %endfor
    }
}


%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"update_ARS_Mom_stage{0}".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* moms_f, 
    ${" ".join(["const scalar* moms_L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_F{0},".format(i) for i in range(1,ar-1)])}
    ${"scalar *moms_F{0}".format(ar-1)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${Nq*Ne}) 
    {   
        // density, x-velocity, y-velocity, z-velocity, temperature
        %for t in [0, 1, 2, 3, 4]:
            moms_F${ar-1}[idx*${nalph}+${t}] = moms_f[idx*${nalph}+${t}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, t) for i in range(1, ar)
                    ])};    
        %endfor
        
    }
}
%endfor


%for ar in nars:
// compute the moment of the esbgk kernel
__global__ void ${"update_ARS_MomES_stage{0}".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* moms_f, 
    ${" ".join(["const scalar* moms_L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_F{0},".format(i) for i in range(1,ar-1)])}
    ${"scalar *moms_F{0}".format(ar-1)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar)])};

    <% epsilon = 1./prefacESBGK %>

    if(idx<${Nq*Ne}) 
    {   
        // density, x-velocity, y-velocity, z-velocity, temperature
        %for t in [0, 1, 2, 3, 10]:
            moms_F${ar-1}[idx*${nalphES}+${t}] = moms_f[idx*${nalphES}+${t}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}]".format(
                            ar, i, nalphES, t) for i in range(1, ar)
                    ])};    
        %endfor

        // T = 1/1.5/rho*(mom10) - (u*u+v*v+w*w)/1.5
        % for i in range(1, ar):
            T${i} = moms_F${i}[idx*${nalphES}+10]/(1.5*moms_F${i}[idx*${nalphES}+0])
            - 
            (
                moms_F${i}[idx*${nalphES}+1]*moms_F${i}[idx*${nalphES}+1]
                + moms_F${i}[idx*${nalphES}+2]*moms_F${i}[idx*${nalphES}+2]
                + moms_F${i}[idx*${nalphES}+3]*moms_F${i}[idx*${nalphES}+3]
            )/(1.5*pow(moms_F${i}[idx*${nalphES}+0], 2.)); 
        %endfor

        // Non-dimensionalization factors
        <% nF = 1. %> 
        <% nF2 = 0.5 %>

        % for t, (ii, jj) in enumerate(['00', '11', '22', '01', '12', '20']):
            <% i, j = map(int, (ii, jj)) %>
            moms_F${ar-1}[idx*${nalphES}+${t}+4] = (
                moms_f[idx*${nalphES}+${t}+4]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphES, t) for _i in range(1, ar)])}
                % for _i in range(1, ar-1):
                  + dt*a${ar}${_i+1}*${1./(Pr*epsilon)}*moms_F${_i}[idx*${nalphES}+0]*pow(T${_i}, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_F${_i}[idx*${nalphES}+0]*T${_i}
                    %endif
                    + ${nF}*moms_F${_i}[idx*${nalphES}+1+${i}]*moms_F${_i}[idx*${nalphES}+1+${j}]
                        /moms_F${_i}[idx*${nalphES}+0]
                    - ${nF}*moms_F${_i}[idx*${nalphES}+${t}+4]
                  )
                %endfor
                + dt*a${ar}${ar}*${1./(Pr*epsilon)}*moms_F${ar-1}[idx*${nalphES}+0]*pow(T${ar-1}, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_F${ar-1}[idx*${nalphES}+0]*T${ar-1}
                    %endif
                    + ${nF}*moms_F${ar-1}[idx*${nalphES}+1+${i}]*moms_F${ar-1}[idx*${nalphES}+1+${j}]
                        /moms_F${ar-1}[idx*${nalphES}+0]
                  )
                )/(
                    1. + dt*a${ar}${ar}*${1./(Pr*epsilon)}*moms_F${ar-1}[idx*${nalphES}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor
    }
}
%endfor


%for ar in nars:
// compute the moment of the esbgk kernel
__global__ void ${"update_ARS_MomSK_stage{0}".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* moms_f, 
    ${" ".join(["const scalar* moms_L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_F{0},".format(i) for i in range(1,ar-1)])}
    ${"scalar *moms_F{0}".format(ar-1)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar)])};

    <% epsilon = 1./prefacSK %>

    if(idx<${Nq*Ne}) 
    {   
        // density, x-velocity, y-velocity, z-velocity, temperature
        %for t in [0, 1, 2, 3, 10]:
            moms_F${ar-1}[idx*${nalphSK}+${t}] = moms_f[idx*${nalphSK}+${t}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}]".format(
                            ar, i, nalphSK, t) for i in range(1, ar)
                    ])};    
        %endfor

        // T = 1/1.5/rho*(mom10) - (u*u+v*v+w*w)/1.5
        % for i in range(1, ar):
            T${i} = moms_F${i}[idx*${nalphSK}+10]/(1.5*moms_F${i}[idx*${nalphSK}+0])
            - 
            (
                moms_F${i}[idx*${nalphSK}+1]*moms_F${i}[idx*${nalphSK}+1]
                + moms_F${i}[idx*${nalphSK}+2]*moms_F${i}[idx*${nalphSK}+2]
                + moms_F${i}[idx*${nalphSK}+3]*moms_F${i}[idx*${nalphSK}+3]
            )/(1.5*pow(moms_F${i}[idx*${nalphSK}+0], 2.)); 
        %endfor

        // Evolve stress
        % for t, (ii, jj) in enumerate(['00', '11', '22', '01', '12', '20']):
            <% i, j = map(int, (ii, jj)) %>
            moms_F${ar-1}[idx*${nalphSK}+${t}+4] = (
                moms_f[idx*${nalphSK}+${t}+4]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                % for _i in range(1, ar-1):
                  + dt*a${ar}${_i+1}*${1./(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_F${_i}[idx*${nalphSK}+0]*T${_i}
                    %endif
                    + moms_F${_i}[idx*${nalphSK}+1+${i}]*moms_F${_i}[idx*${nalphSK}+1+${j}]
                        /moms_F${_i}[idx*${nalphSK}+0]
                    - moms_F${_i}[idx*${nalphSK}+${t}+4]
                  )
                %endfor
                + dt*a${ar}${ar}*${1./(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_F${ar-1}[idx*${nalphSK}+0]*T${ar-1}
                    %endif
                    + moms_F${ar-1}[idx*${nalphSK}+1+${i}]*moms_F${ar-1}[idx*${nalphSK}+1+${j}]
                        /moms_F${ar-1}[idx*${nalphSK}+0]
                  )
                )/(
                    1. + dt*a${ar}${ar}*${1./(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor

        // Evolve heat flux
        % for t, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            moms_F${ar-1}[idx*${nalphSK}+${t}+11] = (
                moms_f[idx*${nalphSK}+${t}+11]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+11]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                % for _i in range(1, ar):
                  + dt*a${ar}${_i+1}*${(1-Pr)/(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    -2.*(
                        moms_F${_i}[idx*${nalphSK}+4+${q1}]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+4+${q2}]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+4+${q3}]*moms_F${_i}[idx*${nalphSK}+3]
                    )/moms_F${_i}[idx*${nalphSK}+0]
                    + moms_F${_i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${_i}[idx*${nalphSK}+1]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+2]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+3]*moms_F${_i}[idx*${nalphSK}+3]
                    )/pow(moms_F${_i}[idx*${nalphSK}+0], 2.)
                    - 1.5*moms_F${_i}[idx*${nalphSK}+1+${t}]*T${_i}
                  )
                  + dt*a${ar}${_i+1}*${1./(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    + moms_F${_i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${_i}[idx*${nalphSK}+1]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+2]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+3]*moms_F${_i}[idx*${nalphSK}+3]
                    )/pow(moms_F${_i}[idx*${nalphSK}+0], 2.)
                    + 2.5*moms_F${_i}[idx*${nalphSK}+1+${t}]*T${_i}
                  )
                %endfor
                % for _i in range(1, ar-1):
                  - dt*a${ar}${_i+1}*${(Pr)/(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    moms_F${_i}[idx*${nalphSK}+${t}+11]
                  )
                %endfor
                )/(
                    1. + dt*a${ar}${ar}*${Pr/(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor
    }
}
%endfor


%for ar in nars:
// compute the moment of the esbgk kernel
__global__ void ${"update_ARS_MomFSES_stage{0}".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* moms_f, 
    ${" ".join(["const scalar* moms_L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_Q{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_F{0},".format(i) for i in range(1,ar-1)])}
    ${"scalar *moms_F{0}".format(ar-1)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar Tn;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar)])};

    <% epsilon_ = 1./(Pr*prefacESBGK) %>
    <% epsilon__ = prefacESBGK/(Pr*prefac) %>
    <% epsilon = 1./(prefac*prefacESBGK) %>

    if(idx<${Nq*Ne}) 
    {   
        // density, x-velocity, y-velocity, z-velocity, temperature
        %for t in [0, 1, 2, 3, 10]:
            moms_F${ar-1}[idx*${nalphES}+${t}] = moms_f[idx*${nalphES}+${t}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}]".format(
                            ar, i, nalphES, t) for i in range(1, ar)
                    ])};    
        %endfor

        Tn = moms_f[idx*${nalphES}+10]/(1.5*moms_f[idx*${nalphES}+0])
            - 
            (
                moms_f[idx*${nalphES}+1]*moms_f[idx*${nalphES}+1]
                + moms_f[idx*${nalphES}+2]*moms_f[idx*${nalphES}+2]
                + moms_f[idx*${nalphES}+3]*moms_f[idx*${nalphES}+3]
            )/(1.5*pow(moms_f[idx*${nalphES}+0], 2.));

        // T = 1/1.5/rho*(mom10) - (u*u+v*v+w*w)/1.5
        % for i in range(1, ar):
            T${i} = moms_F${i}[idx*${nalphES}+10]/(1.5*moms_F${i}[idx*${nalphES}+0])
            - 
            (
                moms_F${i}[idx*${nalphES}+1]*moms_F${i}[idx*${nalphES}+1]
                + moms_F${i}[idx*${nalphES}+2]*moms_F${i}[idx*${nalphES}+2]
                + moms_F${i}[idx*${nalphES}+3]*moms_F${i}[idx*${nalphES}+3]
            )/(1.5*pow(moms_F${i}[idx*${nalphES}+0], 2.)); 
        %endfor

        // Non-dimensionalization factors
        <% nF = 1. %> 
        <% nF2 = 0.5 %>

        % for t, (ii, jj) in enumerate(['00', '11', '22', '01', '12', '20']):
            <% i, j = map(int, (ii, jj)) %>
            moms_F${ar-1}[idx*${nalphES}+${t}+4] = (
                moms_f[idx*${nalphES}+${t}+4]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphES, t) for _i in range(1, ar)])}
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_Q{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphES, t) for _i in range(1, ar)])}
                - dt*_a${ar}1*${1./(Pr*epsilon)}*moms_f[idx*${nalphES}+0]*pow(Tn, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_f[idx*${nalphES}+0]*Tn
                    %endif
                    + ${nF}*moms_f[idx*${nalphES}+1+${i}]*moms_f[idx*${nalphES}+1+${j}]
                        /moms_f[idx*${nalphES}+0]
                    - ${nF}*moms_f[idx*${nalphES}+${t}+4]
                )
                % for _i in range(2, ar):
                    <% __i = _i - 1 %>
                  - dt*(_a${ar}${_i})*${1./(Pr*epsilon)}*moms_F${__i}[idx*${nalphES}+0]*pow(T${__i}, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_F${__i}[idx*${nalphES}+0]*T${__i}
                    %endif
                    + ${nF}*moms_F${__i}[idx*${nalphES}+1+${i}]*moms_F${__i}[idx*${nalphES}+1+${j}]
                        /moms_F${__i}[idx*${nalphES}+0]
                    - ${nF}*moms_F${__i}[idx*${nalphES}+${t}+4]
                  )
                %endfor
                % for _i in range(1, ar-1):
                  + dt*(a${ar}${_i+1})*${1./(Pr*epsilon)}*moms_F${_i}[idx*${nalphES}+0]*pow(T${_i}, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_F${_i}[idx*${nalphES}+0]*T${_i}
                    %endif
                    + ${nF}*moms_F${_i}[idx*${nalphES}+1+${i}]*moms_F${_i}[idx*${nalphES}+1+${j}]
                        /moms_F${_i}[idx*${nalphES}+0]
                    - ${nF}*moms_F${_i}[idx*${nalphES}+${t}+4]
                  )
                %endfor
                + dt*a${ar}${ar}*${1./(Pr*epsilon)}*moms_F${ar-1}[idx*${nalphES}+0]*pow(T${ar-1}, ${1-omega})
                  *(
                    % if i==j:
                        ${nF2}*moms_F${ar-1}[idx*${nalphES}+0]*T${ar-1}
                    %endif
                    + ${nF}*moms_F${ar-1}[idx*${nalphES}+1+${i}]*moms_F${ar-1}[idx*${nalphES}+1+${j}]
                        /moms_F${ar-1}[idx*${nalphES}+0]
                  )
                )/(
                    1. + dt*a${ar}${ar}*${1./(Pr*epsilon)}*moms_F${ar-1}[idx*${nalphES}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor
    }
}
%endfor


%for ar in nars:
// compute the moment of the esbgk kernel
__global__ void ${"update_ARS_MomFSSK_stage{0}".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* moms_f, 
    ${" ".join(["const scalar* moms_L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_Q{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* moms_F{0},".format(i) for i in range(1,ar-1)])}
    ${"scalar *moms_F{0}".format(ar-1)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar Tn;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar)])};

    <% epsilon_ = 1./prefacSK %>
    <% epsilon = 1./(prefac*prefacSK) %>

    if(idx<${Nq*Ne}) 
    {   
        // density, x-velocity, y-velocity, z-velocity, temperature
        %for t in [0, 1, 2, 3, 10]:
            moms_F${ar-1}[idx*${nalphSK}+${t}] = moms_f[idx*${nalphSK}+${t}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}]".format(
                            ar, i, nalphSK, t) for i in range(1, ar)
                    ])};    
        %endfor

        Tn = moms_f[idx*${nalphSK}+10]/(1.5*moms_f[idx*${nalphSK}+0])
            - 
            (
                moms_f[idx*${nalphSK}+1]*moms_f[idx*${nalphSK}+1]
                + moms_f[idx*${nalphSK}+2]*moms_f[idx*${nalphSK}+2]
                + moms_f[idx*${nalphSK}+3]*moms_f[idx*${nalphSK}+3]
            )/(1.5*pow(moms_f[idx*${nalphSK}+0], 2.));

        // T = 1/1.5/rho*(mom10) - (u*u+v*v+w*w)/1.5
        % for i in range(1, ar):
            T${i} = moms_F${i}[idx*${nalphSK}+10]/(1.5*moms_F${i}[idx*${nalphSK}+0])
            - 
            (
                moms_F${i}[idx*${nalphSK}+1]*moms_F${i}[idx*${nalphSK}+1]
                + moms_F${i}[idx*${nalphSK}+2]*moms_F${i}[idx*${nalphSK}+2]
                + moms_F${i}[idx*${nalphSK}+3]*moms_F${i}[idx*${nalphSK}+3]
            )/(1.5*pow(moms_F${i}[idx*${nalphSK}+0], 2.)); 
        %endfor

        // Evolve stress
        % for t, (ii, jj) in enumerate(['00', '11', '22', '01', '12', '20']):
            <% i, j = map(int, (ii, jj)) %>
            moms_F${ar-1}[idx*${nalphSK}+${t}+4] = (
                moms_f[idx*${nalphSK}+${t}+4]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_Q{1}[idx*{2}+{3}+4]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                - dt*_a${ar}1*${1./(epsilon)}*moms_f[idx*${nalphSK}+0]*pow(Tn, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_f[idx*${nalphSK}+0]*Tn
                    %endif
                    + moms_f[idx*${nalphSK}+1+${i}]*moms_f[idx*${nalphSK}+1+${j}]
                        /moms_f[idx*${nalphSK}+0]
                    - moms_f[idx*${nalphSK}+${t}+4]
                )
                % for _i in range(2, ar):
                    <% __i = _i - 1 %>
                  - dt*(_a${ar}${_i})*${1./(epsilon)}*moms_F${__i}[idx*${nalphSK}+0]*pow(T${__i}, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_F${__i}[idx*${nalphSK}+0]*T${__i}
                    %endif
                    + moms_F${__i}[idx*${nalphSK}+1+${i}]*moms_F${__i}[idx*${nalphSK}+1+${j}]
                        /moms_F${__i}[idx*${nalphSK}+0]
                    - moms_F${__i}[idx*${nalphSK}+${t}+4]
                  )
                %endfor
                % for _i in range(1, ar-1):
                  + dt*a${ar}${_i+1}*${1./(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_F${_i}[idx*${nalphSK}+0]*T${_i}
                    %endif
                    + moms_F${_i}[idx*${nalphSK}+1+${i}]*moms_F${_i}[idx*${nalphSK}+1+${j}]
                        /moms_F${_i}[idx*${nalphSK}+0]
                    - moms_F${_i}[idx*${nalphSK}+${t}+4]
                  )
                %endfor
                + dt*a${ar}${ar}*${1./(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                  *(
                    % if i==j:
                        0.5*moms_F${ar-1}[idx*${nalphSK}+0]*T${ar-1}
                    %endif
                    + moms_F${ar-1}[idx*${nalphSK}+1+${i}]*moms_F${ar-1}[idx*${nalphSK}+1+${j}]
                        /moms_F${ar-1}[idx*${nalphSK}+0]
                  )
                )/(
                    1. + dt*a${ar}${ar}*${1./(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor

        // Evolve heat flux
        % for t, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            moms_F${ar-1}[idx*${nalphSK}+${t}+11] = (
                moms_f[idx*${nalphSK}+${t}+11]
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_L{1}[idx*{2}+{3}+11]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                + ${"+ ".join([
                        "dt*_a{0}{1}*moms_Q{1}[idx*{2}+{3}+11]".format(
                            ar, _i, nalphSK, t) for _i in range(1, ar)])}
                - dt*_a${ar}1*${(1-Pr)/(epsilon)}*moms_f[idx*${nalphSK}+0]*pow(Tn, ${1-omega})
                  *(
                    -2.*(
                        moms_f[idx*${nalphSK}+4+${q1}]*moms_f[idx*${nalphSK}+1]
                      + moms_f[idx*${nalphSK}+4+${q2}]*moms_f[idx*${nalphSK}+2]
                      + moms_f[idx*${nalphSK}+4+${q3}]*moms_f[idx*${nalphSK}+3]
                    )/moms_f[idx*${nalphSK}+0]
                    + moms_f[idx*${nalphSK}+1+${t}]*(
                        moms_f[idx*${nalphSK}+1]*moms_f[idx*${nalphSK}+1]
                      + moms_f[idx*${nalphSK}+2]*moms_f[idx*${nalphSK}+2]
                      + moms_f[idx*${nalphSK}+3]*moms_f[idx*${nalphSK}+3]
                    )/pow(moms_f[idx*${nalphSK}+0], 2.)
                    - 1.5*moms_f[idx*${nalphSK}+1+${t}]*Tn
                  )
                - dt*_a${ar}1*${1./(epsilon)}*moms_f[idx*${nalphSK}+0]*pow(Tn, ${1-omega})
                  *(
                    + moms_f[idx*${nalphSK}+1+${t}]*(
                        moms_f[idx*${nalphSK}+1]*moms_f[idx*${nalphSK}+1]
                      + moms_f[idx*${nalphSK}+2]*moms_f[idx*${nalphSK}+2]
                      + moms_f[idx*${nalphSK}+3]*moms_f[idx*${nalphSK}+3]
                    )/pow(moms_f[idx*${nalphSK}+0], 2.)
                    + 2.5*moms_f[idx*${nalphSK}+1+${t}]*Tn
                  )
                + dt*_a${ar}1*${(Pr)/(epsilon)}*moms_f[idx*${nalphSK}+0]*pow(Tn, ${1-omega})
                  *(
                    moms_f[idx*${nalphSK}+${t}+11]
                  )
                % for _i in range(2, ar):
                    <% __i = _i - 1 %>
                  - dt*_a${ar}${_i}*${(1-Pr)/(epsilon)}*moms_F${__i}[idx*${nalphSK}+0]*pow(T${__i}, ${1-omega})
                  *(
                    -2.*(
                        moms_F${__i}[idx*${nalphSK}+4+${q1}]*moms_F${__i}[idx*${nalphSK}+1]
                      + moms_F${__i}[idx*${nalphSK}+4+${q2}]*moms_F${__i}[idx*${nalphSK}+2]
                      + moms_F${__i}[idx*${nalphSK}+4+${q3}]*moms_F${__i}[idx*${nalphSK}+3]
                    )/moms_F${__i}[idx*${nalphSK}+0]
                    + moms_F${__i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${__i}[idx*${nalphSK}+1]*moms_F${__i}[idx*${nalphSK}+1]
                      + moms_F${__i}[idx*${nalphSK}+2]*moms_F${__i}[idx*${nalphSK}+2]
                      + moms_F${__i}[idx*${nalphSK}+3]*moms_F${__i}[idx*${nalphSK}+3]
                    )/pow(moms_F${__i}[idx*${nalphSK}+0], 2.)
                    - 1.5*moms_F${__i}[idx*${nalphSK}+1+${t}]*T${__i}
                  )
                  - dt*_a${ar}${_i}*${1./(epsilon)}*moms_F${__i}[idx*${nalphSK}+0]*pow(T${__i}, ${1-omega})
                  *(
                    + moms_F${__i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${__i}[idx*${nalphSK}+1]*moms_F${__i}[idx*${nalphSK}+1]
                      + moms_F${__i}[idx*${nalphSK}+2]*moms_F${__i}[idx*${nalphSK}+2]
                      + moms_F${__i}[idx*${nalphSK}+3]*moms_F${__i}[idx*${nalphSK}+3]
                    )/pow(moms_F${__i}[idx*${nalphSK}+0], 2.)
                    + 2.5*moms_F${__i}[idx*${nalphSK}+1+${t}]*T${__i}
                  )
                  + dt*_a${ar}${_i}*${(Pr)/(epsilon)}*moms_F${__i}[idx*${nalphSK}+0]*pow(T${__i}, ${1-omega})
                  *(
                    moms_F${__i}[idx*${nalphSK}+${t}+11]
                  )
                %endfor
                % for _i in range(1, ar):
                  + dt*a${ar}${_i+1}*${(1-Pr)/(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    -2.*(
                        moms_F${_i}[idx*${nalphSK}+4+${q1}]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+4+${q2}]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+4+${q3}]*moms_F${_i}[idx*${nalphSK}+3]
                    )/moms_F${_i}[idx*${nalphSK}+0]
                    + moms_F${_i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${_i}[idx*${nalphSK}+1]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+2]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+3]*moms_F${_i}[idx*${nalphSK}+3]
                    )/pow(moms_F${_i}[idx*${nalphSK}+0], 2.)
                    - 1.5*moms_F${_i}[idx*${nalphSK}+1+${t}]*T${_i}
                  )
                  + dt*a${ar}${_i+1}*${1./(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    + moms_F${_i}[idx*${nalphSK}+1+${t}]*(
                        moms_F${_i}[idx*${nalphSK}+1]*moms_F${_i}[idx*${nalphSK}+1]
                      + moms_F${_i}[idx*${nalphSK}+2]*moms_F${_i}[idx*${nalphSK}+2]
                      + moms_F${_i}[idx*${nalphSK}+3]*moms_F${_i}[idx*${nalphSK}+3]
                    )/pow(moms_F${_i}[idx*${nalphSK}+0], 2.)
                    + 2.5*moms_F${_i}[idx*${nalphSK}+1+${t}]*T${_i}
                  )
                %endfor
                % for _i in range(1, ar-1):
                  - dt*a${ar}${_i+1}*${(Pr)/(epsilon)}*moms_F${_i}[idx*${nalphSK}+0]*pow(T${_i}, ${1-omega})
                  *(
                    moms_F${_i}[idx*${nalphSK}+${t}+11]
                  )
                %endfor
                )/(
                    1. + dt*a${ar}${ar}*${Pr/(epsilon)}*moms_F${ar-1}[idx*${nalphSK}+0]*pow(T${ar-1}, ${1-omega})
                );                
        %endfor
    }
}
%endfor

// Construct maxwellian
__global__ void updateDistribution
(
    const scalar dt,
    const scalar* collFreq,
    const scalar* fS, 
    const scalar* Q, 
    const scalar* M0, const scalar* M1,
    scalar* f
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar beta;

    <% epsilon = 1./prefac %>

    if(idx<${vsize}) 
    {
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                beta = collFreq[${k*Ne+e}];
                //beta = collFreq[id];
                //beta = collFreq[0];

                f[id] = ${epsilon}/(${epsilon}+dt*beta)*fS[id]
                        + dt/(${epsilon}+dt*beta)*(${epsilon}*Q[id]-beta*(M0[id]-f[id]))
                        + dt*beta/(${epsilon}+dt*beta)*M1[id]; 

                //f[id] = ${epsilon}/(${epsilon}+dt*beta)*fS[id]
                //        + dt/(${epsilon}+dt*beta)*(Q[id]-beta*(M0[id]-f[id]))
                //        + dt*beta/(${epsilon}+dt*beta)*M1[id]; 

                // Q already has the factor 1/epsilon included
                //f[id] = fS[id] + dt*Q[id];
            %endfor 
        %endfor
    }
}


__global__ void extractAt
(
    const int elem,
    const int mode,
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_s = (mode*${Ne}+elem)*${vsize} + idx; 

    if(idx<${vsize}) {
        out[idx] = in[idx_s];
    }
}

__global__ void insertAtOne
(
    const int elem,
    const int mode,
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idx_s = (mode*${Ne}+elem)*${vsize} + idx; 
    int idx_s = (mode*${Ne}+elem);

    if(idx<1) {
        //out[idx] = in[idx_s];
        out[idx_s] = in[0];
    }
}

%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_BGKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    <% epsilon = 1./prefacBGK %>

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] 
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon) for i in range(2, ar))}
                    + dt*a${ar}${ar}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*nu${ar}[${ke}]/(${epsilon})); 
            %endfor 
        %endfor        
    }
}
%endfor


%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_ESBGKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    <% epsilon = 1./prefacESBGK %>

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] 
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon) for i in range(2, ar))}
                    + dt*a${ar}${ar}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*nu${ar}[${ke}]/(${epsilon})); 
            %endfor 
        %endfor        
    }
}
%endfor


%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_SKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    <% epsilon = 1./prefacSK %>

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] 
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon) for i in range(2, ar))}
                    + dt*a${ar}${ar}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*nu${ar}[${ke}]/(${epsilon})); 
            %endfor 
        %endfor        
    }
}
%endfor


%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_FSBGKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* nu1, const scalar* M1, const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* Q{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;    

    <% epsilon = 1./prefac %>
    <% fac = prefacBGK %>
    // Q is computed from scattering.py; and already contains the factor of epsilon

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] +
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*_a{0}{1}*(Q{1}[id]-{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id]))".format(
                            ar, i, ke, epsilon, fac) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon, fac) for i in range(2, ar))}
                    + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}); 
            %endfor 
        %endfor        
    }
}
%endfor





%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_FSESBGKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* nu1, const scalar* M1, const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* Q{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;    

    <% epsilon = 1./prefac %>
    <% fac = prefacESBGK %>
    // Q is computed from scattering.py; and already contains the factor of epsilon

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] +
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*_a{0}{1}*(Q{1}[id]-{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id]))".format(
                            ar, i, ke, epsilon, fac) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon, fac) for i in range(2, ar))}
                    + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}); 
            %endfor 
        %endfor        
    }
}
%endfor




%for ar in nars:
// compute the moment of the bgk kernel
__global__ void ${"updateDistribution{0}_FSSKARS".format(ar)}
(
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(1,ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(1,ar)])}
    const scalar* nu1, const scalar* M1, const scalar* F1, 
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* Q{0},".format(i) for i in range(1,ar)])}
    ${" ".join(["const scalar* nu{0}, const scalar* M{0}, const scalar* F{0},".format(i) for i in range(2,ar)])}
    ${"const scalar* nu{0}, const scalar* M{0}, scalar *F{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;    

    <% epsilon = 1./prefac %>
    <% fac = prefacSK %>
    // Q is computed from scattering.py; and already contains the factor of epsilon

    if(idx<${vsize}) 
    {   
        % for k in range(K):
            % for e in range(Ne):
                id = ${(k*Ne+e)*vsize}+idx;
                <% ke = (k*Ne+e) %>
                
                F${ar}[id] = (F1[id] +
                    + ${"+ ".join("dt*_a{0}{1}*L{1}[id]".format(
                            ar, i) for i in range(1, ar))}
                    + ${"+ ".join("dt*_a{0}{1}*(Q{1}[id]-{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id]))".format(
                            ar, i, ke, epsilon, fac) for i in range(1, ar))}
                    + ${"+ ".join("dt*a{0}{1}*{4}*nu{1}[{2}]/{3}*(M{1}[id]-F{1}[id])".format(
                            ar, i, ke, epsilon, fac) for i in range(2, ar))}
                    + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}*(M${ar}[id])                    
                )/    
                (1. + dt*a${ar}${ar}*${fac}*nu${ar}[${ke}]/${epsilon}); 
            %endfor 
        %endfor        
    }
}
%endfor

__global__ void totalFlux2
(
    scalar* ux,
    const scalar* cvx,
    const scalar* jL,
    const scalar* jR
)
{
    // Compute the continuous flux for each element
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int ide, idv;

    if(idx<${Ne*vsize}) 
    {
        idv = idx%${vsize};
        ide = idx/${vsize};

        % for k in range(K):
            id = (${k*Ne}+ide)*${vsize}+idv;
            ux[id] += -jL[${(ide+1)*vsize}+idx]*${gRD[k]} 
                          +jR[${ide*vsize}+idx]*${gLD[k]};
        %endfor
    }
}
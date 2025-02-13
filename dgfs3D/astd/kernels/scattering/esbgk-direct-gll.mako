%if dtype == 'double':
    #define scalar double
    #define Exp exp
%elif dtype == 'float':
    #define scalar float
    #define Exp expf
%else:
    #error "undefined floating point data type"
%endif

<%!
import math 
%>


%for ar in nbdf:
// compute the moment of the esbgk kernel
__global__ void ${"updateMom{0}_BDF".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join([
      "const scalar a{0}, const scalar* U{0}, const scalar g{0}, const scalar* LU{0},".format(i) 
      for i in range(ar)
    ])}
    const scalar a${ar}, scalar* U, const scalar b
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar T;

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}))/a${ar};    
        %endfor

        T = (U[idx*${nalph}+4]
            - (
                + U[idx*${nalph}+1]*U[idx*${nalph}+1]
                + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                + U[idx*${nalph}+3]*U[idx*${nalph}+3]
              )/U[idx*${nalph}+0]
          )/(1.5*U[idx*${nalph}+0]);

        // diagonal/normal components of the stress
        %for iter in [5, 6, 7]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}) 
              + b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    0.5*U[idx*${nalph}+0]*T
                  + U[idx*${nalph}+${iter-4}]*U[idx*${nalph}+${iter-4}]
                      /U[idx*${nalph}+0]
                )
            )/(a${ar}+b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])})
            + b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    U[idx*${nalph}+${iter-7}]*U[idx*${nalph}+${(iter-7)%3+1}]
                      /U[idx*${nalph}+0]
                  )
            )/(a${ar}+b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor
    }
}
%endfor


%for ar in nbdf:
// compute the moment of the esbgk kernel
__global__ void ${"updateDistribution{0}_BDF".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join([
      "const scalar a{0}, const scalar* f{0}, const scalar g{0}, const scalar* L{0},".format(i) 
      for i in range(ar)
    ])}
    const scalar b, const scalar* M,
    const scalar a${ar}, const scalar* U, scalar* f
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar nu;

    if(idx<lda) 
    {   
      id = idx/${vsize};
      nu = (  
            U[id*${nalph}+4]
            - (
                U[id*${nalph}+1]*U[id*${nalph}+1]
                + U[id*${nalph}+2]*U[id*${nalph}+2]
                + U[id*${nalph}+3]*U[id*${nalph}+3]
              )/U[id*${nalph}+0]
          )/(1.5*U[id*${nalph}+0]);  // Temperature
      nu = prefac*U[id*${nalph}+0]*pow(nu, ${1.-omega});
      f[idx] = (-(${"+ ".join(["a{0}*f{0}[idx] + g{0}*dt*L{0}[idx]".format(i) 
                                for i in range(ar)
                  ])}) + b*dt*nu*M[idx])/(a${ar}+b*dt*nu);    
    }     
}
%endfor


%for ar in nbdf:
// compute the moment of the bgk kernel
__global__ void ${"updateDistributionNu{0}_BDF".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join([
      "const scalar a{0}, const scalar* f{0}, const scalar g{0}, const scalar* L{0},".format(i) 
      for i in range(ar)
    ])}
    const scalar b, const scalar* M,
    const scalar a${ar}, const scalar* U, scalar* f
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar nu;

    if(idx<lda) 
    {   
      id = idx/${vsize};
      //nu = prefac*U[id*${nalph}+0];
      nu = prefac*U[id];
      f[idx] = (-(${"+ ".join(["a{0}*f{0}[idx] + g{0}*dt*L{0}[idx]".format(i) 
                                for i in range(ar)
                  ])}) + b*dt*nu*M[idx])/(a${ar}+b*dt*nu);    
    }     
}
%endfor



// Construct maxwellian
__global__ void momentNorm
(
  const int lda,
    scalar* U
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar T;

    if(idx<lda) 
    {   
        U[idx*${nalph}+0] = U[idx*${nalph}+0]; // density
        U[idx*${nalph}+1] /= U[idx*${nalph}+0]; // x-velocity
        U[idx*${nalph}+2] /= U[idx*${nalph}+0]; // y-velocity
        U[idx*${nalph}+3] /= U[idx*${nalph}+0]; // z-velocity

        // T = 1/1.5/rho*(mom5 + mom6 + mom7) - (u*u+v*v+w*w)/1.5
        T = (
                U[idx*${nalph}+4]
            )/(1.5*U[idx*${nalph}+0])
          - 
            (
                U[idx*${nalph}+1]*U[idx*${nalph}+1]
                + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                + U[idx*${nalph}+3]*U[idx*${nalph}+3]
            )/1.5;
        
        // Non-dimensionalization factors
        <% nF = 2.0 %> 
        <% nF2 = 1.0 %>

        // Temperature
        U[idx*${nalph}+4] = T;

        // T_xx
        U[idx*${nalph}+5] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+5]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+1]*U[idx*${nalph}+1];

        // T_yy
        U[idx*${nalph}+6] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+6]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+2]*U[idx*${nalph}+2];

        // T_zz
        U[idx*${nalph}+7] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+7]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+3]*U[idx*${nalph}+3];

        // T_xy
        U[idx*${nalph}+8] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+8]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+1]*U[idx*${nalph}+2];

        // T_yz
        U[idx*${nalph}+9] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+9]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+2]*U[idx*${nalph}+3];

        // T_xz
        U[idx*${nalph}+10] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+10]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+3]*U[idx*${nalph}+1];
    }
}

// Construct gaussian
__global__ void cmaxwellian
(
  const int lda,
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* M, const scalar* U
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id, idv;
    scalar detT;

    if(idx<lda) 
    {
      id = idx/${vsize};
      idv = idx%${vsize};

      detT = (-U[id*${nalph}+7]*U[id*${nalph}+8]*U[id*${nalph}+8]
                    + 2*U[id*${nalph}+8]*U[id*${nalph}+9]*U[id*${nalph}+10] 
                    - U[id*${nalph}+5]*U[id*${nalph}+9]*U[id*${nalph}+9] 
                    - U[id*${nalph}+6]*U[id*${nalph}+10]*U[id*${nalph}+10] 
                    + U[id*${nalph}+5]*U[id*${nalph}+6]*U[id*${nalph}+7]);

      M[idx] = U[id*${nalph}+0]/(${(math.pi)**1.5}*sqrt(detT))*exp(
                -(
                    + (U[id*${nalph}+6]*U[id*${nalph}+7] - U[id*${nalph}+9]*U[id*${nalph}+9])*((cx[idv]-U[id*${nalph}+1])*(cx[idv]-U[id*${nalph}+1]))
                    + (U[id*${nalph}+5]*U[id*${nalph}+7] - U[id*${nalph}+10]*U[id*${nalph}+10])*((cy[idv]-U[id*${nalph}+2])*(cy[idv]-U[id*${nalph}+2]))
                    + (U[id*${nalph}+5]*U[id*${nalph}+6] - U[id*${nalph}+8]*U[id*${nalph}+8])*((cz[idv]-U[id*${nalph}+3])*(cz[idv]-U[id*${nalph}+3]))
                    + 2*(U[id*${nalph}+9]*U[id*${nalph}+10] - U[id*${nalph}+7]*U[id*${nalph}+8])*((cx[idv]-U[id*${nalph}+1])*(cy[idv]-U[id*${nalph}+2]))
                    + 2*(U[id*${nalph}+8]*U[id*${nalph}+10] - U[id*${nalph}+5]*U[id*${nalph}+9])*((cy[idv]-U[id*${nalph}+2])*(cz[idv]-U[id*${nalph}+3]))
                    + 2*(U[id*${nalph}+8]*U[id*${nalph}+9] - U[id*${nalph}+10]*U[id*${nalph}+6])*((cz[idv]-U[id*${nalph}+3])*(cx[idv]-U[id*${nalph}+1]))
                )/(detT)
            );
    }
}

// Construct the collision operator
__global__ void collide
(
    const scalar prefac,
    const int lda,
    const scalar* M, const scalar* U, const scalar* f, scalar* Q
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar nu;

    if(idx<lda) 
    {   
      id = idx/${vsize};
      nu = (  
            U[id*${nalph}+4]
            - (
                U[id*${nalph}+1]*U[id*${nalph}+1]
                + U[id*${nalph}+2]*U[id*${nalph}+2]
                + U[id*${nalph}+3]*U[id*${nalph}+3]
              )/U[id*${nalph}+0]
          )/(1.5*U[id*${nalph}+0]);  // Temperature
      nu = prefac*U[id*${nalph}+0]*pow(nu, ${1.-omega});
      Q[idx] = nu*(M[idx]-f[idx]); 
    }     
}

// Construct the collision operator
__global__ void collide_nu
(
    const scalar prefac,
    const int lda,
    const scalar* M, const scalar* U, const scalar* f, scalar* Q
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<lda) 
    {   
      id = idx/${vsize};
      //Q[idx] = prefac*U[id*${nalph}+0]*(M[idx]-f[idx]);
      Q[idx] = prefac*U[id]*(M[idx]-f[idx]); 
    }     
}

%for ar in nars:
__global__ void ${"updateMom{0}_ARS".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar* LU{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar)])}
    ${"scalar *U{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar+1)])};

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U${ar}[idx*${nalph}+${iter}] = U0[idx*${nalph}+${iter}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])};    
        %endfor

        %for i in range(1, ar+1):
            T${i} = (  
                  U${i}[idx*${nalph}+4]
                  - (
                      U${i}[idx*${nalph}+1]*U${i}[idx*${nalph}+1]
                      + U${i}[idx*${nalph}+2]*U${i}[idx*${nalph}+2]
                      + U${i}[idx*${nalph}+3]*U${i}[idx*${nalph}+3]
                    )/U${i}[idx*${nalph}+0]
                )/(1.5*U${i}[idx*${nalph}+0]);  // Temperature
        %endfor

        // diagonal/normal components of the stress
        %for iter in [5, 6, 7]:
            U${ar}[idx*${nalph}+${iter}] = (U0[idx*${nalph}+${iter}] +
              + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])} 
              %for i in range(1, ar+1):
              + a${ar}${i}/${Pr}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    0.5*U${i}[idx*${nalph}+0]*T${i}
                  + U${i}[idx*${nalph}+${iter-4}]*U${i}[idx*${nalph}+${iter-4}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U${ar}[idx*${nalph}+${iter}] = (U0[idx*${nalph}+${iter}] +
              + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])} 
              %for i in range(1, ar+1):
              + a${ar}${i}/${Pr}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    U${i}[idx*${nalph}+${iter-7}]*U${i}[idx*${nalph}+${(iter-7)%3+1}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor
    }
}
%endfor


%for ar in nars:
__global__ void ${"updateDistribution{0}_ARS".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar* L{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar+1)])}
    ${" ".join(["const scalar* M{0},".format(i) for i in range(1,ar+1)])}
    ${" ".join(["const scalar* f{0},".format(i) for i in range(ar)])}
    scalar *f${ar}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar ${", ".join(["nu{0}".format(i) for i in range(1, ar+1)])};

    if(idx<lda) 
    {   
      id = idx/${vsize};

      %for i in range(1, ar+1):
        nu${i} = (  
              U${i}[id*${nalph}+4]
              - (
                  U${i}[id*${nalph}+1]*U${i}[id*${nalph}+1]
                  + U${i}[id*${nalph}+2]*U${i}[id*${nalph}+2]
                  + U${i}[id*${nalph}+3]*U${i}[id*${nalph}+3]
                )/U${i}[id*${nalph}+0]
            )/(1.5*U${i}[id*${nalph}+0]);  // Temperature
        nu${i} = prefac*U${i}[id*${nalph}+0]*pow(nu${i}, ${1.-omega});
      %endfor
          
      f${ar}[idx] = (f0[idx] 
        + ${"+ ".join("_a{0}{1}*dt*L{1}[idx]".format(ar, i) for i in range(ar))}
        + ${"+ ".join("a{0}{1}*dt*nu{1}*(M{1}[idx]-f{1}[idx])".format(
                            ar, i) for i in range(1, ar))}
        + a${ar}${ar}*dt*nu${ar}*(M${ar}[idx])                    
      )
      /(1. + a${ar}${ar}*dt*nu${ar}); 
    }
}
%endfor


%for ar in nars:
__global__ void ${"updateDistributionWeight{0}_SSPL".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar* L{0},".format(i) for i in range(1, ar+1)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar+1)])}
    ${" ".join(["const scalar* M{0},".format(i) for i in range(1, ar+1)])}
    ${" ".join(["const scalar* f{0},".format(i) for i in range(ar+1)])}
    scalar *f${ar+1}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar ${", ".join(["nu{0}".format(i) for i in range(1, ar+1)])};

    if(idx<lda) 
    {   
      id = idx/${vsize};

      %for i in range(1, ar+1):
        nu${i} = (  
              U${i}[id*${nalph}+4]
              - (
                  U${i}[id*${nalph}+1]*U${i}[id*${nalph}+1]
                  + U${i}[id*${nalph}+2]*U${i}[id*${nalph}+2]
                  + U${i}[id*${nalph}+3]*U${i}[id*${nalph}+3]
                )/U${i}[id*${nalph}+0]
            )/(1.5*U${i}[id*${nalph}+0]);  // Temperature
        nu${i} = prefac*U${i}[id*${nalph}+0]*pow(nu${i}, ${1.-omega});
      %endfor
          
      f${ar+1}[idx] = (f0[idx] 
        + ${"+ ".join("_a{0}{1}*dt*L{1}[idx]".format(ar, i) for i in range(1, ar+1))}
        + ${"+ ".join("a{0}{1}*dt*nu{1}*(M{1}[idx]-f{1}[idx])".format(
                            ar, i) for i in range(1, ar+1))}
      ); 
    }
}
%endfor

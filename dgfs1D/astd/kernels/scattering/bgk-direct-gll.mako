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
// compute the moment of the bgk kernel
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

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}))/a${ar};    
        %endfor
    }
}
%endfor


%for ar in nbdf:
// compute the moment of the bgk kernel
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
__global__ void cmaxwellian
(
  const int lda,
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* M, const scalar* U
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id, idv;
    scalar T;

    if(idx<lda) 
    {
      id = idx/${vsize};
      idv = idx%${vsize};
      T = (
            U[id*${nalph}+4]
            - (
                + U[id*${nalph}+1]*U[id*${nalph}+1]
                + U[id*${nalph}+2]*U[id*${nalph}+2]
                + U[id*${nalph}+3]*U[id*${nalph}+3]
              )/U[id*${nalph}+0]
          )/(1.5*U[id*${nalph}+0]);

      M[idx] = U[id*${nalph}+0]/pow(${math.pi}*T, 1.5)*exp(
                -(
                    (cx[idv]-U[id*${nalph}+1]/U[id*${nalph}+0])
                    *(cx[idv]-U[id*${nalph}+1]/U[id*${nalph}+0])
                  + (cy[idv]-U[id*${nalph}+2]/U[id*${nalph}+0])
                    *(cy[idv]-U[id*${nalph}+2]/U[id*${nalph}+0])
                  + (cz[idv]-U[id*${nalph}+3]/U[id*${nalph}+0])
                    *(cz[idv]-U[id*${nalph}+3]/U[id*${nalph}+0])
                  )/T
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
        %if ar>1:
        + ${"+ ".join("a{0}{1}*dt*nu{1}*(M{1}[idx]-f{1}[idx])".format(
                            ar, i) for i in range(1, ar))}
        %endif
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


%for ar in nbdf:
__global__ void ${"updateMom{0}_LM".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join(["const scalar a{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar b{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar c{0},".format(i) for i in range(ar+1)])}
    ${" ".join(["const scalar* LU{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* LM{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar)])}
    ${"scalar *U{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U${ar}[idx*${nalph}+${iter}] = (${"+ ".join(
                [
                  "-a{4}*U{1}[idx*{2}+{3}] + dt*b{4}*LU{1}[idx*{2}+{3}] + dt*c{5}*LM{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter, ar-i-1, ar-i) for i in range(ar)
                ])});    
        %endfor
    }
}
%endfor


%for ar in nbdf:
__global__ void ${"updateDistribution{0}_LM".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join(["const scalar a{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar b{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar c{0},".format(i) for i in range(ar+1)])}
    ${" ".join(["const scalar* L{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* M{0},".format(i) for i in range(ar)])}
    ${"scalar *M{0}".format(ar)},
    ${" ".join(["const scalar* F{0},".format(i) for i in range(ar)])}
    ${"scalar *f{0}".format(ar)},
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar)])}
    ${"scalar *U{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar ${", ".join(["nu{0}".format(i) for i in range(ar,ar+1)])};

    if(idx<lda) 
    {   
      id = idx/${vsize};

      %for i in range(ar,ar+1):
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
          
      f${ar}[idx] = (
          ${"+ ".join("-a{2}*F{1}[idx] + dt*b{2}*L{1}[idx] + dt*c{3}*M{1}[idx]"
              .format(ar, i, ar-i-1, ar-i) for i in range(ar))}
          + c0*dt*nu${ar}*(M${ar}[idx])                    
        )
        /(1. + c0*dt*nu${ar}); 
    }
}
%endfor
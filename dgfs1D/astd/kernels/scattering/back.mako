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

        T = (U${ar}[idx*${nalph}+4]
            - (
                + U${ar}[idx*${nalph}+1]*U${ar}[idx*${nalph}+1]
                + U${ar}[idx*${nalph}+2]*U${ar}[idx*${nalph}+2]
                + U${ar}[idx*${nalph}+3]*U${ar}[idx*${nalph}+3]
              )/U${ar}[idx*${nalph}+0]
          )/(1.5*U${ar}[idx*${nalph}+0]);

        // diagonal/normal, off-diagonal components of the stress
        %for iter in [5, 6, 7, 8, 9, 10]:
            U${ar}[idx*${nalph}+${iter}] = ((${"+ ".join([
                        "-a{4}*U{1}[idx*{2}+{3}] + dt*b{4}*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter, ar-i-1, ar-i) for i in range(ar)
                    ])}) 

              %for i in range(ar):
              + c${ar-i}/${Pr}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    %if iter<=7:
                      0.5*U${i}[idx*${nalph}+0]*T${i}
                    %endif
                    + U${i}[idx*${nalph}+${iter-4}]*U${i}[idx*${nalph}+${iter-4}]
                      /U${i}[idx*${nalph}+0]
                    - U${i}[idx*${nalph}+${iter}]
                )
              %endfor

              + c0/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    %if iter<=7:
                      0.5*U${ar}[idx*${nalph}+0]*T
                    %endif
                    + U${ar}[idx*${nalph}+${iter-4}]*U[idx*${nalph}+${iter-4}]
                      /U${ar}[idx*${nalph}+0]
                )
            )/(1+c0/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor
    }
}
%endfor

// diagonal/normal, off-diagonal components of the stress
        %for iter in [5, 6, 7, 8, 9, 10]:
            U${ar}[idx*${nalph}+${iter}] = ((${"+ ".join([
                        "-a{4}*U{1}[idx*{2}+{3}] + dt*b{4}*LU{1}[idx*{2}+{3}] + dt*c{5}*LM{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter, ar-i-1, ar-i) for i in range(ar)
                    ])}) 

              + c0/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    %if iter<=7:
                      0.5*U${ar}[idx*${nalph}+0]*T
                    %endif
                    + U${ar}[idx*${nalph}+${iter-4}]*U${ar}[idx*${nalph}+${iter-4}]
                      /U${ar}[idx*${nalph}+0]
                )
            )/(1+c0/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor
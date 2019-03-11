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

__global__ void flocKern
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


__global__ void mom1
(
    const scalar* cx,
    const scalar* cy,
    const scalar* cz,    
    const scalar* f,
    scalar* mom10, scalar* mom11, scalar* mom12
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) {
        mom10[idx] = f[idx]*cx[idx];
        mom11[idx] = f[idx]*cy[idx];
        mom12[idx] = f[idx]*cz[idx];
    }
}

__global__ void mom01Norm
(
    scalar* locRho, scalar* locUx, scalar* locUy, scalar* locUz
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx==0)
    {
        locUx[0] /= locRho[0];
        locUy[0] /= locRho[0];
        locUz[0] /= locRho[0];
        locRho[0] *= ${cw};
    }
}

__global__ void mom2
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    const scalar* f,
    scalar* mom2,
    scalar* locRho, scalar* locUx, scalar* locUy, scalar* locUz
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) {
        mom2[idx] = f[idx]*(
             (cx[idx]-locUx[0])*(cx[idx]-locUx[0])
           + (cy[idx]-locUy[0])*(cy[idx]-locUy[0])
           + (cz[idx]-locUz[0])*(cz[idx]-locUz[0])
        );
    }
}

__global__ void equiDistInit
(
    ${", ".join(["scalar* moms{0}".format(i) for i in range(nalph)])}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx==0)
    {
        // Add the missing factor 
        moms4[0] *= ${cw}/(1.5*moms0[0]);
    }
}


__global__ void equiDistCompute
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    scalar* fe, 
    const scalar* locRho, 
    const scalar* locUx, const scalar* locUy, const scalar* locUz, 
    const scalar* locT
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;    

    if(idx<${vsize}) {
        fe[idx] = locRho[0]/pow(${math.pi}*locT[0], 1.5)
        *Exp(
            -(
                (cx[idx]-locUx[0])*(cx[idx]-locUx[0])
              + (cy[idx]-locUy[0])*(cy[idx]-locUy[0])
              + (cz[idx]-locUz[0])*(cz[idx]-locUz[0])
            )/locT[0]
        ); 
    }
}

__global__ void mom2ES
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    const scalar* f, const scalar* feBGK,
    const scalar* locRho, 
    const scalar* locUx, const scalar* locUy, const scalar* locUz,
    scalar* equiMom2es_xx, scalar* equiMom2es_yy,
    scalar* equiMom2es_zz, scalar* equiMom2es_xy,
    scalar* equiMom2es_yz, scalar* equiMom2es_zx
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) {
    // This is fine: The cross terms for feBGK would be zero
      % for i, j in ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']:
        ${"equiMom2es_{0}{1}".format(i,j)}[idx] = 
          ${"(c{0}[idx]-locU{0}[0])*(c{1}[idx]-locU{1}[0])".format(i, j)}
          *(${1.-1./Pr}*f[idx] + ${1./Pr}*feBGK[idx]);
      %endfor    
    }
}

__global__ void mom2ESNorm
(
    ${", ".join(["const scalar* moms{0}".format(i) for i in range(nalph)])},
    ${", ".join(["scalar* momsES{0}".format(i) for i in range(nalphES)])}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx==0) {
      // Copy the first four moments
      % for i in range(4):
        ${"momsES{0}[0] = moms{0}[0]".format(i)};
      %endfor

      // normalize the next four moments by density
      % for i in range(4, nalphES):
        ${"momsES{0}[0] *= ({1}/moms0[0])".format(i, cw)};
      %endfor
    }
}

__global__ void equiESDistCompute
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    scalar* feES, 
    ${", ".join(["scalar* momsES{0}".format(i) for i in range(nalphES)])}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Symmetric matrix
    scalar detT = -momsES6[0]*momsES7[0]*momsES7[0]
            + 2*momsES7[0]*momsES8[0]*momsES9[0] 
            - momsES4[0]*momsES8[0]*momsES8[0] 
            - momsES5[0]*momsES9[0]*momsES9[0] 
            + momsES4[0]*momsES5[0]*momsES6[0];

    // inverse of the symmetric matrix
    __shared__ scalar A[6];  // xx, yy, zz, xy, yz, zx; 11 22 33 12 23 31
    if(threadIdx.x == 0)
    {
        A[0] = (momsES5[0]*momsES6[0] - momsES8[0]*momsES8[0])/detT; // 11
        A[1] = (momsES4[0]*momsES6[0] - momsES9[0]*momsES9[0])/detT; // 22
        A[2] = (momsES4[0]*momsES5[0] - momsES7[0]*momsES7[0])/detT; // 33

        A[3] = (momsES8[0]*momsES9[0] - momsES6[0]*momsES7[0])/detT; // 12
        A[4] = (momsES7[0]*momsES9[0] - momsES4[0]*momsES8[0])/detT; // 23
        A[5] = (momsES7[0]*momsES8[0] - momsES9[0]*momsES5[0])/detT; // 31
    }
    __syncthreads();
    
    if(idx<${vsize}) {
        feES[idx] = 
            momsES0[0]/(${(2*math.pi)**1.5}*sqrt(detT))*Exp(
            -(
                + A[0]*((cx[idx]-momsES1[0])*(cx[idx]-momsES1[0]))
                + A[1]*((cy[idx]-momsES2[0])*(cy[idx]-momsES2[0]))
                + A[2]*((cz[idx]-momsES3[0])*(cz[idx]-momsES3[0]))
                + 2*A[3]*((cx[idx]-momsES1[0])*(cy[idx]-momsES2[0]))
                + 2*A[4]*((cy[idx]-momsES2[0])*(cz[idx]-momsES3[0]))
                + 2*A[5]*((cz[idx]-momsES3[0])*(cx[idx]-momsES1[0]))
            )/2.
        );
    }
}

__global__ void output
(
    const int elem,
    const int modein,
    const int modeout,
    const scalar* locRho, const scalar* locT,
    const scalar* fe,
    scalar* floc,
    scalar* Q
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idx_s = (modein*${Ne}+elem)*${vsize} + idx;

    // collision frequency (non-dimensional)
    // freq = Pr*locRho/((locT)**(self._omega-1))
    if(idx<${vsize}) {
        Q[(modeout*${Ne}+elem)*${vsize} + idx] += ${prefac}*
        (locRho[0]/pow(locT[0], ${omega-1.}))*(fe[idx] - floc[idx]);
    }
}


// Summation
// From: https://github.com/inducer/pycuda/blob/master/pycuda/reduction.py
#define BLOCK_SIZE ${block_size}
#define READ_AND_MAP(i) (in[i])
#define REDUCE(a, b) ((a+b))

// seq_count and n are fixed for a given velocity mesh size
__global__ void sum_
(
  scalar* in, scalar *out,
  unsigned int seq_count, unsigned int n
)
{
  // Needs to be variable-size to prevent the braindead CUDA compiler from
  // running constructors on this array. Grrrr.
  __shared__ scalar sdata[BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*BLOCK_SIZE*seq_count + tid;
  scalar acc = 0.;
  for(unsigned int s=0; s<seq_count; ++s)
  {
    if (i >= n)
      break;

    acc = REDUCE(acc, READ_AND_MAP(i));
    i += BLOCK_SIZE;
  }

  sdata[tid] = acc;
  __syncthreads();

  #if (BLOCK_SIZE >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
  #endif
  #if (BLOCK_SIZE >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
  #endif
  #if (BLOCK_SIZE >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
  #endif
  
  if (tid < 32)
  {
    // 'volatile' required according to Fermi compatibility guide 1.2.2
    volatile scalar *smem = sdata;
    if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
    if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
    if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
    if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
    if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
    if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
  }
  
  if (tid == 0) out[blockIdx.x] = sdata[0];
}
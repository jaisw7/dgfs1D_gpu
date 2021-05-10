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
  const scalar* cx, const scalar* cy, const scalar* cz,    
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

__global__ void mom23Sk
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    const scalar* f, 
    const scalar* locRho, 
    const scalar* locUx, const scalar* locUy, const scalar* locUz,
    scalar* mom2, 
    scalar* mom3sk_x, scalar* mom3sk_y, scalar* mom3sk_z
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar Csqr;

    if(idx<${vsize}) {
      Csqr = (
             (cx[idx]-locUx[0])*(cx[idx]-locUx[0])
           + (cy[idx]-locUy[0])*(cy[idx]-locUy[0])
           + (cz[idx]-locUz[0])*(cz[idx]-locUz[0])
      );
      mom2[idx] = f[idx]*Csqr;
      mom3sk_x[idx] = f[idx]*Csqr*(cx[idx]-locUx[0]);
      mom3sk_y[idx] = f[idx]*Csqr*(cy[idx]-locUy[0]);
      mom3sk_z[idx] = f[idx]*Csqr*(cz[idx]-locUz[0]);
    }
}

__global__ void mom23SkNorm
(
    ${", ".join(["scalar* momsSk{0}".format(i) for i in range(nalphSk)])}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx==0) {

      // Normalize temperature
      momsSk4[0] *= ${cw}/(1.5*momsSk0[0]);

      // Normalize heat fluxes
      momsSk5[0] *= ${cw};
      momsSk6[0] *= ${cw};
      momsSk7[0] *= ${cw};
    }
}

__global__ void equiSkDistCompute
(
    const scalar* cx, const scalar* cy, const scalar* cz,    
    scalar* feSk, 
    ${", ".join(["scalar* momsSk{0}".format(i) for i in range(nalphSk)])}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar fe;
    
    if(idx<${vsize}) {

      fe = momsSk0[0]/pow(${math.pi}*momsSk4[0], 1.5)*Exp(
          -(
                (cx[idx]-momsSk1[0])*(cx[idx]-momsSk1[0])
              + (cy[idx]-momsSk2[0])*(cy[idx]-momsSk2[0])
              + (cz[idx]-momsSk3[0])*(cz[idx]-momsSk3[0])
            )/momsSk4[0]
        ); 

      feSk[idx] = fe*(
        1. 
        + ${(1.-Pr)/5.}*4.*(  // Factor of 4 is due to non-dimensionalization
                momsSk5[0]*(cx[idx]-momsSk1[0]) 
              + momsSk6[0]*(cy[idx]-momsSk2[0]) 
              + momsSk7[0]*(cz[idx]-momsSk3[0])
            )/(momsSk0[0]*momsSk4[0]*momsSk4[0])
            *
            (
              (
                  (cx[idx]-momsSk1[0])*(cx[idx]-momsSk1[0])
                + (cy[idx]-momsSk2[0])*(cy[idx]-momsSk2[0])
                + (cz[idx]-momsSk3[0])*(cz[idx]-momsSk3[0])
              )/momsSk4[0] - 2.5
            )
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
#include <cufft.h>

%if dtype == 'double':
    #define Cplx cufftDoubleComplex
    #define scalar double
    #define Cos cos
    #define Sin sin

    static __device__ __host__ inline double sinc(const double s)
    {
        return sin(s+1e-15)/(s+1e-15);
    }

%elif dtype == 'float':
    #define Cplx cufftComplex
    #define scalar float
    #define Cos cosf
    #define Sin sinf

    static __device__ __host__ inline float sinc(const float s)
    {
        return sin(s+1e-6)/(s+1e-6);
    }

%else:
    #error "undefined floating point data type"
%endif

<%!
import math 
%>

/*
static __device__ __host__ inline Cplx CplxMul(Cplx a, Cplx b) {
  Cplx c; c.x = a.x * b.x - a.y*b.y; c.y = a.x*b.y + a.y*b.x; return c;
}

static __device__ __host__ inline Cplx CplxAdd(Cplx a, Cplx b) {
  Cplx c; c.x = a.x + b.x; c.y = a.y + b.y; return c;
}
*/

__global__ void precompute_aa
(
    const scalar* d_lx,
    const scalar* d_ly,
    const scalar* d_lz,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if(idx<${vsize}) 
    {
        % for p in range(Nrho):
            <% fac =  math.pi/L*qz[p]/2. %>

            % for q in range(M):
                id = ${(p*M+q)*vsize}+idx;
                out[id] = ${fac}*((${sz[q,0]}*d_lx[idx]) 
                    + (${sz[q,1]}*d_ly[idx]) + (${sz[q,2]}*d_lz[idx])
                );
            % endfor
        % endfor
    }
}

__global__ void precompute_bb
(
    const scalar* d_lx,
    const scalar* d_ly,
    const scalar* d_lz,
    scalar* d_bb1,
    scalar* d_bb2
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar cSqr = 0., d_bb2_sum = 0.;

    if(idx<${vsize}) 
    {
        cSqr = sqrt(d_lx[idx]*d_lx[idx]
            + d_ly[idx]*d_ly[idx]
            + d_lz[idx]*d_lz[idx]);

        % for p in range(Nrho):
            id = ${p*vsize}+idx;
            d_bb1[id] = ${pow(qz[p],gamma+2.)*4*math.pi}
                        *sinc(${math.pi/L*qz[p]/2.}*cSqr);

            d_bb2_sum += ${qw[p]*pow(qz[p],gamma+2.)*16*math.pi*math.pi}
                        *sinc(${math.pi/L*qz[p]}*cSqr);
        % endfor

        d_bb2[idx] = d_bb2_sum;
    }
}

__global__ void cosSinMul
(
    const scalar* a,
    Cplx* FTf,
    Cplx* t1,
    Cplx* t2
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    scalar cosa, sina;

    //extern __shared__ Cplx sFTf[];
    if(idx<${vsize}) 
    {
        // scale
        FTf[idx].x /= ${vsize}; FTf[idx].y /= ${vsize};

        Cplx lFTf = FTf[idx];

        % for p in range(Nrho):
            % for q in range(M):

            id = ${(p*M+q)*vsize}+idx;
            cosa = cos(a[id]);
            sina = sin(a[id]);

            t1[id].x = cosa*lFTf.x;
            t1[id].y = cosa*lFTf.y;

            t2[id].x = sina*lFTf.x;
            t2[id].y = sina*lFTf.y;

            % endfor
        % endfor
    }
}

__global__ void magSqr
(
    const Cplx* in1,
    const Cplx* in2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    Cplx lin1, lin2;

    if(idx<${vsize}) 
    {
        % for p in range(Nrho):
            % for q in range(M):

                id = ${(p*M+q)*vsize}+idx;
                lin1 = in1[id];
                lin2 = in2[id];
                out[id].x = lin1.x*lin1.x - lin1.y*lin1.y
                            + lin2.x*lin2.x - lin2.y*lin2.y;
                out[id].y = 2*lin1.x*lin1.y + 2*lin2.x*lin2.y;

            % endfor
        % endfor
    }
}


__global__ void computeQG
(
    const scalar *d_bb1,
    const Cplx* d_t1,
    Cplx* d_QG
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    scalar intw_p = 0;

    Cplx d_QG_sum;
    d_QG_sum.x = 0.; d_QG_sum.y = 0.; 

    //int id;
    if(idx<${vsize}) {
        
        //d_QG[idv].x = 0;
        //d_QG[idv].y = 0;
        
        % for p in range(Nrho):

            intw_p = ${2.*qw[p]*sw/vsize}*d_bb1[${p*vsize}+idx];

            % for q in range(M):

                //id = ${(p*M+q)*vsize}+idv;
                d_QG_sum.x += intw_p*d_t1[${(p*M+q)*vsize}+idx].x;
                d_QG_sum.y += intw_p*d_t1[${(p*M+q)*vsize}+idx].y;

            % endfor
        % endfor

        d_QG[idx] = d_QG_sum;
    }
}

__global__ void ax
(
    const scalar* d_bb2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) {
        out[idx].x *= d_bb2[idx];
        out[idx].y *= d_bb2[idx];
    }
}

__global__ void scale
(
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) {
        out[idx].x /= ${vsize};
        out[idx].y /= ${vsize};
    }
}

__global__ void scale_MN
(
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;

    if(idx<${vsize}) 
    {
        % for p in range(Nrho):
            % for q in range(M):

                id = ${(p*M+q)*vsize}+idx;
                out[id].x /= ${vsize};
                out[id].y /= ${vsize};

            % endfor
        % endfor
    }
}



// Addition for the vhs-gll-nodal version

__global__ void cosMul
(
    const scalar* a,
    Cplx* FTf,
    Cplx* FTg,
    Cplx* t1,
    Cplx* t2
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    scalar cosa;

    //extern __shared__ Cplx sFTf[];
    if(idx<${vsize}) 
    {
        // scale
        FTf[idx].x /= ${vsize}; FTf[idx].y /= ${vsize};
        FTg[idx].x /= ${vsize}; FTg[idx].y /= ${vsize};

        Cplx lFTf = FTf[idx];
        Cplx lFTg = FTg[idx];

        % for p in range(Nrho):
            % for q in range(M):

            id = ${(p*M+q)*vsize}+idx;
            cosa = cos(a[id]);

            t1[id].x = cosa*lFTf.x;
            t1[id].y = cosa*lFTf.y;

            t2[id].x = cosa*lFTg.x;
            t2[id].y = cosa*lFTg.y;

            % endfor
        % endfor
    }
}

// Note the scaling is performed in the cosMul
__global__ void sinMul
(
    const scalar* a,
    Cplx* FTf,
    Cplx* FTg,
    Cplx* t1,
    Cplx* t2
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    scalar sina;

    //extern __shared__ Cplx sFTf[];
    if(idx<${vsize}) 
    {
        // scale
        //FTf[idx].x /= ${vsize}; FTf[idx].y /= ${vsize};
        //FTg[idx].x /= ${vsize}; FTg[idx].y /= ${vsize};

        Cplx lFTf = FTf[idx];
        Cplx lFTg = FTg[idx];

        % for p in range(Nrho):
            % for q in range(M):

            id = ${(p*M+q)*vsize}+idx;
            sina = sin(a[id]);

            t1[id].x = sina*lFTf.x;
            t1[id].y = sina*lFTf.y;

            t2[id].x = sina*lFTg.x;
            t2[id].y = sina*lFTg.y;

            % endfor
        % endfor
    }
}


__global__ void cplxMul
(
    const Cplx* in1,
    const Cplx* in2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    Cplx lin1, lin2;

    if(idx<${vsize}) 
    {
        % for p in range(Nrho):
            % for q in range(M):

                id = ${(p*M+q)*vsize}+idx;
                lin1 = in1[id];
                lin2 = in2[id];
                out[id].x = lin1.x*lin2.x - lin1.y*lin2.y;
                out[id].y = lin1.x*lin2.y + lin1.y*lin2.x;

            % endfor
        % endfor
    }
}

__global__ void cplxMulAdd
(
    const Cplx* in1,
    const Cplx* in2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int idv = idx%${vsize};
    int id;
    Cplx lin1, lin2;

    if(idx<${vsize}) 
    {
        % for p in range(Nrho):
            % for q in range(M):

                id = ${(p*M+q)*vsize}+idx;
                lin1 = in1[id];
                lin2 = in2[id];
                out[id].x += lin1.x*lin2.x - lin1.y*lin2.y;
                out[id].y += lin1.x*lin2.y + lin1.y*lin2.x;
            % endfor
        % endfor
    }
}

__global__ void r2z_
(
    const int elem,
    const int mode,
    const scalar* in,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_s = (mode*${Ne}+elem)*${vsize} + idx; 

    if(idx<${vsize}) {
        out[idx].x = in[idx_s];
        out[idx].y = 0.;
    }
}

__global__ void output_append_
(
    const int elem,
    const int modein,
    const int modeout,
    const Cplx* in_1,
    const Cplx* in_2,
    scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) {
        out[(modeout*${Ne}+elem)*${vsize} + idx] += ${prefac}*(
            in_1[idx].x - in[(modein*${Ne}+elem)*${vsize} + idx]*in_2[idx].x);
        //out[(elem*${Ne}+modeout)*${vsize} + idx] += ${prefac}*(in_1[idx].x - in[(modein*${Ne}+elem)*${vsize} + idx]*in_2[idx].x);
    }
}


// Summation
// From: https://github.com/inducer/pycuda/blob/master/pycuda/reduction.py
#define BLOCK_SIZE ${block_size}
#define READ_AND_MAP(i) (in[i].x)
//#define REDUCE(a, b) ((a+b))
#define REDUCE(a, b) ((a)>(b) ? (a): (b))

// seq_count and n are fixed for a given velocity mesh size
__global__ void sumCplx_
(
  Cplx* in, scalar *out,
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

#undef READ_AND_MAP
#define READ_AND_MAP(i) (in[i])
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



__global__ void nu
(
    const int elem,
    const int mode,
    const scalar* in,
    scalar* nu
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_s = (mode*${Ne}+elem); 

    if(idx<=1) {
        //nu[idx_s] += ${prefac}*in[idx]*${cw};
        //nu[idx_s] += in[idx]*${cw};
        //nu[idx_s] += ${prefac}*in[idx];
        nu[idx_s] += in[idx];
    }
}



__global__ void nu2
(
    const int elem,
    const int modeout,
    const Cplx* in,
    scalar* nu
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) {
        nu[(modeout*${Ne}+elem)*${vsize} + idx] += ${prefac}*in[idx].x;
    }
}

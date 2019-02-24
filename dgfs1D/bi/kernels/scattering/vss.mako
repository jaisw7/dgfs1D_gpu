#include <cufft.h>

%if dtype == 'double':
    #define Cplx cufftDoubleComplex
    #define scalar double
    #define Cos cos
    #define Sin sin
%elif dtype == 'float':
    #define Cplx cufftComplex
    #define scalar float
    #define Cos cosf
    #define Sin sinf
%else:
    #error "undefined floating point data type"
%endif

<%!
import math
%>

//- May be some other day 
//__device__ __constant__ scalar aa[${Nrho*M*vsize}];
//__device__ __constant__ scalar b[${Nrho*vsize}];
//__device__ __constant__ scalar bb2_00[${vsize}];
//__device__ __constant__ scalar bb2_01[${vsize}];
//__device__ __constant__ scalar bb2_10[${vsize}];
//__device__ __constant__ scalar bb2_11[${vsize}];

__global__ void precompute_a
(
    const int* lx, const int* ly, const int* lz,
    scalar* a
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global thread identifier
    if(idx<${vsize}) // check if idx is in range {0,...,N^3}
    {
        // loop over all the points in the radial direction
        % for x in range(Nrho):
            <% fac=math.pi/L*qz[x] %>
            // loop over all points on the full sphere
            % for y in range(M):
                // compute step 8 of Algo. (1)
                a[${(x*M+y)*vsize}+idx] = ${fac}*(${sz[y,0]}*lx[idx]
                    + ${sz[y,1]}*ly[idx] + ${sz[y,2]}*lz[idx]);
            % endfor
        % endfor
    }
}

% for p, q in cases: 
<% 
    pq = str(p) + str(q)
    mpq = masses[q]/(masses[p]+masses[q])
    etapq = eta[pq]
%>
__global__ void ${"precompute_bc_{0}{1}".format(p,q)}
(
    const int x, const int y,
    const scalar fac, const scalar fac_b, const scalar fac_c,
    const int* lx, const int* ly, const int* lz,
    const scalar* szx, const scalar* szy, const scalar* szz,
    ${"Cplx* b_{0}{1}".format(p,q)},
    ${"Cplx* c_{0}{1}".format(p,q)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar L_dot_szpre = 0., Bpq = 0.;
    ${"Cplx b_{0}{1}s, c_{0}{1}s;".format(p,q)}
    ${"b_{0}{1}s.x= 0; b_{0}{1}s.y= 0.;".format(p,q)}
    ${"c_{0}{1}s.x= 0; c_{0}{1}s.y= 0.;".format(p,q)}
    if(idx<${vsize}) 
    {
        id = (x*${M}+y)*${vsize}+idx;
        
        % for ypre in range(Mpre):
            Bpq = pow(1+szx[y]*${szpre[ypre,0]} 
                    + szy[y]*${szpre[ypre,1]} 
                    + szz[y]*${szpre[ypre,2]}, ${etapq}); 
            L_dot_szpre = fac*((${szpre[ypre,0]}*lx[idx])+(${szpre[ypre,1]}*ly[idx])+(${szpre[ypre,2]}*lz[idx]));
            ${"b_{0}{1}s.x".format(p,q)} += Bpq*fac_b*Cos(${mpq}*L_dot_szpre);
            ${"b_{0}{1}s.y".format(p,q)} += -Bpq*fac_b*Sin(${mpq}*L_dot_szpre);
            ${"c_{0}{1}s.x".format(p,q)} += Bpq*fac_c*Cos(L_dot_szpre);
            ${"c_{0}{1}s.y".format(p,q)} += -Bpq*fac_c*Sin(L_dot_szpre);
        %endfor
        ${"b_{0}{1}[id].x = b_{0}{1}s.x;".format(p,q)}
        ${"b_{0}{1}[id].y = b_{0}{1}s.y;".format(p,q)}
        ${"c_{0}{1}[idx].x += c_{0}{1}s.x;".format(p,q)}
        ${"c_{0}{1}[idx].y += c_{0}{1}s.y;".format(p,q)}
    }
}
%endfor


__global__ void r2z
(
    const int Ne,
    const int elem,
    const int mode,
    const scalar* in,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_s = (mode*Ne+elem)*${vsize} + idx; 

    if(idx<${vsize}) {
        out[idx].x = in[idx_s];
        out[idx].y = 0.;
    }
}

% for p, q in cases: 
<% mp, mq = masses[p], masses[q] %>
__global__ void ${"cosSinMul_{p}{q}".format(p=p, q=q)}
(
    const scalar* a,
    Cplx* FTf, Cplx* FTg,
    Cplx* t1, Cplx* t2
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar cosa, sina;

    if(idx<${vsize}) 
    {
        // scale FTf, FTg (normalization factor of CUFFT)
        FTf[idx].x /= ${vsize}; FTf[idx].y /= ${vsize};
        FTg[idx].x /= ${vsize}; FTg[idx].y /= ${vsize};

        Cplx lFTf = FTf[idx], lFTg = FTg[idx];

        % for x in range(Nrho):
            % for y in range(M):
            id = ${(x*M+y)*vsize}+idx;

            cosa = Cos(a[id]*(${mq/(mp+mq)}));
            sina = Sin(a[id]*(${mq/(mp+mq)}));
            t1[id].x = cosa*lFTf.x - sina*lFTf.y;
            t1[id].y = sina*lFTf.x + cosa*lFTf.y;

            cosa = Cos(a[id]*(${mp/(mp+mq)}));
            sina = Sin(a[id]*(${mp/(mp+mq)}));
            t2[id].x = cosa*lFTg.x + sina*lFTg.y;
            t2[id].y = -sina*lFTg.x + cosa*lFTg.y;
            % endfor
        % endfor
    }
}
%endfor

__global__ void prod
(
    const Cplx* in1,
    const Cplx* in2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    Cplx lin1, lin2;

    if(idx<${vsize}) 
    {
        % for x in range(Nrho):
            % for y in range(M):

                id = ${(x*M+y)*vsize}+idx;
                lin1 = in1[id];
                lin2 = in2[id];
                out[id].x = lin1.x*lin2.x - lin1.y*lin2.y;
                out[id].y = lin1.x*lin2.y + lin1.y*lin2.x;

            % endfor
        % endfor
    }
}

% for p, q in cases: 
<% 
mp, mq = masses[p], masses[q] 
gammapq = gamma[str(p)+str(q)]
%>
__global__ void ${"computeQG_{p}{q}".format(p=p, q=q)}
(
    const Cplx* b,
    const Cplx* t1,
    Cplx* QG
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar intw_x = 0;

    Cplx QG_sum;
    QG_sum.x = 0; QG_sum.y = 0;

    if(idx<${vsize}) {        
        % for x in range(Nrho):
            // scaled by ${vsize} to take care of CUFFT normalization
            intw_x = ${qw[x]*sw/vsize};
            % for y in range(M):
                id = ${(x*M+y)*vsize}+idx;
                QG_sum.x += intw_x*(t1[id].x*b[id].x - t1[id].y*b[id].y);
                QG_sum.y += intw_x*(t1[id].x*b[id].y + t1[id].y*b[id].x);
            % endfor
        % endfor
    }
    QG[idx] = QG_sum;
}
%endfor

__global__ void computeQG
(
    const Cplx* b,
    const Cplx* t1,
    Cplx* QG
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;
    scalar intw_x = 0;

    Cplx QG_sum;
    QG_sum.x = 0; QG_sum.y = 0;

    if(idx<${vsize}) {        
        % for x in range(Nrho):
            // scaled by ${vsize} to take care of CUFFT normalization
            intw_x = ${qw[x]*sw/vsize};
            % for y in range(M):
                id = ${(x*M+y)*vsize}+idx;
                QG_sum.x += intw_x*(t1[id].x*b[id].x - t1[id].y*b[id].y);
                QG_sum.y += intw_x*(t1[id].x*b[id].y + t1[id].y*b[id].x);
            % endfor
        % endfor
    }
    QG[idx] = QG_sum;
}

__global__ void ax2
(
    const Cplx* in1,
    const Cplx* in2,
    Cplx* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //out[idx].x = bb2[idx];
    //out[idx].y = bb2[idx];

    if(idx < ${vsize}) {
        Cplx lin1 = in1[idx];
        Cplx lin2 = in2[idx];
        out[idx].x = lin1.x*lin2.x - lin1.y*lin2.y;
        out[idx].y = lin1.x*lin2.y + lin1.y*lin2.x;
    }
}

% for p, q in cases: 
<% 
bpq = prefac[str(p)+str(q)];
%>
__global__ void ${"output_{p}{q}".format(p=p, q=q)}
(
    const int Ne,
    const int elem,
    const int mode,
    const int modeout,
    const Cplx* in_1,
    const Cplx* in_2,
    scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<${vsize}) {
        // when j is 0, d_Qij = Q (fi, fj), else d_Qij += Q (fi, fj)
        out[(modeout*Ne+elem)*${vsize} + idx] += 
        ${bpq}*(in_1[idx].x - in[(mode*Ne+elem)*${vsize} + idx]*in_2[idx].x);
    }
}
%endfor
// clang-format off
#define scalar ${dtype}
// clang-format on

__global__ void uflux(const scalar *uL, const scalar *uR, const scalar *cvx,
                      scalar *jL, scalar *jR) {
  // Flux splitting
  /*
    cvP = cvx_pos (i.e., positive cvx)
    cvM = cvx_neg (i.e., negative cvx)
    diff = uR - uL
    jL, jR = -cvP * diff, cvM * diff
  */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ${vsize}) {
    const scalar cvP = cvx[idx] > 0 ? cvx[idx] : 0;
    const scalar cvM = cvx[idx] < 0 ? cvx[idx] : 0;

    % for t in range(len(mapL)):
        jL[${t*vsize} + idx] = (uR[${t*vsize}+idx] - uL[${t*vsize}+idx]);

    jR[${t * vsize} + idx] = jL[${t * vsize} + idx];

    jL[${t * vsize} + idx] *= cvM;
    jR[${t * vsize} + idx] *= -cvP;
    % endfor
  }
}

__global__ void splitFlux(const scalar *cvx, const scalar *u, scalar *f,
                          scalar *g) {
  // Compute the upwind derivative in each element
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < ${K * Ne * vsize}) {
    const scalar cv = cvx[idx % ${vsize}];
    f[idx] = cv > 0 ? (cv * u[idx]) : 0;
    g[idx] = cv < 0 ? (cv * u[idx]) : 0;
  }
}
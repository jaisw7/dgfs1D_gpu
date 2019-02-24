### Discontinuous Galerkin Fast Spectral (DGFS) in one dimension
This is a research code, written, so as to demonstrate that one can solve the full Boltzmann in ~6 seconds for common standard rarefied 1D problems such as couette flow.

The codebase consists of the single/multi species high order discontinuous Galerkin solver utilizing Fast fourier transform for evaluating the collision operator. 

From a DG perspective, we use two methodologies: 

* the collocation penalty variant (as per Hasthaven 1998, Hasthaven 2008)
* the classical modal variant (as per Karniadakis 1999)

By exploiting the sparse structure of the underlying matrices using SVD decomposition, the scheme has the complexity of \sum_{m} R_m MN^4 log(N), where R_m is the rank of "constant" matrix H (see DGFS). As a direct consequence of SVD decomposition, the collocation scheme results in a scheme with a complexity of K MN^4 log(N), where K is the number of coefficients in the local element expansion. For classical modal basis, \sum_{m} R_m < K^2, while for the modified orthogonal basis (see Karniadakis 1999), \sum_{m} R_m = K^2. Note that the code automatically identifies the sparsity, and adjusts the complexity of the scheme depending on the basis in use. Other basis can be straightforwardly incorporated by introducing a new class in basis.py. 

From a time integration perspective, we use: 

* 1st order forward euler
* 2nd and 3rd order Strong Stability Preserving Runge Kutta 
* 4th order 5 stages Low storage Runge Kutta scheme

Other integration schmes can be incorporated by introducing a new class in integrator.py in std and/or bi folder. For steady state problems, the euler integration works reasonably well. 

From a fast Fourier spectral perspective, needed for evaluating the collision integral, we use the method described in (Gamba 2017, Jaiswal 2019a, Jaiswal 2019b) -- simply because the method applies straightforwardly to general collision kernels, and the results can be "directly" compared against DSMC without need of any recalibration or parametric fitting. 

The primary scheme involves the explicit evaluation of the convective and collision terms, since, it's theoretically difficult to resolve the collision kernel implicitly. One can however, consider, evaluating the convective term implicitly and collision term explicitly, and seeking a steady state by an iterative update procedure, however, the convergence of such semi-implicit schemes would be hampered by the explicit evaluation of the collision operator, requiring need of larger velocity grids to resolve the non-linearity of the collision operator. 

By the virtue of the design of DGFS (methodology and software), it is fairly straightforward to extend DGFS to multi-species cases (for example, one can run a diff on std.py and bi.py). The entire code structure remains similar for the single-species, and multi-species implementation, as well as for explicit and semi-implicit variants. Extension from single species to multi-species is straightforward, and so is the extension from explicit to semi-implicit. 

It is understandable that the codebase can be made more compact!

### Examples
* The sparta folder 

### Parametric study
#### Single species

* Effect of number of elements on the solution
  ```bash
  for i in $(seq 2 2 8);
  do 
    dgfsStd1D run dgfs1D.ini -v mesh::Ne $i -v time-integrator::dt 0.001/$i; 
  done
  ```
  This runs the simulation for Ne={2, 4, 6, 8} elements by adjusting the time-step. 

* Effect of polynomial order on the solution
  ```bash
  for i in {3,5,7};
  do 
      dgfsStd1D run dgfs1D.ini -v basis::K $i -v time-integrator::dt 0.001/$i; 
  done
  ```
  This runs the simulation for (spatial scheme order) K={3, 5, 7} by adjusting the time-step. The order of the underlying polynomial approximation is K-1. 

* Effect of time integration on the solution
  ```bash
  for i in {euler,ssp-rk2,ssp-rk3,lesrk-45};
  do 
      dgfsStd1D run dgfs1D.ini -v time-integrator::scheme $i;
  done
  ```
  This runs the simulation for (temporal scheme order) L={1, 2, 3, 4}. 

#### Multi species

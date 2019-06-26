### 1D3V Discontinuous Galerkin Fast Spectral (DGFS)
This is a research code, written, so as to demonstrate that one can solve the full Boltzmann in ~6 seconds for common standard rarefied 1D problems such as couette/fourier/osc-couette.

The codebase consists of the single/multi species high order discontinuous Galerkin solver utilizing Fast fourier transform for evaluating the collision operator. 
<br/>  

From a DG perspective, we use two methodologies: 

* the collocation penalty variant (as per **[Hasthaven 2007]**)
* the classical modal variant (as per **[Karniadakis 1999]**)
<br/>

From a basis perspective:

> By exploiting the sparse structure of the underlying matrices using SVD decomposition, the scheme has the complexity of \sum_{m} R_m M N_rho N^3 log(N), where R_m is the rank of "constant" matrix H (see **[Jaiswal 2019a]**), where N is the number of discretization points in each velocity dimension, N_rho ~ O(N) is the number of discretization points in the radial direction needed for low-rank decomposition **[Gamba 2017]**, and M is the number of discretization points on the sphere. As a direct consequence of SVD decomposition, the collocation scheme results in a scheme with a complexity of K MN^4 log(N), where K is the number of coefficients in the local element expansion. For classical modal basis, \sum_{m} R_m < K^2, while for the modified orthogonal basis (see **[Karniadakis 1999]**), \sum_{m} R_m = K^2. Note that the code automatically identifies the sparsity, and adjusts the complexity of the scheme depending on the basis in use. Other basis can be straightforwardly incorporated by introducing a new class in *basis.py*. 
<br/>

From a time integration perspective, we use: 
>
* 1st order forward euler
* 2nd and 3rd order Strong Stability Preserving Runge Kutta 
* 4th order 5 stages Low storage Runge Kutta scheme

Other integration schmes can be incorporated by introducing a new class in integrator.py in std and/or bi folder. For steady state problems, the euler integration works reasonably well. 
<br/><br/>
  
From a fast Fourier spectral perspective:
> For evaluating the collision integral, we use the method described in (**[Gamba 2017, Jaiswal 2019a, Jaiswal 2019b]**) -- simply because the method applies straightforwardly to general collision kernels, and the results can be "directly" compared against DSMC without need of any recalibration or parametric fitting.   

<br/>
The overall DGFS method is simple from mathematical and implementation perspective; highly accurate in both physical and velocity spaces as well as time; robust, i.e. applicable for general geometry and spatial mesh; exhibits nearly linear parallel scaling; and directly applies to general collision kernels needed for high fidelity modelling. By the virtue of the design of DGFS (methodology and software), it is fairly straightforward to extend DGFS to multi-species cases (for example, one can run a diff on std.py and bi.py)    

<br/>  

> For verification and tests, we have also added BGK/ESBGK **[Mieussens 2000]** linear/kinetic scattering models as well (see examples/std/couette/bgk).

### Examples
* examples/
  * bi (binary mixtures)
      * couette (1D couette flow : VSS model)
      * fourier (1D fourier heat transfer: VSS model)
      * oscCouette (1D oscillatory couette flow: VSS model)
  * std (single species)
      * couette (1D couette flow: VHS model)
      * oscCouette (1D oscillatory couette flow: VHS model)
      * fourier (1D fourier heat transfer: Maxwell model)
      * normalshock (1D normal shock: HS model)
> For most of these cases, the [DSMC/SPARTA](https://sparta.sandia.gov/) simulation script have been made available in the corresponding folders.

### Parametric study
#### Single species: see examples/std directory

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


#### Multi species: see examples/bi directory
> Replace *dgfsStd1D* by *dgfsBi1D* in the aforementioned examples

* **Benchmark [Jaiswal 2019c]**: Performance of the solver for Couette flow test cases. The phase-space is defined using a convenient triplet notation 
**Ne/K/N^3**, which corresponds to *Ne* elements in physical space, *K* order nodal DG (equivalently Np = K − 1 order polynomial for 1-D domain), and 
**N^3** points in velocity space. n*G* (n > 1) denotes GPU/CUDA/MPI/parallel execution on n GPUs shared equally across (n/3) nodes. **Work units 
represent the total simulation time for first 52 timesteps**. Efficiency is defined as ratio (1*G*/n*G*)/n, where 1*G* and n*G* are execution-times on 
one GPU and n GPU respectively. M = 12 and N_rho = 8 is used for all cases

| Phase Space | Work Units (s) |         |         |         |         |        |        | Efficiency |       |       |        |        |        |
|:-----------:|:--------------:|:-------:|:-------:|:-------:|:-------:|:------:|:------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
|             |       1G       |    3G   |    6G   |    9G   |   12G   |   24G  |   36G  |    1G/3G   | 1G/6G | 1G/9G | 1G/12G | 1G/24G | 1G/36G |
|  72/3/20^3  |      47.58     |  16.155 |  8.339  |  5.698  |  4.392  |  2.423 |  1.774 |    0.98    |  0.95 |  0.93 |   0.9  |  0.82  |  0.84  |
|  72/3/32^3  |     126.601    |  42.616 |  21.551 |  14.563 |  11.038 |  5.784 |  4.03  |    0.99    |  0.98 |  0.97 |  0.96  |  0.91  |  0.98  |
|  72/3/48^3  |     391.943    | 131.081 |  65.913 |  44.218 |  33.513 | 17.224 | 11.621 |      1     |  0.99 |  0.98 |  0.97  |  0.95  |  1.05  |
|  72/6/20^3  |     94.682     |  31.957 |  16.197 |  10.944 |  8.331  |  4.392 |  30.79 |    0.99    |  0.97 |  0.96 |  0.95  |   0.9  |  0.96  |
|  72/6/32^3  |     253.016    |  84.834 |  42.741 |  28.697 |  21.703 | 11.158 |  7.693 |    0.99    |  0.99 |  0.98 |  0.97  |  0.94  |  1.03  |
|  72/6/48^3  |     782.343    | 261.601 | 131.217 |  87.755 |  66.009 |  33.52 | 22.509 |      1     |  0.99 |  0.99 |  0.99  |  0.97  |  1.09  |
|  216/3/20^3 |     141.754    |  47.641 |  24.033 |  16.182 |  12.326 |  6.356 |  4.388 |    0.99    |  0.98 |  0.97 |  0.96  |  0.93  |  1.01  |
|  216/3/32^3 |     378.956    | 126.853 |  63.676 |  42.636 |  32.066 | 16.295 | 11.041 |      1     |  0.99 |  0.99 |  0.98  |  0.97  |  1.07  |
|  216/3/48^3 |    1172.907    | 391.916 | 196.439 | 131.153 |  98.538 | 49.652 | 33.471 |      1     |   1   |   1   |  0.99  |  0.98  |   1.1  |
|  216/6/20^3 |     283.091    |  94.737 |  47.679 |  31.903 |  24.06  | 12.262 |  8.32  |      1     | 0.99  |  0.99 |  0.98  |  0.96  |  1.06  |
|  216/6/32^3 |     759.149    | 253.498 | 127.004 |  84.932 |  63.78  | 32.212 | 21.672 |      1     |   1   |   1   |  0.99  |  0.98  |  1.09  |
|  216/6/48^3 |    2347.099    | 783.642 |  392.47 | 261.817 | 196.552 |  98.68 | 66.018 |      1     |   1   |   1   |    1   |  0.99  |  1.11  |

**Hardware**: Serial and parallel implementations of multi-species DGFS solver are run on 15-node Brown-GPU RCAC cluster at Purdue University.
Each node is equipped with two 12-core Intel Xeon Gold 6126 CPU, and three Tesla-P100 GPU. The operating system used is 64-bit
CentOS 7.4.1708 (Core) with NVIDIA Tesla-P100 GPU accompanying CUDA driver 8.0 and CUDA runtime 8.0. The GPU has 10752 CUDA cores, 
16GB device memory, and compute capability of 6.0.The solver has been written in Python/PyCUDA and is compiled using OpenMPI 2.1.0, 
g++ 5.2.0, and nvcc 8.0.61 compiler with third level optimization flag. All the simulations are done with double precision floating point values.

### References:
* **[Karniadakis 1999]** Karniadakis, George, and Spencer Sherwin. 
  *Spectral/hp element methods for computational fluid dynamics.* Oxford University Press, 2013.
* **[Hesthaven 2007]** Hesthaven, Jan S., and Tim Warburton. 
  *Nodal discontinuous Galerkin methods: algorithms, analysis, and applications.* Springer Science & Business Media, 2007.
* **[Gamba 2017]** Gamba, I. M., Haack, J. R., Hauck, C. D., & Hu, J. (2017). 
  *A fast spectral method for the Boltzmann collision operator with general collision kernels.* SIAM Journal on Scientific Computing, 39(4), B658-B674.
* **[Jaiswal 2019a]** Jaiswal, S., Alexeenko, A. A., and Hu, J. (2019)
  *A discontinuous Galerkin fast spectral method for the full Boltzmann equation with general collision kernels.* Journal of Computational Physics 378: 178-208. https://doi.org/10.1016/j.jcp.2018.11.001
* **[Jaiswal 2019b]** Jaiswal, S., Alexeenko, A. A., and Hu, J. (2019)
  *A discontinuous Galerkin fast spectral method for the multi-species full Boltzmann equation.* Computer Methods in Applied Mechanics and Engineering 352: 56-84. https://doi.org/10.1016/j.cma.2019.04.015
* **[Jaiswal 2019c]** Jaiswal, S., Hu, J., and Alexeenko, A. A. (2019)
  *A discontinuous Galerkin fast spectral method for multi-species full Boltzmann equation on streaming multi-processors.* Proceedings of the Platform for Advanced Scientific Computing Conference (ACM PASC'19) 4:1-4:9. https://doi.org/10.1145/3324989.3325714
* **[Jaiswal 2019d]** Jaiswal, S., Pikus, A., Strongrich A., Sebastiao I. B., Hu, J., and Alexeenko, A. A. (2019)
  *Quantification of thermally-driven flows in microsystems using Boltzmann equation in deterministic and stochastic context.* preprint: https://arxiv.org/abs/1905.01385 
* **[Mieussens 2000]** Mieussens, L. (2000) 
  *Discrete-velocity models and numerical schemes for the Boltzmann-BGK equation in plane and axisymmetric geometries.* Journal of Computational Physics 162.2: 429-466.

### License:
*dgfs1D_gpu* is released as GNU GPLv2 open-source software. The intention is to keep everything transparent, and adopt the practice in early part of research career.  

Portions of the code have been derived from "Polylib" and "PyFR". Please see licenses folder for restrictions.

### Confessions:
I admit that the codebase can be made more compact!

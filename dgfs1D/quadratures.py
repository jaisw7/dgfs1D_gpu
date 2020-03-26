import numpy as np
from math import gamma, cos, pi
from collections import Iterable

# The following modules are based on the "PyFR" implementation 
# (See licences/LICENSE_PyFR)
def jacobi(n, a, b, z):
    j = [1]

    if n >= 1:
        j.append(((a + b + 2)*z + a - b) / 2)
    if n >= 2:
        apb, bbmaa = a + b, b*b - a*a

        for q in range(2, n + 1):
            qapbpq, apbp2q = q*(apb + q), apb + 2*q
            apbp2qm1, apbp2qm2 = apbp2q - 1, apbp2q - 2

            aq = apbp2q*apbp2qm1/(2*qapbpq)
            bq = apbp2qm1*bbmaa/(2*qapbpq*apbp2qm2)
            cq = apbp2q*(a + q - 1)*(b + q - 1)/(qapbpq*apbp2qm2)

            # Update
            j.append((aq*z - bq)*j[-1] - cq*j[-2])

    return j

def jacobi_diff(n, a, b, z):
    dj = [0]

    if n >= 1:
        dj.extend(jp*(i + a + b + 2)/2
                  for i, jp in enumerate(jacobi(n - 1, a + 1, b + 1, z)))

    return dj

def ortho_basis_at_py(order, p):
    jp = jacobi(order - 1, 0, 0, p)
    return [np.sqrt(i + 0.5)*p for i, p in enumerate(jp)]

def jac_ortho_basis_at_py(order, p):
    djp = jacobi_diff(order - 1, 0, 0, p)
    return [(np.sqrt(i + 0.5)*p,) for i, p in enumerate(djp)]

def ortho_basis_at(order, pts):
    if len(pts) and not isinstance(pts[0], Iterable):
        pts = [(p,) for p in pts]
        return np.array([ortho_basis_at_py(order, *p) for p in pts]).T

def jac_ortho_basis_at(order, pts):
    if len(pts) and not isinstance(pts[0], Iterable):
        pts = [(p,) for p in pts]
    J = [jac_ortho_basis_at_py(order, *p) for p in pts]
    return np.array(J).swapaxes(0, 2)

def nodal_basis_at(order, pts, epts):
    return np.linalg.solve(ortho_basis_at(order, pts), 
        ortho_basis_at(order, epts)).T

def jac_nodal_basis_at(order, pts, epts):
    return np.linalg.solve(ortho_basis_at(order, pts), 
        jac_ortho_basis_at(order, epts))


# The following modules are based on the "Polylib" implementation 
# (See licences/LICENSE_Polylib)
def jacobz(n, a, b):
    if(not n): return [] 

    z = np.zeros(n)
    dth = pi/(2.0*n)
    rlast = 0.0

    for k in range(n):
        r = -cos((2.0*k + 1.0)*dth);

        if k>=1: 
            r = 0.5*(r + rlast)

        for j in range(1, 5000):
            poly = jacobi(n, a, b, r)[-1];
            pder = jacobi_diff(n, a, b, r)[-1];

            tsum = 0.0
            for i in range(k): 
                tsum += 1.0/(r - z[i]);

            delr = -poly / (pder - tsum * poly);
            r   += delr;

            if( abs(delr) < 1e-20 ): break;

        z[k]  = r;
        rlast = r;
    return z

# Compute Gauss-Jacobi points and weights
def _zwgj(Np, a, b):
    z= jacobz(Np, a, b)
    w = jacobi_diff(Np, a, b, z)[-1]

    fac  = pow(2.0, a+b+1.0)*gamma(a+Np+1.0)*gamma(b+Np+1.0);
    fac /= gamma(Np+1.0)*gamma(a+b+Np+1.0);

    for i in range(Np): 
        w[i] = fac/(w[i]*w[i]*(1-z[i]*z[i]))
    
    return z, w

# Compute Gauss-Lobatto-Jacobi points and weights
def _zwglj(Np, a, b):
    z= np.zeros(Np)
    w = np.ones(Np)*2.0

    if Np>=1:
        z[0], z[Np-1] = -1.0, 1.0;

        z[1:-1] = jacobz(Np-2, a+1.0, b+1.0); 
        w = jacobi(Np-1, a, b, z)[-1];

        fac  = pow(2.0, a+b+1.0)*gamma(a+Np)*gamma(b+Np);
        fac /= (Np-1.0)*gamma(Np)*gamma(a+b+Np+1.0);

        w = fac/(w*w)
        w[0], w[Np-1] = w[0]*(b  + 1.0), w[Np-1]*(a + 1.0);

    return z, w

# Compute Gauss-Radau-Jacobi points and weights (z=-1)
def zwgrjm(Np, a, b):
    z= np.zeros(Np)
    w = np.ones(Np)*2.0

    if Np>=1:
        z[0] = -1.0;

        z[1:] = jacobz(Np-1, a, b+1.0); 
        w = jacobi(Np, a, b, z)[-1];

        fac  = pow(2.0, a+b)*gamma(a+Np)*gamma(b+Np);
        fac /= (b+Np)*gamma(Np)*gamma(a+b+Np+1.0);

        w = fac*(1-z)/(w*w)
        w[0] = w[0]*(b  + 1.0)

    return z, w

# Compute Gauss-Radau-Jacobi points and weights (z=+1)
def zwgrjp(Np, a, b):
    z= np.zeros(Np)
    w = np.ones(Np)*2.0

    if Np>=1:
        z[Np-1] = 1.0

        z[:-1] = jacobz(Np-1, a+1, b); 
        w = jacobi(Np, a, b, z)[-1];

        fac  = pow(2.0, a+b)*gamma(a+Np)*gamma(b+Np);
        fac /= (a+Np)*gamma(Np)*gamma(a+b+Np+1.0);

        w = fac*(1+z)/(w*w)
        w[Np-1] = w[Np-1]*(a  + 1.0)

    return z, w



"""Computation of gauss quadratures via eigenvalue decomposition. 
Ref: Orthogonal Polynomials: Computation and Approximation, Walter Gautschi"""
def rjacobi(n, a, b):
    ra, rb = np.zeros(n), np.zeros(n) 

    apbp2 = 2. + a + b;    
    ra[0] = (b-a)/apbp2;
    rb[0] = np.power(2., a+b+1)*(  
            gamma(a+1.)*gamma(b+1.)/gamma(apbp2)
        ); 
    rb[1] = (4.*(a+1.)*(b+1.)/((apbp2+1.)*apbp2*apbp2));        

    # Compute other terms        
    apbp2 += 2;
    for i in range(1, n-1):
        ra[i] = (b*b-a*a)/((apbp2-2.)*apbp2)
        rb[i+1] = (
            4.*(i+1)*(i+1+a)*(i+1+b)*(i+1+a+b)/((apbp2*apbp2-1)*apbp2*apbp2)
        );
        apbp2 += 2
        
    ra[n-1] = (b*b-a*a)/((apbp2-2.)*apbp2);
    return ra, rb

def gauss(n, ra, rb):
    scal = rb[0];

    rb[:-1] = np.sqrt(rb[1:]);
    z, V = np.linalg.eigh(np.diag(ra) + np.diag(rb[:-1],-1))
    zidx = np.argsort(z); z.sort();
    V = V[:,zidx];

    w = V[0,:];
    w = scal*(w**2);
    return z, w

def zwgj(n, a, b):
    ra, rb = rjacobi(n, a, b)
    z, w = gauss(n, ra, rb)
    return z, w

def zwglj(n, a, b):
    N = n - 2;
    z, w = rjacobi(n, a, b);

    apb1 = a + b + 1.;
    z[n-1] = (a-b)/(2.*N+apb1+1.);
    w[n-1] = (4.*(N+a+1.)*(N+b+1.)*(N+apb1)
                    /((2.*N+apb1)*np.power(2*N+apb1+1, 2.)));

    z, w = gauss(n, z, w);
    return z, w
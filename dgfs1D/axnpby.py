import numpy as np
from pycuda import compiler
import pycuda.driver as cuda
from mako.template import Template
from dgfs1D.util import np_rmap, get_kernel

axnpbysrc = """
#define scalar ${dtype}

__global__ void
axnpby(scalar* __restrict__ x0,
       ${', '.join('const scalar* __restrict__ x' + str(i)
                   for i in range(1, nv))},
       ${', '.join('scalar a' + str(i) for i in range(nv))})
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if (idx < ${size} && a0 == 0.0)
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] = ${" +".join('a{l}*x{l}[id]'.format(l=l) for l in range(1, nv))};
    % endfor
    }
    else if (idx < ${size} && a0 == 1.0)
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] += ${" +".join('a{l}*x{l}[id]'.format(l=l) for l in range(1, nv))};
    % endfor
    }
    else if (idx < ${size})
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] = ${" +".join('a{l}*x{l}[id]'.format(l=l) for l in range(nv))};
    % endfor
    }
}
"""

def get_axnpby_kerns(nv, subdims, size, dtype):
    kernsrc = Template(axnpbysrc).render(
        nv=nv, subdims=subdims, size=size, dtype=np_rmap[dtype]
    )
    kernmod = compiler.SourceModule(kernsrc)

    # for extracting left face values
    #axnpby = kernmod.get_function("axnpby")
    #axnpby.prepare([np.intp]*nv + [dtype]*nv)
    #axnpby.set_cache_config(cuda.func_cache.PREFER_L1)
    axnpby = get_kernel(kernmod, "axnpby", [np.intp]*nv + [dtype]*nv)

    return axnpby

# Second type: constant "a"

axnpbyconstsrc = """
#define scalar ${dtype}

__global__ void
axnpby_const(scalar* __restrict__ x0,
       ${', '.join('const scalar* __restrict__ x' + str(i)
                   for i in range(1, nv))}
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if (idx < ${size})
    {
    % for k in subdims:
        id = ${k*size} + idx;
        %if a[0] == 0.0
            x0[id] = ${" +".join('{0}*x{1}[id]'.format(a[l], l) for l in range(1, nv))};
        %elif a[1] == 1.0
            x0[id] += ${" +".join('{0}*x{1}[id]'.format(a[l], l) for l in range(1, nv))};        
        %else
            x0[id] = ${" +".join('{0}*x{1}[id]'.format(a[l], l) for l in range(nv))};
        %endif
    %endfor
    }
}
"""

def get_axnpby_kerns_const(nv, subdims, size, a):
    kernsrc = Template(axnpbyconstsrc).render(
        nv=nv, subdims=subdims, size=size, a=a, dtype=np_rmap[dtype]
    )
    kernmod = compiler.SourceModule(kernsrc)

    # for extracting left face values
    #axnpby = kernmod.get_function("axnpby_const")
    #axnpby.prepare([np.intp]*nv)
    #axnpby.set_cache_config(cuda.func_cache.PREFER_L1)
    axnpby = get_kernel(kernmod, "axnpby_const", [np.intp]*nv)

    return axnpby


# third type: If a is on device

axnpbysrc2 = """
#define scalar ${dtype}
__global__ void
axnpby2(scalar* __restrict__ x0,
       ${', '.join('const scalar* __restrict__ x' + str(i)
                   for i in range(1, nv))},
       ${', '.join('scalar* a' + str(i) for i in range(nv))})
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int id;

    if (idx < ${size} && a0[0] == 0.0)
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] = ${" +".join('a{l}[0]*x{l}[id]'.format(l=l) for l in range(1, nv))};
    % endfor
    }
    else if (idx < ${size} && a0[0] == 1.0)
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] += ${" +".join('a{l}[0]*x{l}[id]'.format(l=l) for l in range(1, nv))};
    % endfor
    }
    else if (idx < ${size})
    {
    % for k in subdims:
        id = ${k*size} + idx;
        x0[id] = ${" +".join('a{l}[0]*x{l}[id]'.format(l=l) for l in range(nv))};
    % endfor
    }
}
"""

def get_axnpby2_kerns(nv, subdims, size, dtype):
    kernsrc = Template(axnpbysrc2).render(
        nv=nv, subdims=subdims, size=size, dtype=np_rmap[dtype]
    )
    kernmod = compiler.SourceModule(kernsrc)
    axnpby2 = get_kernel(kernmod, "axnpby2", [np.intp]*(nv*2))
    return axnpby2
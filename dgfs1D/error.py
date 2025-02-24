# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from dgfs1D.quadratures import zwglj, nodal_basis_at
import re
import numpy as np
import h5py

def getNeNq(file):
    m = re.search(r"(s=\d+)\.(k=\d+)", file)
    if not m: raise ValueError("Improper filename")
    return int(m.group(1).split("=")[-1]), int(m.group(2).split("=")[-1])

def decode(file):
    Ne, Nq = getNeNq(file)
    try:
        with h5py.File(file, 'r') as h5f:
            dst = h5f['coeff']
            iK, iNe, iNv = dst.attrs['K'], dst.attrs['Ne'], dst.attrs['Nv']
            assert iK==Nq, "Inconsistent distribution K"
            assert iNe==Ne, "Inconsistent distribution Ne"
            data = np.array(dst[:]).reshape((Nq, Ne, -1))
    except:
        data = np.loadtxt(file, comments="#").reshape((Ne, Nq, -1)).swapaxes(0,1)
    return Ne, Nq, data

def compare(Nq, f, f2):
    _, Ne2, nvars = f2.shape
    r, w = zwglj(Nq, 0., 0.);
    r2 = r*0.5 + 0.5;
    r2 = np.array([*(-np.flip(r2)), *(r2)]);
    im = nodal_basis_at(Nq, r, r2)

    fr = np.einsum('rm,mjk->rjk', im, f)
    fr = fr.swapaxes(0,2).reshape((-1, Ne2, Nq)).swapaxes(0,2);
    f2r = f2;
    diff = np.abs(fr - f2r); # L1

    L1e = 0.5*np.einsum('q,qjk->jk', w, diff)

    if nvars>32:
        return [0, np.sum(L1e, axis=(0, 1))/Ne2/nvars]
    else:
        return np.sum(L1e, axis=0)

def compute_error(file1, file2):
    # Read in the solutions
    Ne1, Nq1, usol1 = decode(file1)
    Ne2, Nq2, usol2 = decode(file2)

    assert Ne2//Ne1 == 2, "Elements in file2 != 2 * (elements in file1)"
    assert Nq1==Nq2, "Number of quadrature points don't match"
    gerr = compare(Nq1, usol1, usol2)
    print(*gerr)


def __main__():
    ap = ArgumentParser()
    ap.add_argument('file1', type=FileType('r'), help='input file')
    ap.add_argument('file2', type=FileType('r'), help='input file')
    args = ap.parse_args()
    return compute_error(args.file1, args.file2)


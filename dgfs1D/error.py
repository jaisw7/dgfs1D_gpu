# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from dgfs1D.quadratures import zwglj, nodal_basis_at
import numpy as np
import h5py
from pathlib import Path

def decode(file):
    with h5py.File(file, 'r') as h5f:
        dst = h5f['coeff']
        Nq, Ne, Nv = dst.attrs['K'], dst.attrs['Ne'], dst.attrs['Nv']
        data = np.array(dst[:]).reshape((Nq, Ne, Nv))
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
    return np.sum(L1e, axis=(0, 1))/Ne2/nvars

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
    ap.add_argument('file1', type=Path, help='input file')
    ap.add_argument('file2', type=Path, help='input file')
    args = ap.parse_args()
    assert args.file1.exists() and args.file2.exists(), "Input files not found"
    return compute_error(args.file1, args.file2)

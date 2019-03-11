# -*- coding: utf-8 -*-

# The following modules are based on the "PyFR" implementation 
# (See licences/LICENSE_PyFR)

from ctypes import POINTER, c_int, c_double, c_float, c_void_p
import numpy as np
import ctypes
import ctypes.util
import os
import sys

def platform_libname(name):
    if sys.platform == 'darwin':
        return 'lib{0}.dylib'.format(name)
    elif sys.platform == 'win32':
        return '{0}.dll'.format(name)
    else:
        return 'lib{0}.so'.format(name)

def load_library(name):
    # If an explicit override has been given then use it
    lpath = os.environ.get('DGFS_{0}_LIBRARY_PATH'.format(name.upper()))
    if lpath:
        return ctypes.CDLL(lpath)

    # Otherwise synthesise the library name and start searching
    lname = platform_libname(name)

    # Start with system search path
    try:
        return ctypes.CDLL(lname)
    # ..and if this fails then run our own search
    except OSError:
        raise OSError('Unable to load {0}'.format(name))


# Possible CUBLAS exception types
CUBLASError = type('CUBLASError', (Exception,), {})
CUBLASNotInitialized = type('CUBLASNotInitialized', (CUBLASError,), {})
CUBLASAllocFailed = type('CUBLASAllocFailed', (CUBLASError,), {})
CUBLASInvalidValue = type('CUBLASInvalidValue', (CUBLASError,), {})
CUBLASArchMismatch = type('CUBLASArchMismatch', (CUBLASError,), {})
CUBLASMappingError = type('CUBLASMappingError', (CUBLASError,), {})
CUBLASExecutionFailed = type('CUBLASExecutionFailed', (CUBLASError,), {})
CUBLASInternalError = type('CUBLASInternalError', (CUBLASError,), {})


class CUBLASWrappers(object):
    # Possible return codes
    _statuses = {
        0x1: CUBLASNotInitialized,
        0x3: CUBLASAllocFailed,
        0x7: CUBLASInvalidValue,
        0x8: CUBLASArchMismatch,
        0xb: CUBLASMappingError,
        0xd: CUBLASExecutionFailed,
        0xe: CUBLASInternalError
    }

    def __init__(self):
        lib = load_library('cublas')

        # Type of matrix transpose operation (From header file)
        self.CUBLAS_OP_N = 0
        self.CUBLAS_OP_T = 1
        self.CUBLAS_OP_C = 2

        # Which part, lower/upper, of the matrix was filled
        self.CUBLAS_FILL_MODE_LOWER = 0
        self.CUBLAS_FILL_MODE_UPPER = 1

        # Which part, lower/upper, of the matrix was filled
        self.CUBLAS_DIAG_NON_UNIT = 0
        self.CUBLAS_DIAG_UNIT = 1

        # left/right
        self.CUBLAS_SIDE_LEFT = 0
        self.CUBLAS_SIDE_RIGHT = 1

        # cublasCreate
        self.cublasCreate = lib.cublasCreate_v2
        self.cublasCreate.argtypes = [POINTER(c_void_p)]
        self.cublasCreate.errcheck = self._errcheck

        # cublasDestroy
        self.cublasDestroy = lib.cublasDestroy_v2
        self.cublasDestroy.argtypes = [c_void_p]
        self.cublasDestroy.errcheck = self._errcheck

        # cublasSetStream
        self.cublasSetStream = lib.cublasSetStream_v2
        self.cublasSetStream.argtypes = [c_void_p, c_void_p]
        self.cublasSetStream.errcheck = self._errcheck

        # cublasDgemm
        self.cublasDgemm = lib.cublasDgemm_v2
        self.cublasDgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_double), c_void_p, c_int
        ]
        self.cublasDgemm.errcheck = self._errcheck

        # cublasSgemm
        self.cublasSgemm = lib.cublasSgemm_v2
        self.cublasSgemm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int,
            POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
            POINTER(c_float), c_void_p, c_int
        ]
        self.cublasSgemm.errcheck = self._errcheck

        # cublasDgemv
        self.cublasDgemv = lib.cublasDgemv_v2
        self.cublasDgemv.argtypes = [
            c_void_p, c_int, c_int, c_int, # handle, trans, m, n
            POINTER(c_double), c_void_p, c_int, # alpha, A, lda
            c_void_p, c_int, POINTER(c_double), # x, incx, beta
            c_void_p, c_int # y, incy
        ]
        self.cublasDgemv.errcheck = self._errcheck

        # cublasSgemv
        self.cublasSgemv = lib.cublasSgemv_v2
        self.cublasSgemv.argtypes = [
            c_void_p, c_int, c_int, c_int, # handle, trans, m, n
            POINTER(c_float), c_void_p, c_int, # alpha, A, lda
            c_void_p, c_int, POINTER(c_float), # x, incx, beta
            c_void_p, c_int # y, incy
        ]
        self.cublasSgemv.errcheck = self._errcheck

        # cublasDtrsm
        self.cublasDtrsm = lib.cublasDtrsm_v2
        self.cublasDtrsm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, 
            c_int, c_int, POINTER(c_double), 
            c_void_p, c_int, c_void_p, c_int
        ]
        self.cublasDtrsm.errcheck = self._errcheck

        # cublasStrsm
        self.cublasStrsm = lib.cublasStrsm_v2
        self.cublasStrsm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, 
            c_int, c_int, POINTER(c_float), 
            c_void_p, c_int, c_void_p, c_int
        ]
        self.cublasStrsm.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CUBLASError


class CUDACUBLASKernels(object):
    def __init__(self):
        # Load and wrap CUBLAS
        self._wrappers = CUBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.cublasCreate(self._handle)

    """
    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cublasDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cublasDestroy
        try:
            #import pycuda.autoinit
            if pycuda.autoinit.context:
                self._wrappers.cublasDestroy(self._handle)
        except TypeError:
            pass
    """

    # here we have adopted things for our specific case
    def mul(self, a, sA, b, sB, c, sC, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if sA[0] != sC[0] or sA[1] != sB[1] or sB[0] != sC[1]:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T = (B^T)*(A^T)
        # We need to transpose B => C^T = (B)*(A^T)
        m, n, k = sB[0], sA[0], sA[1]
        A, B, C = b, a, c # swap a and b

        # Do not transpose B; transpose A (because we have swapped a and b)
        opA, opB = w.CUBLAS_OP_T, w.CUBLAS_OP_N

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        #w.cublasSetStream(self._handle, queue.cuda_stream_comp.handle)
        cublasgemm(self._handle, opA, opB, m, n, k,
                    alpha_ct, A.ptr, sB[1], B.ptr, sA[1],
                    beta_ct, C.ptr, sC[1])

    
    # the original implementation
    def mulO(self, a, sA, b, sB, c, sC, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if sA[0] != sC[0] or sA[1] != sB[0] or sB[1] != sC[0]:
            raise ValueError('Incompatible matrices for out = a*b')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T = (B^T)*(A^T)
        m, n, k = sB[1], sA[0], sA[1]
        A, B, C = b, a, c # swap a and b

        # Do not transpose B; transpose A (because we have swapped a and b)
        opA, opB = w.CUBLAS_OP_N, w.CUBLAS_OP_N

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        #w.cublasSetStream(self._handle, queue.cuda_stream_comp.handle)
        cublasgemm(self._handle, opA, opB, m, n, k,
                    alpha_ct, A.ptr, sB[1], B.ptr, sA[1],
                    beta_ct, C.ptr, sC[1])


    # the original implementation
    def gemvO(self, a, sA, x, n, y, alpha=1.0, beta=0.0):
        w = self._wrappers

        # A: m x n, x = n x 1, y = m x 1

        # Ensure the matrices are compatible
        if sA[1] != n:
            raise ValueError('Incompatible matrices for out = A*x')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T = (B^T)*(A^T)
        m, n = sA[1], sA[0]

        # Transpose
        opA = w.CUBLAS_OP_T

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemv = w.cublasDgemv
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemv = w.cublasSgemv
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        #w.cublasSetStream(self._handle, queue.cuda_stream_comp.handle)
        cublasgemv(self._handle, opA, m, n,
                    alpha_ct, a.ptr, sA[1], x.ptr, 1,
                    beta_ct, y.ptr, 1)


    # computes x, R*x = B, output returned in "y"
    def trsm(self, a, sA, y, n): 
        w = self._wrappers

        # A: m x n, x = n x 1, y = m x 1
        cs = w.CUBLAS_SIDE_LEFT, 
        fm = w.CUBLAS_FILL_MODE_UPPER, 
        opA = w.CUBLAS_OP_N, 
        du = w.CUBLAS_DIAG_NON_UNIT

        # Ensure the matrices are compatible
        if sA[1] != n:
            raise ValueError('Incompatible matrices for trsm')

        # CUBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T = (B^T)*(A^T)
        m, n = sA[1], sA[0]

        # Transpose
        opA = w.CUBLAS_OP_T

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublastrsm = w.cublasDtrsm
            alpha_ct = c_double(1)
        else:
            cublastrsm = w.cublasStrsm
            alpha_ct = c_float(1)

        #w.cublasSetStream(self._handle, queue.cuda_stream_comp.handle)
        cublastrsm(self._handle, cs, fm, opA, du, m, n,
                    alpha_ct, a.ptr, sA[1], x.ptr)

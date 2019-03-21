# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p, byref
import numpy as np
import ctypes
import ctypes.util
import os
import sys
from dgfs1D.cublas import CUDACUBLASKernels
from pycuda import gpuarray
import pycuda.driver as cuda

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


# Possible CUSPARSE exception types
CUSPARSEError = type('CUSPARSEError', (Exception,), {})
CUSPARSENotInitialized = type('CUSPARSENotInitialized', (CUSPARSEError,), {})
CUSPARSEAllocFailed = type('CUSPARSEAllocFailed', (CUSPARSEError,), {})
CUSPARSEInvalidValue = type('CUSPARSEInvalidValue', (CUSPARSEError,), {})
CUSPARSEArchMismatch = type('CUSPARSEArchMismatch', (CUSPARSEError,), {})
CUSPARSEMappingError = type('CUSPARSEMappingError', (CUSPARSEError,), {})
CUSPARSEExecutionFailed = type('CUSPARSEExecutionFailed', (CUSPARSEError,), {})
CUSPARSEInternalError = type('CUSPARSEInternalError', (CUSPARSEError,), {})
CUSPARSEMatrixTypeNotSupported = type('CUSPARSEMatrixTypeNotSupported', (CUSPARSEError,), {})

# Enums

# Matrix types:
CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

# Whether or not a matrix' diagonal entries are unity:
CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

# Matrix index bases:
CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

# Operation types:
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

# Whether or not to parse elements of a dense matrix row or column-wise.
CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

# Helper function (Somehow doesn't work)
class cusparseMatDescr(ctypes.Structure):
    _fields_ = [
        ('MatrixType', ctypes.c_int),
        ('FillMode', ctypes.c_int),
        ('DiagType', ctypes.c_int),
        ('IndexBase', ctypes.c_int)
    ]

    def __init__(self, MatrixType = CUSPARSE_MATRIX_TYPE_GENERAL,
        FillMode=CUSPARSE_FILL_MODE_UPPER,
        DiagType=CUSPARSE_DIAG_TYPE_NON_UNIT,
        IndexBase=CUSPARSE_INDEX_BASE_ZERO):

        super().__init__(MatrixType, FillMode, DiagType, IndexBase)


class CUSPARSEWrappers(object):
    # Possible return codes
    _statuses = {
        0x1: CUSPARSENotInitialized,
        0x2: CUSPARSEAllocFailed,
        0x3: CUSPARSEInvalidValue,
        0x4: CUSPARSEArchMismatch,
        0x5: CUSPARSEMappingError,
        0x6: CUSPARSEExecutionFailed,
        0x7: CUSPARSEInternalError,
        0x8: CUSPARSEMatrixTypeNotSupported
    }

    def __init__(self):
        lib = load_library('cusparse')

        # cusparseCreate
        self.cusparseCreate = lib.cusparseCreate
        self.cusparseCreate.argtypes = [POINTER(c_void_p)]
        self.cusparseCreate.errcheck = self._errcheck

        # cusparseDestroy
        self.cusparseDestroy = lib.cusparseDestroy
        self.cusparseDestroy.argtypes = [c_void_p]
        self.cusparseDestroy.errcheck = self._errcheck

        # cusparseSetStream
        self.cusparseSetStream = lib.cusparseSetStream
        self.cusparseSetStream.argtypes = [c_void_p, c_void_p]
        self.cusparseSetStream.errcheck = self._errcheck

        # cusparseSetMatType
        self.cusparseSetMatType = lib.cusparseSetMatType
        self.cusparseSetMatType.argtypes = [c_void_p, c_int]
        self.cusparseSetMatType.errcheck = self._errcheck

        # cusparseSetMatFillMode
        self.cusparseSetMatFillMode = lib.cusparseSetMatFillMode
        self.cusparseSetMatFillMode.argtypes = [c_void_p, c_int]
        self.cusparseSetMatFillMode.errcheck = self._errcheck

        # cusparseSetMatDiagType
        self.cusparseSetMatDiagType = lib.cusparseSetMatDiagType
        self.cusparseSetMatDiagType.argtypes = [c_void_p, c_int]
        self.cusparseSetMatDiagType.errcheck = self._errcheck

        # cusparseSetMatIndexBase
        self.cusparseSetMatIndexBase = lib.cusparseSetMatIndexBase
        self.cusparseSetMatIndexBase.argtypes = [c_void_p, c_int]
        self.cusparseSetMatIndexBase.errcheck = self._errcheck

        # cusparseCreateMatDescr
        self.cusparseCreateMatDescr = lib.cusparseCreateMatDescr
        self.cusparseCreateMatDescr.argtypes = [POINTER(c_void_p)]
        self.cusparseCreateMatDescr.errcheck = self._errcheck

        # cusparseDestroyMatDescr
        self.cusparseDestroyMatDescr = lib.cusparseDestroyMatDescr
        self.cusparseDestroyMatDescr.argtypes = [c_void_p]
        self.cusparseDestroyMatDescr.errcheck = self._errcheck

        # cusparseDcsrmv
        self.cusparseDcsrmv = lib.cusparseDcsrmv
        self.cusparseDcsrmv.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, POINTER(c_double),  
            c_void_p, c_void_p, c_void_p, c_void_p, 
            c_void_p, POINTER(c_double), c_void_p
        ]
        self.cusparseDcsrmv.errcheck = self._errcheck

        # cusparseScsrmv
        self.cusparseScsrmv = lib.cusparseScsrmv
        self.cusparseScsrmv.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, POINTER(c_float),
            c_void_p, c_void_p, c_void_p, c_void_p, 
            c_void_p, POINTER(c_float), c_void_p
        ]
        self.cusparseScsrmv.errcheck = self._errcheck

        # cusparseDbsrmv
        self.cusparseDbsrmv = lib.cusparseDbsrmv
        self.cusparseDbsrmv.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, POINTER(c_double),  
            c_void_p, c_void_p, c_void_p, c_void_p, c_int,
            c_void_p, POINTER(c_double), c_void_p
        ]
        self.cusparseDbsrmv.errcheck = self._errcheck

        # cusparseSbsrmv
        self.cusparseSbsrmv = lib.cusparseSbsrmv
        self.cusparseSbsrmv.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, POINTER(c_float),
            c_void_p, c_void_p, c_void_p, c_void_p, c_int,
            c_void_p, POINTER(c_float), c_void_p
        ]
        self.cusparseSbsrmv.errcheck = self._errcheck

        # cusparseDcsrmm
        self.cusparseDcsrmm = lib.cusparseDcsrmm
        self.cusparseDcsrmm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, 
            POINTER(c_double), c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_int, POINTER(c_double), 
            c_void_p, c_int
        ]
        self.cusparseDcsrmm.errcheck = self._errcheck

        # cusparseScsrmm
        self.cusparseScsrmm = lib.cusparseScsrmm
        self.cusparseScsrmm.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, 
            POINTER(c_float), c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_int, POINTER(c_float), 
            c_void_p, c_int
        ]
        self.cusparseScsrmm.errcheck = self._errcheck

        # cusparseDcsrmm
        self.cusparseDcsrmm2 = lib.cusparseDcsrmm2
        self.cusparseDcsrmm2.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, 
            POINTER(c_double), c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_int, POINTER(c_double), 
            c_void_p, c_int
        ]
        self.cusparseDcsrmm.errcheck = self._errcheck

        # cusparseScsrmm
        self.cusparseScsrmm2 = lib.cusparseScsrmm2
        self.cusparseScsrmm2.argtypes = [
            c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, 
            POINTER(c_float), c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_int, POINTER(c_float), 
            c_void_p, c_int
        ]
        self.cusparseScsrmm2.errcheck = self._errcheck


    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CUSPARSEError
        

class cusparseCsrMat(object):
    def __init__(self, m, n, nnz, 
        csrVal, csrRowPtr, csrColInd):
        
        self._m, self._n, self._nnz = m, n, nnz
        self._csrVal, self._csrRowPtr = csrVal, csrRowPtr
        self._csrColInd = csrColInd

    @property
    def shape(self): return (self._m, self._n, self._nnz)

    @property
    def csrVal(self): return self._csrVal

    @property
    def csrRowPtr(self): return self._csrRowPtr

    @property
    def csrColInd(self): return self._csrColInd


class CUSPARSECache(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self): return self._data


class CUDACUSPARSEKernels(object):
    def __init__(self):
        # Load and wrap CUSPARSE
        self._wrappers = CUSPARSEWrappers()

        # Init CUSPARSE
        self._handle = c_void_p()
        self._wrappers.cusparseCreate(self._handle)

        # Load blas kernels
        self._kernels_blas = CUDACUBLASKernels()

        # scratch data cache
        self._cache = {}


    """
    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cusparseDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cusparseDestroy
        try:
            #import pycuda.autoinit
            if pycuda.autoinit.context:
                self._wrappers.cusparseDestroy(self._handle)
        except TypeError:
            pass
    """    

    def creatematdescr(self, MatrixType = CUSPARSE_MATRIX_TYPE_GENERAL,
        FillMode=CUSPARSE_FILL_MODE_UPPER,
        DiagType=CUSPARSE_DIAG_TYPE_NON_UNIT,
        IndexBase=CUSPARSE_INDEX_BASE_ZERO):

        descr = c_void_p()
        self._wrappers.cusparseCreateMatDescr(descr)
        self._wrappers.cusparseSetMatType(descr, MatrixType)
        self._wrappers.cusparseSetMatFillMode(descr, FillMode)
        self._wrappers.cusparseSetMatDiagType(descr, DiagType)
        self._wrappers.cusparseSetMatIndexBase(descr, IndexBase)

        return descr


    # sparse matrix vector multiplication
    def csrmv(self, descrA, A, x, y, alpha=1.0, beta=0.0):
        w = self._wrappers

        # A: m x n, x = n x 1, y = m x 1
        m, n, nnz = A.shape

        # Ensure the matrices are compatible
        #if sA[1] != len(x):
        #    raise ValueError('Incompatible matrices for out = A*x')

        # Transpose
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE

        # α and β factors for C = α*(A*op(B)) + β*C
        if A.csrVal.dtype == np.float64:
            cusparsecsrmv = w.cusparseDcsrmv
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cusparsecsrmv = w.cusparseScsrmv
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        #w.cusparseSetStream(self._handle, queue.cuda_stream_comp.handle)
        cusparsecsrmv(self._handle, opA, m, n, nnz,
                    alpha_ct, descrA, A.csrVal.ptr, 
                    A.csrRowPtr.ptr, A.csrColInd.ptr, x.ptr, beta_ct, y.ptr)



    # sparse matrix vector multiplication for block-csr matrices
    def bsrmv(self, bsrBlkDim, descrA, A, x, y, alpha=1.0, beta=0.0):
        w = self._wrappers

        # A: m x n, x = n x 1, y = m x 1
        m, n, nnz = A.shape

        # Ensure the matrices are compatible
        #if sA[1] != len(x):
        #    raise ValueError('Incompatible matrices for out = A*x')

        # no transpose
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE

        # storage format of blocks
        dirA = CUSPARSE_DIRECTION_ROW

        # α and β factors for C = α*(A*op(B)) + β*C
        if A.csrVal.dtype == np.float64:
            cusparsebsrmv = w.cusparseDbsrmv
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cusparsebsrmv = w.cusparseSbsrmv
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        #w.cusparseSetStream(self._handle, queue.cuda_stream_comp.handle)
        cusparsebsrmv(self._handle, dirA, opA, m, n, nnz,
                    alpha_ct, descrA, A.csrVal.ptr, 
                    A.csrRowPtr.ptr, A.csrColInd.ptr, bsrBlkDim,
                    x.ptr, beta_ct, y.ptr)


    # sparse matrix matrix multiplication for csr matrices
    def csrmm(self, descrA, a, b, sB, c, sC, alpha=1.0, beta=0.0):
        w = self._wrappers

        # (m,k) x (k,n) = (m,n)

        sA = a.shape
        m, k, nnz = a.shape
        n = sB[1]

        # Ensure the matrices are compatible
        #if sA[0] != sC[0] or sA[1] != sB[0] or sB[1] != sC[1]:
        #    raise ValueError('Incompatible matrices for out = a*b')

        # Do not transpose A
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.csrVal.dtype == np.float64:
            cusparsecsrmm = w.cusparseDcsrmm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cusparsecsrmm = w.cusparseScsrmm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        # works if b and c are in fortran major
        cusparsecsrmm(self._handle, opA, m, n, k, nnz, 
                    alpha_ct, descrA, a.csrVal.ptr, 
                    a.csrRowPtr.ptr, a.csrColInd.ptr, 
                    b.ptr, k, beta_ct, c.ptr, m)


    # sparse matrix matrix multiplication for csr matrices
    def csrmm2(self, descrA, a, b, sB, c, sC, alpha=1.0, beta=0.0):
        w = self._wrappers

        # (m,k) x (k,n) = (m,n)

        m, k, nnz = a.shape
        n = sB[1]

        # Ensure the matrices are compatible
        #if sA[0] != sC[0] or sA[1] != sB[0] or sB[1] != sC[1]:
        #    raise ValueError('Incompatible matrices for out = a*b')

        # Do not transpose A
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE
        opB = CUSPARSE_OPERATION_TRANSPOSE

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.csrVal.dtype == np.float64:
            cusparsecsrmm2 = w.cusparseDcsrmm2
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cusparsecsrmm2 = w.cusparseScsrmm2
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        cusparsecsrmm2(self._handle, opA, opB, m, n, k, nnz, 
                    alpha_ct, descrA, a.csrVal.ptr, 
                    a.csrRowPtr.ptr, a.csrColInd.ptr, 
                    b.ptr, n, beta_ct, c.ptr, m)

        # transpose
        key = hash((sC))

        if key in self._cache:            
            temp = self._cache[key].data
        else:
            temp = gpuarray.empty(sC[0]*sC[1], dtype=a.csrVal.dtype)
            self._cache[key] = CUSPARSECache(temp)

        self._kernels_blas.transpose(c, (m, n), temp)
        cuda.memcpy_dtod(c.ptr, temp.ptr, c.nbytes)

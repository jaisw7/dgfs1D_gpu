# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_float, c_void_p, byref
import numpy as np
import ctypes
import ctypes.util
import os
import sys
from cgfs1D.cublas import CUBLASWrappers
from pycuda import gpuarray

def platform_libname(name):
    if sys.platform == 'darwin':
        return 'lib{0}.dylib'.format(name)
    elif sys.platform == 'win32':
        return '{0}.dll'.format(name)
    else:
        return 'lib{0}.so'.format(name)

def load_library(name):
    # Fix for GOMP weirdness with CUDA 8.0 on Fedora (#171):
    try:
        ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
    except:
        pass
    try:
        ctypes.CDLL('libgomp.so', mode=ctypes.RTLD_GLOBAL)
    except:
        pass

    # If an explicit override has been given then use it
    lpath = os.environ.get('CGFS_{0}_LIBRARY_PATH'.format(name.upper()))
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


# Possible CUSOLVER exception types
CUSOLVERError = type('CUSOLVERError', (Exception,), {})
CUSOLVERNotInitialized = type('CUSOLVERNotInitialized', (CUSOLVERError,), {})
CUSOLVERAllocFailed = type('CUSOLVERAllocFailed', (CUSOLVERError,), {})
CUSOLVERInvalidValue = type('CUSOLVERInvalidValue', (CUSOLVERError,), {})
CUSOLVERArchMismatch = type('CUSOLVERArchMismatch', (CUSOLVERError,), {})
CUSOLVERMappingError = type('CUSOLVERMappingError', (CUSOLVERError,), {})
CUSOLVERExecutionFailed = type('CUSOLVERExecutionFailed', (CUSOLVERError,), {})
CUSOLVERInternalError = type('CUSOLVERInternalError', (CUSOLVERError,), {})
CUSOLVERMatrixTypeNotSupported = type('CUSOLVERMatrixTypeNotSupported', (CUSOLVERError,), {})
CUSOLVERNotSupported = type('CUSOLVERNotSupported', (CUSOLVERError,), {})
CUSOLVERZeroPivot = type('CUSOLVERZeroPivot', (CUSOLVERError,), {})
CUSOLVERInvalidLicense = type('CUSOLVERInvalidLicense', (CUSOLVERError,), {})

class CUSOLVERWrappers(object):
    # Possible return codes
    _statuses = {
        0x1: CUSOLVERNotInitialized,
        0x2: CUSOLVERAllocFailed,
        0x3: CUSOLVERInvalidValue,
        0x4: CUSOLVERArchMismatch,
        0x5: CUSOLVERMappingError,
        0x6: CUSOLVERExecutionFailed,
        0x7: CUSOLVERInternalError,
        0x8: CUSOLVERMatrixTypeNotSupported,
        0x9: CUSOLVERNotSupported,
        0xa: CUSOLVERZeroPivot,
        0xb: CUSOLVERInvalidLicense
    }

    def __init__(self):
        lib = load_library('cusolver')

        # Constants (transformation on matrix)
        self.CUBLAS_OP_N = 0
        self.CUBLAS_OP_T = 1
        self.CUBLAS_OP_C = 2

        # cusolverDnCreate
        self.cusolverDnCreate = lib.cusolverDnCreate
        self.cusolverDnCreate.argtypes = [POINTER(c_void_p)]
        self.cusolverDnCreate.errcheck = self._errcheck

        # cusolverDnDestroy
        self.cusolverDnDestroy = lib.cusolverDnDestroy
        self.cusolverDnDestroy.argtypes = [c_void_p]
        self.cusolverDnDestroy.errcheck = self._errcheck

        # cusolverDnSetStream
        self.cusolverDnSetStream = lib.cusolverDnSetStream
        self.cusolverDnSetStream.argtypes = [c_void_p, c_void_p]
        self.cusolverDnSetStream.errcheck = self._errcheck

        # cusolverDnDgetrf_bufferSize
        self.cusolverDnDgetrf_bufferSize = lib.cusolverDnDgetrf_bufferSize
        self.cusolverDnDgetrf_bufferSize.argtypes = [
            c_void_p, c_int, c_int, c_void_p, c_int, c_void_p
        ]
        self.cusolverDnDgetrf_bufferSize.errcheck = self._errcheck

        # cusolverDnSgetrf_bufferSize
        self.cusolverDnSgetrf_bufferSize = lib.cusolverDnSgetrf_bufferSize
        self.cusolverDnSgetrf_bufferSize.argtypes = [
            c_void_p, c_int, c_int, c_void_p, c_int, c_void_p
        ]
        self.cusolverDnSgetrf_bufferSize.errcheck = self._errcheck

        # cusolverDnDgetrf
        self.cusolverDnDgetrf = lib.cusolverDnDgetrf
        self.cusolverDnDgetrf.argtypes = [
            c_void_p, c_int, c_int, c_void_p, c_int, 
            c_void_p, c_void_p, c_void_p
        ]
        self.cusolverDnDgetrf.errcheck = self._errcheck

        # cusolverDnSgeqrf
        self.cusolverDnSgetrf = lib.cusolverDnSgetrf
        self.cusolverDnSgetrf.argtypes = [
            c_void_p, c_int, c_int, c_void_p, c_int, 
            c_void_p, c_void_p, c_void_p
        ]
        self.cusolverDnSgetrf.errcheck = self._errcheck

        # cusolverDnDgetrs
        self.cusolverDnDgetrs = lib.cusolverDnDgetrs
        self.cusolverDnDgetrs.argtypes = [
            c_void_p, c_int, c_int, c_int, c_void_p, c_int,
            c_void_p, c_void_p, c_int, c_void_p
        ]
        self.cusolverDnDgetrs.errcheck = self._errcheck

        # cusolverDnSgeqrs
        self.cusolverDnSgetrs = lib.cusolverDnSgetrs
        self.cusolverDnSgetrs.argtypes = [
            c_void_p, c_int, c_int, c_int, c_void_p, c_int,
            c_void_p, c_void_p, c_int, c_void_p
        ]
        self.cusolverDnSgetrs.errcheck = self._errcheck

        # sparse functions

        # cusolverSpCreate
        self.cusolverSpCreate = lib.cusolverSpCreate
        self.cusolverSpCreate.argtypes = [POINTER(c_void_p)]
        self.cusolverSpCreate.errcheck = self._errcheck

        # cusolverSpDestroy
        self.cusolverSpDestroy = lib.cusolverSpDestroy
        self.cusolverSpDestroy.argtypes = [c_void_p]
        self.cusolverSpDestroy.errcheck = self._errcheck

        # cusolverSpSetStream
        self.cusolverSpSetStream = lib.cusolverSpSetStream
        self.cusolverSpSetStream.argtypes = [c_void_p, c_void_p]
        self.cusolverSpSetStream.errcheck = self._errcheck

        # cusolverSpScsrlsvqr
        self.cusolverSpScsrlsvqr = lib.cusolverSpScsrlsvqr
        self.cusolverSpScsrlsvqr.argtypes = [
            c_void_p, c_int, c_int, 
            c_void_p, c_void_p, c_void_p, c_void_p, 
            c_void_p, c_float, c_int, c_void_p, POINTER(c_int)
        ]
        self.cusolverSpScsrlsvqr.errcheck = self._errcheck

        # cusolverSpDcsrlsvqe
        self.cusolverSpDcsrlsvqr = lib.cusolverSpDcsrlsvqr
        self.cusolverSpDcsrlsvqr.argtypes = [
            c_void_p, c_int, c_int, 
            c_void_p, c_void_p, c_void_p, c_void_p, 
            c_void_p, c_double, c_int, c_void_p, POINTER(c_int)
        ]
        self.cusolverSpDcsrlsvqr.errcheck = self._errcheck

        # for batched QR decomposition

        # cusolverSpCreateCsrqrInfo
        self.cusolverSpCreateCsrqrInfo = lib.cusolverSpCreateCsrqrInfo
        self.cusolverSpCreateCsrqrInfo.argtypes = [POINTER(c_void_p)]
        self.cusolverSpCreateCsrqrInfo.errcheck = self._errcheck
        #self.cusolverSpCreateCsrqrInfo.rettype = c_int

        # cusolverSpDestroyCsrqrInfo
        self.cusolverSpDestroyCsrqrInfo = lib.cusolverSpDestroyCsrqrInfo
        self.cusolverSpDestroyCsrqrInfo.argtypes = [c_void_p]
        self.cusolverSpDestroyCsrqrInfo.errcheck = self._errcheck

        # analyze the sparse structure of the matrix
        self.cusolverSpXcsrqrAnalysisBatched = (
            lib.cusolverSpXcsrqrAnalysisBatched)
        self.cusolverSpXcsrqrAnalysisBatched.argtypes = [
            c_void_p, c_int, c_int, c_int,
            c_void_p, c_void_p, c_void_p, c_void_p
        ]
        self.cusolverSpXcsrqrAnalysisBatched.errcheck = self._errcheck

        # allocate buffers for batched QR decomposition (float)
        self.cusolverSpScsrqrBufferInfoBatched = (
            lib.cusolverSpScsrqrBufferInfoBatched)
        self.cusolverSpScsrqrBufferInfoBatched.argtypes = [
            c_void_p, c_int, c_int, c_int,
            c_void_p, c_void_p, c_void_p, c_void_p
        ]
        self.cusolverSpScsrqrBufferInfoBatched.errcheck = self._errcheck

        # allocate buffers for batched QR decomposition (double)
        self.cusolverSpDcsrqrBufferInfoBatched = (
            lib.cusolverSpDcsrqrBufferInfoBatched)
        self.cusolverSpDcsrqrBufferInfoBatched.argtypes = [
            c_void_p, c_int, c_int, c_int,
            c_void_p, c_void_p, c_void_p, c_void_p
        ]
        self.cusolverSpDcsrqrBufferInfoBatched.errcheck = self._errcheck

        # CSR-QR batched (float)
        self.cusolverSpScsrqrsvBatched = lib.cusolverSpScsrqrsvBatched
        self.cusolverSpScsrqrsvBatched.argtypes = [
            c_void_p, c_int, c_int, c_int,
            c_void_p, 
            c_void_p, c_void_p, c_void_p, 
            c_void_p, c_void_p, c_int, c_void_p, c_void_p
        ]
        self.cusolverSpScsrqrsvBatched.errcheck = self._errcheck

        # CSR-QR batched (double)
        self.cusolverSpDcsrqrsvBatched = lib.cusolverSpDcsrqrsvBatched
        self.cusolverSpDcsrqrsvBatched.argtypes = [
            c_void_p, c_int, c_int, c_int,
            c_void_p, 
            c_void_p, c_void_p, c_void_p, 
            c_void_p, c_void_p, c_int, c_void_p, c_void_p
        ]
        self.cusolverSpDcsrqrsvBatched.errcheck = self._errcheck


    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CUSOLVERError


class CUSOLVERCache(object):
    def __init__(self, ipiv, buffer, info):
        self._ipiv = ipiv
        self._buffer = buffer
        self._info = info

    @property
    def ipiv(self): return self._ipiv

    @property
    def buffer(self): return self._buffer

    @property
    def info(self): return self._info


class CUSOLVERGenericCache(object):
    def __init__(self, *args):
        self._data = args

    @property
    def data(self): return self._data
        

class CUDACUSOLVERKernels(object):
    def __init__(self):
        # Load and wrap CUSOLVER
        self._wrappers = CUSOLVERWrappers()

        # Load and wrap CUBLAS
        self._wrappers_blas = CUBLASWrappers()

        # Init CUSOLVER
        self._handle = c_void_p()
        self._wrappers.cusolverDnCreate(self._handle)

        self._handle_sp = c_void_p()
        self._wrappers.cusolverSpCreate(self._handle_sp)

        # Init CUBLAS
        self._handle_blas = c_void_p()
        self._wrappers_blas.cublasCreate(self._handle_blas)

        # scratch data cache 
        self._cache = {}

        # scratch data cache for batched QR decomposition
        self._cache_QR = {}

    """
    def __del__(self):
        # PyCUDA registers an atexit handler to destroy the CUDA context
        # when Python terminates; however in exceptional circumstances this
        # can be *before* we are garbage collected (negating the need to call
        # cusolverDestroy as we're terminating anyway).  We therefore need to
        # check for a valid context before calling cusolverDestroy
        try:
            #import pycuda.autoinit
            if pycuda.autoinit.context:
                self._wrappers.cusolverDnDestroy(self._handle)
        except TypeError:
            pass
    """
    
    # Solve Ax=b via LU decomposition: Faster than QRF for small matrices
    def solveLU(self, A, sA, b, n):
        # Tested just for square matrices
        w, wb = self._wrappers, self._wrappers_blas

        # Ensure the matrices are compatible
        if sA[1] != n:
            raise ValueError('Incompatible matrices for solve(A, b)')

        # CUSOLVER expects inputs to be column-major (or Fortran order in
        # numpy parlance). 
        m, n = sA[0], sA[1]

        if A.dtype == np.float64:
            cusolvergetrf_buffersize = w.cusolverDnDgetrf_bufferSize
            cusolvergetrf = w.cusolverDnDgetrf
            cusolvergetrs = w.cusolverDnDgetrs
        else:
            cusolversgetrf_buffersize = w.cusolverDnSgetrf_bufferSize
            cusolvergetrf = w.cusolverDnSgetrf
            cusolvergetrs = w.cusolverDnSgeStrs

        key = hash((sA, n))

        if key in self._cache:
            
            item = self._cache[key]
            #ipiv, buffer, info = item.ipiv, item.buffer, item.info
            ipiv, buffer, info = item.ipiv, item.buffer, item.info

        else:

            bufferSize = c_int(0)

            # Query working space of geqrf and ormqr
            #w.cusolverSetStream(self._handle, queue.cuda_stream_comp.handle)
            cusolvergetrf_buffersize(self._handle, n, n, A.ptr, m, 
                byref(bufferSize))

            ipiv = gpuarray.empty(n, dtype=np.int)
            buffer = gpuarray.empty(bufferSize.value, dtype=A.dtype)
            info = gpuarray.zeros(1, dtype=np.int)

            self._cache[key] = CUSOLVERCache(ipiv, buffer, info)


        # compute LU decomposition
        cusolvergetrf(self._handle, n, n, A.ptr, m, 
            buffer.ptr, ipiv.ptr, info.ptr)

        # compute 
        cusolvergetrs(self._handle, w.CUBLAS_OP_T, n, 1, A.ptr, m, ipiv.ptr, 
            b.ptr, n, info.ptr)



    # Solve Ax=b via LU decomposition: Faster than QRF for small matrices
    def solveQRSparse(self, descrA, a, b, x, tol=1e-6, reorder=0):
        # Works for square matrices only
        w = self._wrappers

        # Ensure the matrices are compatible
        #if sA[1] != n:
        #    raise ValueError('Incompatible matrices for solve(A, b)')

        # CUSOLVER expects inputs to be column-major (or Fortran order in
        # numpy parlance). 
        m, n, nnz = a.shape

        if a.csrVal.dtype == np.float64:
            cusolverspcsrlsvqr = w.cusolverSpDcsrlsvqr
            tol_ct = c_double(tol)
        else:
            cusolverspcsrlsvqr = w.cusolverSpScsrlsvqr
            tol_ct = c_float(tol)

        reorder_ct, ret = c_int(reorder), c_int(0)

        cusolverspcsrlsvqr(self._handle_sp, n, nnz, 
                    descrA, a.csrVal.ptr, a.csrRowPtr.ptr, a.csrColInd.ptr, 
                    b.ptr, tol_ct, reorder_ct, x.ptr, ret)


    # solve Ax=b batched QR decomposition

    # utility for creating struct for CSR QR decompostion information
    def createcsrqrinfo(self):
        info = c_void_p()
        stat = self._wrappers.cusolverSpCreateCsrqrInfo(info)
        return info

    # Solve Ax=b via QR decomposition
    def solveBatchedQRSparse(self, bSz, infoA, descrA, a, b, x):
        # Wroks for square matrices only
        w = self._wrappers

        # Ensure the matrices are compatible
        #if sA[1] != n:
        #    raise ValueError('Incompatible matrices for solve(A, b)')

        # CUSOLVER expects inputs to be column-major (or Fortran order in
        # numpy parlance). 
        m, n, nnzA = a.shape

        key = hash((m, n, nnzA, bSz))

        if key in self._cache_QR:
            
            item = self._cache_QR[key]
            qr_buffer, = item.data

        else:

            sz_internal_bytes, sz_qr_bytes = c_int(0), c_int(0)

            if a.csrVal.dtype == np.float64:
                spcsrqrBufferInfoBatched = w.cusolverSpDcsrqrBufferInfoBatched
            else:
                spcsrqrBufferInfoBatched = w.cusolverSpScsrqrBufferInfoBatched

            # perform an analysis on the matrix
            w.cusolverSpXcsrqrAnalysisBatched(self._handle_sp, n, n, nnzA,
                descrA, a.csrRowPtr.ptr, a.csrColInd.ptr, infoA)
            
            # Query working space of csrqrBatched
            #w.cusolverSetStream(self._handle, queue.cuda_stream_comp.handle)
            spcsrqrBufferInfoBatched(self._handle_sp, n, n, nnzA,
                descrA, a.csrVal.ptr, a.csrRowPtr.ptr, a.csrColInd.ptr,
                bSz, infoA, byref(sz_internal_bytes), byref(sz_qr_bytes))

            # normalize workspace size
            #sz_qr = sz_qr_bytes.value//a.csrVal.itemsize
            #qr_buffer = gpuarray.empty(sz_qr, dtype=a.csrVal.dtype)

            sz_qr = sz_qr_bytes.value//np.dtype(np.intp).itemsize
            qr_buffer = gpuarray.empty(sz_qr, dtype=np.intp)
            
            self._cache_QR[key] = CUSOLVERGenericCache(qr_buffer)


        if a.csrVal.dtype == np.float64:
            cusolverspcsrqrsvBatched = w.cusolverSpDcsrqrsvBatched
        else:
            cusolverspcsrqrsvBatched = w.cusolverSpScsrqrsvBatched

        cusolverspcsrqrsvBatched(self._handle_sp, n, n, nnzA, 
                    descrA, a.csrVal.ptr, a.csrRowPtr.ptr, a.csrColInd.ptr, 
                    b.ptr, x.ptr, bSz, infoA, qr_buffer.ptr)



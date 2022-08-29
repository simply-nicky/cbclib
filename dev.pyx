#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.math cimport ceil, exp, sqrt, atan2, acos, fabs, sin, cos, log
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def kernel_regression(np.ndarray y not None, np.ndarray x not None, np.ndarray x_hat, double sigma, double cutoff,
                      double epsilon=1e-12, unsigned int num_threads=1):
    y = check_array(y, np.NPY_FLOAT64)
    x = check_array(x, np.NPY_FLOAT64)
    x_hat = check_array(x_hat, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef np.npy_intp ndim = x.shape[x.ndim - 1]
    cdef np.npy_intp npts = x.size / ndim, nhat = x_hat.size / ndim

    if x.shape[x.ndim - 1] != x_hat.shape[x_hat.ndim - 1]:
        raise ValueError('`x` and `x_hat` have incompatible shapes')
    if npts != y.size:
        raise ValueError('`x` and `y` have incompatible shapes')

    cdef np.ndarray y_hat = <np.ndarray>np.PyArray_SimpleNew(x_hat.ndim - 1, x_hat.shape, np.NPY_FLOAT64)
    cdef double *_y_hat = <double *>np.PyArray_DATA(y_hat)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef double *_x = <double *>np.PyArray_DATA(x)
    cdef double *_x_hat = <double *>np.PyArray_DATA(x_hat)

    with nogil:
        fail = predict_kerreg(_y, _x, npts, ndim, _y_hat, _x_hat, nhat, rbf, sigma, cutoff, epsilon, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return y_hat
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.string cimport memcmp
from libc.math cimport ceil, exp, sqrt, atan, atan2, acos, fabs, sin, cos, log, floor
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

cdef line_profile profiles[3]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2}

build_profiles()

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7
DEF CUTOFF = 3.0

def searchsorted(np.ndarray a not None, np.ndarray v not None, str side='left'):
    if not np.PyArray_IS_C_CONTIGUOUS(a):
        a = np.PyArray_GETCONTIGUOUS(a)
    if not np.PyArray_IS_C_CONTIGUOUS(v):
        v = np.PyArray_GETCONTIGUOUS(v)

    cdef int type_num = np.PyArray_TYPE(a)
    if np.PyArray_TYPE(v) != type_num:
        raise ValueError('Input arrays have incompatible data types')
    
    cdef int i, i_end = v.size
    cdef unsigned long _asize = a.size
    cdef void *_a = np.PyArray_DATA(a)
    cdef void *_v = np.PyArray_DATA(v)
    cdef np.ndarray out = np.PyArray_SimpleNew(v.ndim, v.shape, np.NPY_UINT64)
    cdef unsigned long *_out = <unsigned long *>np.PyArray_DATA(out)
    cdef int _side = side_to_code(side)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(double), _a, _asize, sizeof(double), _side, compare_double)
        elif type_num == np.NPY_FLOAT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(float), _a, _asize, sizeof(float), _side, compare_float)
        elif type_num == np.NPY_INT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(int), _a, _asize, sizeof(int), _side, compare_int)
        elif type_num == np.NPY_UINT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(unsigned int), _a, _asize, sizeof(unsigned int), _side, compare_uint)
        elif type_num == np.NPY_UINT64:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(unsigned long), _a, _asize, sizeof(unsigned long), _side, compare_ulong)
        else:
            raise TypeError(f'a argument has incompatible type: {str(a.dtype)}')
    return out

def xtal_interpolate(np.ndarray xidx not None, np.ndarray xmap not None,
                     np.ndarray xtal not None, unsigned num_threads=1):
    if xidx.size != xmap.shape[0]:
        raise ValueError('Input arrays have incompatible shapes')
    
    xidx = check_array(xidx, np.NPY_UINT32)
    xmap = check_array(xmap, np.NPY_FLOAT32)
    xtal = check_array(xtal, np.NPY_FLOAT32)

    cdef int fail = 0
    cdef unsigned long *_ddims = <unsigned long *>xtal.shape
    cdef unsigned long _isize = xidx.size

    cdef np.ndarray xtal_bi = np.PyArray_SimpleNew(1, xidx.shape, np.NPY_FLOAT32)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef unsigned *_xidx = <unsigned *>np.PyArray_DATA(xidx)
    cdef float *_xmap = <float *>np.PyArray_DATA(xmap)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)

    with nogil:
        fail = xtal_interp(_xtal_bi, _xidx, _xmap, _xtal, _ddims, _isize, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return xtal_bi
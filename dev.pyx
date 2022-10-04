#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.string cimport memcmp
from libc.math cimport ceil, exp, sqrt, atan2, acos, fabs, sin, cos, log
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

cdef line_profile profiles[3]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2}

build_profiles()

def draw_line_stack(np.ndarray inp not None, dict lines not None, int max_val=1, double dilation=0.0,
                     str profile='tophat', unsigned int num_threads=1):
    if inp.ndim < 2:
        raise ValueError('Input array must be >=2D array.')
    inp = check_array(inp, np.NPY_UINT32)

    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef int ndim = inp.ndim
    cdef unsigned int *_inp = <unsigned int *>np.PyArray_DATA(inp)
    cdef int _X = <int>inp.shape[ndim - 1]
    cdef int _Y = <int>inp.shape[ndim - 2]
    cdef int repeats = inp.size / _X / _Y

    cdef int i, N = len(lines)
    cdef list frames = list(lines)
    cdef dict _lines = {}
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long **_ldims = <unsigned long **>malloc(N * sizeof(unsigned long *))
    for i in range(N):
        _lines[i] = check_array(lines[frames[i]], np.NPY_FLOAT32)
        _lptrs[i] = <float *>np.PyArray_DATA(_lines[i])
        _ldims[i] = <unsigned long *>(<np.ndarray>_lines[i]).shape

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = draw_line_c(_inp + i * _Y * _X, _Y, _X, max_val, _lptrs[i], _ldims[i], <float>dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    free(_lptrs); free(_ldims)

    return inp

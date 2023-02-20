#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.string cimport memcmp, memcpy
from libc.math cimport ceil, exp, sqrt, atan, atan2, acos, fabs, sin, cos, log, floor
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

cdef line_profile profiles[4]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile
    profiles[3] = gauss_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2, 'gauss': 3}

build_profiles()

cdef loss_func lfuncs[3]
cdef loss_func gfuncs[3]
cdef void build_loss():
    lfuncs[0] = l2_loss
    lfuncs[1] = l1_loss
    lfuncs[2] = huber_loss
    gfuncs[0] = l2_grad
    gfuncs[1] = l1_grad
    gfuncs[2] = huber_grad

cdef dict loss_scheme = {'l2': 0, 'l1': 1, 'huber': 2}

build_loss()

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

def refine_pattern(np.ndarray inp not None, object lines not None, float dilation,
                   str profile='gauss', unsigned int num_threads=1):
    if inp.ndim != 3:
        raise ValueError('Input array must be a 3D array.')
    inp = check_array(inp, np.NPY_FLOAT32)

    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef float *_inp = <float *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>inp.shape + 1
    cdef int repeats = inp.size / _dims[0] / _dims[1]

    cdef np.ndarray out = np.PyArray_SimpleNew(3, inp.shape, np.NPY_FLOAT32)
    cdef float *_out = <float *>np.PyArray_DATA(out)

    cdef int i, N = len(lines)
    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))
    for i in range(N):
        arr = check_array(lines[i], np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")
        _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
        _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = refine_line(_lptrs[i], _inp + i * _dims[0] * _dims[1], _dims,
                           _lptrs[i], _ldims + 2 * i, dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    cdef dict line_dict = {}
    for i in range(repeats):
        arr = ArrayWrapper.from_ptr(<void *>_lptrs[i]).to_ndarray(2, <np.npy_intp *>(_ldims + 2 * i), np.NPY_FLOAT32)
        line_dict[i] = arr

    free(_lptrs); free(_ldims)

    return line_dict

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

def draw_lines(np.ndarray inp not None, np.ndarray lines not None, int max_val=255, double dilation=0.0, str profile='tophat'):
    inp = check_array(inp, np.NPY_UINT32)
    lines = check_array(lines, np.NPY_FLOAT32)

    if inp.ndim != 2:
        raise ValueError("Input array must be two-dimensional")
    if lines.ndim != 2 or lines.shape[1] < 5:
        raise ValueError("lines array has an incompatible shape")
    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef unsigned int *_inp = <unsigned int *>np.PyArray_DATA(inp)
    cdef unsigned long _Y = inp.shape[0]
    cdef unsigned long _X = inp.shape[1]
    cdef float *_lines = <float *>np.PyArray_DATA(lines)
    cdef unsigned long *_ldims = <unsigned long *>lines.shape

    cdef line_profile _prof = profiles[profile_scheme[profile]]
    cdef int fail

    with nogil:
        fail = draw_lines_c(_inp, _Y, _X, max_val, _lines, _ldims, <float>dilation, _prof)
    if fail:
        raise RuntimeError('C backend exited with error.')    
    return inp
